import { useMemo } from "react";
import { getScalingGroupName } from "../utils/scaling";
import {
  computeLinearRegression,
  computePowerLawWithOffset,
  predictWithPowerLaw,
  type PowerLawResult,
  type RegressionResult,
} from "../utils/regression";

const MIN_SCALING_LEADERBOARD_FLOPS = 3e20;
const FORECAST_TARGET_FLOPS = 1e22;
const LOG_FORECAST_TARGET = Math.log(FORECAST_TARGET_FLOPS);
const REGRESSION_POINTS = 60;
const RELATIVE_RATIO_EPS = 1e-12;
const FLOP_MATCH_REL_EPS = 1e-12;
// Large flop counts lose precision in floating point, so treat near-equal values as identical.
function flopCountsMatch(a: number, b: number) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    return false;
  }
  const scale = Math.max(Math.abs(a), Math.abs(b), 1);
  return Math.abs(a - b) <= scale * FLOP_MATCH_REL_EPS;
}

type AxisX = "training_hardware_flops" | "model_flops";
type AxisY = "absolute" | "relative";

interface Run {
  run_name: string;
  author: {
    name: string;
    url: string;
    affiliation?: string;
  };
  eval_paloma_c4_en_bpb: number;
  training_hardware_flops: number;
  model_flops: number;
  model_size: number;
  training_time?: number;
  wandb_link?: string;
  results_filepath: string;
  run_completion_timestamp: string;
}

interface Track {
  id: string;
  name: string;
  color: string;
  target_bpb?: number;
}

interface DataPoint {
  x: number;
  y: number;
  name: string;
  inTrack?: boolean;
  trainingFlops?: number;
  modelFlops?: number;
  modelSize?: number;
  tokens?: number;
  bpb?: number;
}

interface ScalingGroupMeta {
  author: string;
  authorUrl: string;
  affiliation?: string;
  wandb?: string;
  filepath: string;
  date: string;
  // Power law parameters: L = intercept * x^(-slope) + asymptote
  intercept?: number | null; // scaling coefficient
  slope?: number | null; // scaling exponent
  asymptote?: number | null; // irreducible loss floor
  r2?: number | null;
  projected?: number | null;
}

interface ScalingGroup {
  name: string;
  data: DataPoint[];
  regression: Array<{ x: number; y: number; name: string }>;
  forecastedBpbAt1e22: number;
  leaderboard?: ScalingGroupMeta | null;
}

interface StandardChartData {
  type: "standard";
  all: DataPoint[];
  highlighted: DataPoint[];
  pareto: DataPoint[];
  yDomain: [number | string, number | string];
}

interface ScalingChartData {
  type: "scaling";
  background: DataPoint[];
  groups: ScalingGroup[];
  yDomain: [number | string, number | string];
  top3Groups: Set<string>;
}

export type ChartData = StandardChartData | ScalingChartData;

export interface StandardLeaderboardRow {
  type: "standard";
  rank: number;
  name: string;
  author: string;
  authorUrl: string;
  affiliation?: string;
  modelSize: number;
  trainingTime: number;
  flops: number;
  bpb: number;
  wandb?: string;
  filepath: string;
  date: string;
}

export interface ScalingLeaderboardRow extends ScalingGroupMeta {
  type: "scaling";
  name: string;
}

export type LeaderboardRow = StandardLeaderboardRow | ScalingLeaderboardRow;

interface UseSpeedrunDataParams {
  runs: Run[];
  filteredRuns: Run[];
  trackId: string;
  currentTrack?: Track;
  tracks: Track[];
  xAxis: AxisX;
  yAxis: AxisY;
}

interface SpeedrunDataResult {
  chartData: ChartData;
  nextLower: number | null;
  xTicks: number[];
  leaderboardRows: LeaderboardRow[];
}

interface BaselineModel {
  predict: (x: number) => number | null;
  regression: PowerLawResult | null;
}

interface FormattedRun {
  run: Run;
  absolutePoint: DataPoint;
  relativePoint: DataPoint;
}

const NO_BASELINE: BaselineModel = {
  predict: () => null,
  regression: null,
};

export function useSpeedrunData({
  runs,
  filteredRuns,
  trackId,
  currentTrack,
  tracks,
  xAxis,
  yAxis,
}: UseSpeedrunDataParams): SpeedrunDataResult {
  const baseline = useMemo(
    () => buildBaselineModel(runs, xAxis),
    [runs, xAxis]
  );

  const formattedRuns = useMemo(
    () =>
      filteredRuns.map((run) => formatRunMetrics(run, xAxis, baseline.predict)),
    [baseline, filteredRuns, xAxis]
  );

  const absolutePoints = useMemo(
    () => formattedRuns.map((entry) => entry.absolutePoint),
    [formattedRuns]
  );

  const processedPoints = useMemo(() => {
    if (yAxis === "absolute") return absolutePoints;
    return formattedRuns.map((entry) => entry.relativePoint);
  }, [absolutePoints, formattedRuns, yAxis]);

  const paretoPoints = useMemo(
    () => computeParetoFront(absolutePoints),
    [absolutePoints]
  );
  const yDomain = useMemo(
    () => computeYDomain(processedPoints, yAxis),
    [processedPoints, yAxis]
  );

  const chartData = useMemo<ChartData>(() => {
    if (trackId === "scaling") {
      return buildScalingChart({
        formattedRuns,
        yAxis,
        background: processedPoints,
        yDomain,
      });
    }

    return {
      type: "standard",
      all: processedPoints,
      highlighted: processedPoints,
      pareto:
        yAxis === "relative"
          ? paretoPoints.map((p) => ({ ...p, y: 0 }))
          : paretoPoints,
      yDomain,
    };
  }, [paretoPoints, processedPoints, formattedRuns, trackId, yAxis, yDomain]);

  const nextLower = useMemo(() => {
    if (
      trackId === "all" ||
      trackId === "scaling" ||
      !currentTrack?.target_bpb
    ) {
      return null;
    }

    const sortedTracks = tracks
      .filter((t) => t.id !== "all" && t.target_bpb)
      .sort((a, b) => (b.target_bpb || 0) - (a.target_bpb || 0));

    const idx = sortedTracks.findIndex((t) => t.id === trackId);
    return idx < sortedTracks.length - 1
      ? sortedTracks[idx + 1].target_bpb || 0
      : 0;
  }, [trackId, currentTrack, tracks]);

  const xTicks = useMemo(() => buildXTicks(chartData), [chartData]);

  const leaderboardRows = useMemo<LeaderboardRow[]>(() => {
    if (trackId === "scaling" && chartData.type === "scaling") {
      return chartData.groups
        .map((group) => {
          if (!group.leaderboard) return null;

          const hasQualifyingRun = group.data.some(
            (point) =>
              (point.trainingFlops ?? 0) > MIN_SCALING_LEADERBOARD_FLOPS
          );
          if (!hasQualifyingRun) return null;

          return {
            type: "scaling" as const,
            name: group.name,
            ...group.leaderboard,
          };
        })
        .filter((row): row is ScalingLeaderboardRow => Boolean(row));
    }

    return [...filteredRuns]
      .sort((a, b) => a.eval_paloma_c4_en_bpb - b.eval_paloma_c4_en_bpb)
      .map((run, idx) => ({
        type: "standard" as const,
        rank: idx + 1,
        name: run.run_name,
        author: run.author.name,
        authorUrl: run.author.url,
        affiliation: run.author.affiliation,
        modelSize: run.model_size,
        trainingTime: run.training_time,
        flops: run.training_hardware_flops,
        bpb: run.eval_paloma_c4_en_bpb,
        wandb: run.wandb_link,
        filepath: run.results_filepath,
        date: run.run_completion_timestamp,
      }));
  }, [chartData, filteredRuns, trackId]);

  return {
    chartData,
    nextLower,
    xTicks,
    leaderboardRows,
  };
}

function createBasePoint(run: Run, xAxis: AxisX): DataPoint {
  const tokens =
    run.training_hardware_flops && run.model_flops
      ? Math.round(run.training_hardware_flops / (6 * run.model_flops))
      : undefined;

  return {
    x: run[xAxis],
    y: run.eval_paloma_c4_en_bpb,
    name: run.run_name,
    inTrack: true,
    trainingFlops: run.training_hardware_flops,
    modelFlops: run.model_flops,
    modelSize: run.model_size,
    tokens,
    bpb: run.eval_paloma_c4_en_bpb,
  };
}

function buildBaselineModel(runs: Run[], xAxis: AxisX): BaselineModel {
  const baselineRuns = runs
    .filter((r) => r.run_name.toLowerCase().includes("adamw_llama_scaling"))
    .map((r) => ({ x: r[xAxis], y: r.eval_paloma_c4_en_bpb }))
    .filter((point) => point.x > 0 && point.y > 0)
    .sort((a, b) => a.x - b.x);

  if (baselineRuns.length === 0) {
    return NO_BASELINE;
  }

  // Fit power law with irreducible loss: L = a * x^(-b) + c
  const regression = computePowerLawWithOffset(baselineRuns);

  const findBaselineMatch = (x: number) =>
    baselineRuns.find((point) => flopCountsMatch(point.x, x)) ?? null;

  return {
    regression,
    predict: (x: number) => {
      if (x <= 0) return null;
      const exactMatch = findBaselineMatch(x);
      if (exactMatch) {
        return exactMatch.y;
      }
      const first = baselineRuns[0];
      const last = baselineRuns[baselineRuns.length - 1];

      // For values within the observed range, interpolate
      if (x >= first.x && x <= last.x) {
        for (let i = 0; i < baselineRuns.length - 1; i++) {
          const left = baselineRuns[i];
          const right = baselineRuns[i + 1];
          if (x >= left.x && x <= right.x) {
            // Log-space interpolation for smoother results
            const logX = Math.log(x);
            const logLeft = Math.log(left.x);
            const logRight = Math.log(right.x);
            const t = (logX - logLeft) / (logRight - logLeft);
            return left.y + t * (right.y - left.y);
          }
        }
      }

      // For extrapolation (beyond observed range), use the power law fit
      if (regression) {
        return predictWithPowerLaw(regression, x);
      }

      // Fallback: clamp to nearest observed value
      if (x < first.x) return first.y;
      return last.y;
    },
  };
}

function getRelativePercent(
  value: number,
  x: number,
  predictBaseline: (x: number) => number | null
): number {
  const baselineValue = predictBaseline(x);
  if (!baselineValue || baselineValue <= 0 || value <= 0) {
    return 0;
  }

  const ratio = value / baselineValue;
  const normalizedRatio = Math.abs(1 - ratio) < RELATIVE_RATIO_EPS ? 1 : ratio;
  return (normalizedRatio - 1) * 100;
}

function formatRunMetrics(
  run: Run,
  xAxis: AxisX,
  predictBaseline: (x: number) => number | null
): FormattedRun {
  const absolutePoint = createBasePoint(run, xAxis);
  const relativePercent = getRelativePercent(
    absolutePoint.y,
    absolutePoint.x,
    predictBaseline
  );

  return {
    run,
    absolutePoint,
    relativePoint: {
      ...absolutePoint,
      y: relativePercent,
    },
  };
}

function computeParetoFront(points: DataPoint[]) {
  const sorted = [...points].sort((a, b) => a.x - b.x);
  const pareto: DataPoint[] = [];
  let best = Infinity;

  for (const point of sorted) {
    if (point.y < best) {
      best = point.y;
      pareto.push(point);
    }
  }

  return pareto;
}

function computeYDomain(
  points: DataPoint[],
  yAxis: AxisY
): [number | string, number | string] {
  if (yAxis === "relative" || points.length === 0) {
    return ["auto", "auto"];
  }

  const values = points.map((point) => point.y);
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return ["auto", "auto"];
  }

  const range = Math.max(max - min, Math.abs(max) * 0.1);
  const margin = range * 0.05;
  return [min - margin, max + margin];
}

function buildScalingChart({
  formattedRuns,
  yAxis,
  background,
  yDomain,
}: {
  formattedRuns: FormattedRun[];
  yAxis: AxisY;
  background: DataPoint[];
  yDomain: [number | string, number | string];
}): ScalingChartData {
  const groupsByName = new Map<string, FormattedRun[]>();
  formattedRuns.forEach((entry) => {
    const name = getScalingGroupName(entry.run.run_name);
    if (!groupsByName.has(name)) {
      groupsByName.set(name, []);
    }
    groupsByName.get(name)!.push(entry);
  });

  const groups = Array.from(groupsByName.entries())
    .map(([name, groupEntries]) =>
      createScalingGroup(name, groupEntries, yAxis)
    )
    .filter((group): group is ScalingGroup => Boolean(group));

  const qualified = groups.filter((group) =>
    group.data.some(
      (point) => (point.trainingFlops ?? 0) > MIN_SCALING_LEADERBOARD_FLOPS
    )
  );

  const sorted = qualified.sort(
    (a, b) => a.forecastedBpbAt1e22 - b.forecastedBpbAt1e22
  );

  const top3Groups = new Set(sorted.slice(0, 3).map((group) => group.name));
  const backgroundPoints = background.filter((point) => point.inTrack);

  return {
    type: "scaling",
    background: backgroundPoints,
    groups: sorted,
    yDomain,
    top3Groups,
  };
}

function createScalingGroup(
  name: string,
  entries: FormattedRun[],
  yAxis: AxisY
): ScalingGroup | null {
  const data = entries
    .map((entry) =>
      yAxis === "relative" ? entry.relativePoint : entry.absolutePoint
    )
    .sort((a, b) => a.x - b.x);

  if (data.length === 0) {
    return null;
  }

  // Prepare samples for power law fitting (need raw x,y values)
  const absoluteSamples = entries
    .filter((entry) => entry.absolutePoint.x > 0 && entry.absolutePoint.y > 0)
    .map((entry) => ({ x: entry.absolutePoint.x, y: entry.absolutePoint.y }));

  // Fit power law with offset: L = intercept * x^(-slope) + asymptote
  const absoluteRegression = computePowerLawWithOffset(absoluteSamples);

  // For relative view, fit a linear regression in log-log space on relative values
  const relativeSamples = entries
    .filter(
      (entry) =>
        entry.relativePoint.x > 0 && Number.isFinite(entry.relativePoint.y)
    )
    .map((entry) => ({
      x: Math.log(entry.relativePoint.x),
      y: entry.relativePoint.y, // Already in percent, keep as-is for linear fit
    }));
  const relativeRegression =
    relativeSamples.length >= 2
      ? computeLinearRegression(relativeSamples)
      : null;

  const firstPositive = data.find((point) => point.x > 0);
  let regression: Array<{ x: number; y: number; name: string }>;

  if (yAxis === "relative" && relativeRegression && firstPositive) {
    // Use linear regression for relative view
    regression = buildLinearRegressionLine(
      relativeRegression,
      firstPositive.x,
      FORECAST_TARGET_FLOPS,
      name
    );
  } else if (absoluteRegression && firstPositive) {
    // Use power law for absolute view
    regression = buildPowerLawRegressionLine(
      absoluteRegression,
      firstPositive.x,
      FORECAST_TARGET_FLOPS,
      name
    );
  } else {
    regression = data.map((p) => ({ x: p.x, y: p.y, name }));
  }

  // Forecast at target FLOPs using the power law
  const forecasted = absoluteRegression
    ? predictWithPowerLaw(absoluteRegression, FORECAST_TARGET_FLOPS)
    : Infinity;

  const firstRun = entries[0].run;
  const leaderboard = firstRun
    ? {
        author: firstRun.author.name,
        authorUrl: firstRun.author.url,
        affiliation: firstRun.author.affiliation,
        wandb: firstRun.wandb_link,
        filepath: firstRun.results_filepath.split("/").slice(0, -1).join("/"),
        date: firstRun.run_completion_timestamp,
        intercept: absoluteRegression?.intercept ?? null,
        slope: absoluteRegression?.slope ?? null,
        asymptote: absoluteRegression?.asymptote ?? null,
        r2: absoluteRegression?.r2 ?? null,
        projected: Number.isFinite(forecasted) ? forecasted : null,
      }
    : null;

  return {
    name,
    data,
    regression,
    forecastedBpbAt1e22: forecasted,
    leaderboard,
  };
}

/**
 * Build regression line for power law fit (absolute view)
 */
function buildPowerLawRegressionLine(
  regression: PowerLawResult,
  minX: number,
  maxX: number,
  name: string
) {
  const safeMin = Math.max(minX, 1);
  const start = Math.log(safeMin);
  const end = Math.log(Math.max(maxX, safeMin * 1.01));
  const points: Array<{ x: number; y: number; name: string }> = [];

  for (let i = 0; i < REGRESSION_POINTS; i++) {
    const t = i / (REGRESSION_POINTS - 1);
    const logX = start + t * (end - start);
    const x = Math.exp(logX);
    // Use power law: L = intercept * x^(-slope) + asymptote
    const y = predictWithPowerLaw(regression, x);
    points.push({ x, y, name });
  }

  return points;
}

/**
 * Build regression line for linear fit in log-x space (relative view)
 */
function buildLinearRegressionLine(
  regression: RegressionResult,
  minX: number,
  maxX: number,
  name: string
) {
  const safeMin = Math.max(minX, 1);
  const start = Math.log(safeMin);
  const end = Math.log(Math.max(maxX, safeMin * 1.01));
  const points: Array<{ x: number; y: number; name: string }> = [];

  for (let i = 0; i < REGRESSION_POINTS; i++) {
    const t = i / (REGRESSION_POINTS - 1);
    const logX = start + t * (end - start);
    const x = Math.exp(logX);
    // Linear regression: y = intercept + slope * log(x)
    const y = regression.intercept + regression.slope * logX;
    points.push({ x, y, name });
  }

  return points;
}

function buildXTicks(chartData: ChartData) {
  const xValues: number[] = [];
  if (chartData.type === "scaling") {
    chartData.groups.forEach((group) => {
      group.data.forEach((point) => xValues.push(point.x));
    });
  } else {
    chartData.all.forEach((point) => xValues.push(point.x));
  }

  if (xValues.length === 0) return [];

  const minX = Math.min(...xValues);
  const maxX = FORECAST_TARGET_FLOPS;
  const minExp = Math.floor(Math.log10(minX));
  const maxExp = Math.ceil(Math.log10(maxX));

  const ticks: number[] = [];
  for (let exp = minExp; exp <= maxExp; exp++) {
    ticks.push(Math.pow(10, exp));
    ticks.push(Math.pow(10, exp + 0.5));
  }

  return ticks.filter((tick) => tick >= minX * 0.8 && tick <= maxX * 1.2);
}
