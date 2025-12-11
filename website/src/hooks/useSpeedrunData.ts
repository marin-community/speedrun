import { useMemo } from 'react';
import { getScalingGroupName } from '../utils/scaling';
import { computeLinearRegression } from '../utils/regression';

const MIN_SCALING_LEADERBOARD_FLOPS = 3e20; // Require at least this much compute to appear in the scaling leaderboard

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
  intercept?: number | null;
  slope?: number | null;
  r2?: number | null;
  projected?: number | null;
}

interface ScalingGroup {
  name: string;
  data: Array<
    DataPoint & {
      logX: number;
      logAbsoluteY: number;
      logRelativeY?: number;
    }
  >;
  regression: Array<{ x: number; y: number; name: string }>;
  forecastedBpbAt1e22: number;
  leaderboard?: ScalingGroupMeta | null;
}

interface StandardChartData {
  type: 'standard';
  all: DataPoint[];
  highlighted: DataPoint[];
  pareto: DataPoint[];
  yDomain: [number | string, number | string];
}

interface ScalingChartData {
  type: 'scaling';
  background: DataPoint[];
  groups: ScalingGroup[];
  yDomain: [number | string, number | string];
  top3Groups: Set<string>;
}

export type ChartData = StandardChartData | ScalingChartData;

export interface StandardLeaderboardRow {
  type: 'standard';
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
  type: 'scaling';
  name: string;
}

export type LeaderboardRow = StandardLeaderboardRow | ScalingLeaderboardRow;

interface UseSpeedrunDataParams {
  runs: Run[];
  filteredRuns: Run[];
  trackId: string;
  currentTrack?: Track;
  tracks: Track[];
  xAxis: 'training_hardware_flops' | 'model_flops';
  yAxis: 'absolute' | 'relative';
}

interface SpeedrunDataResult {
  chartData: ChartData;
  nextLower: number | null;
  xTicks: number[];
  leaderboardRows: LeaderboardRow[];
}

export function useSpeedrunData({
  runs,
  filteredRuns,
  trackId,
  currentTrack,
  tracks,
  xAxis,
  yAxis
}: UseSpeedrunDataParams): SpeedrunDataResult {
  const nextLower = useMemo(() => {
    if (trackId === 'all' || trackId === 'scaling' || !currentTrack?.target_bpb) {
      return null;
    }

    const sortedTracks = tracks
      .filter(t => t.id !== 'all' && t.target_bpb)
      .sort((a, b) => (b.target_bpb || 0) - (a.target_bpb || 0));

    const idx = sortedTracks.findIndex(t => t.id === trackId);
    return idx < sortedTracks.length - 1 ? sortedTracks[idx + 1].target_bpb || 0 : 0;
  }, [trackId, currentTrack, tracks]);

  const chartData = useMemo<ChartData>(() => {
    const allPoints: DataPoint[] = filteredRuns.map(r => {
      const tokens =
        r.training_hardware_flops && r.model_flops
          ? Math.round(r.training_hardware_flops / (6 * r.model_flops))
          : undefined;

      return {
        x: r[xAxis],
        y: r.eval_paloma_c4_en_bpb,
        name: r.run_name,
        inTrack: true,
        trainingFlops: r.training_hardware_flops,
        modelFlops: r.model_flops,
        modelSize: r.model_size,
        tokens,
        bpb: r.eval_paloma_c4_en_bpb
      };
    });

    const baselineRuns = runs.filter(r => r.run_name.toLowerCase().includes('adamw'));
    const baselinePoints = baselineRuns
      .map(r => ({
        x: r[xAxis],
        y: r.eval_paloma_c4_en_bpb,
        name: r.run_name
      }))
      .sort((a, b) => a.x - b.x);

    const baselineRegressionData = baselinePoints
      .filter(point => point.x > 0 && point.y > 0)
      .map(point => ({
        x: Math.log(point.x),
        y: Math.log(point.y)
      }));
    const baselineRegression =
      baselineRegressionData.length >= 2 ? computeLinearRegression(baselineRegressionData) : null;

    const paretoPoints: typeof allPoints = [];
    const sorted = [...allPoints].sort((a, b) => a.x - b.x);
    let minY = Infinity;

    for (const point of sorted) {
      if (point.y < minY) {
        minY = point.y;
        paretoPoints.push(point);
      }
    }

    const getBaselineValue = (x: number, fallback: number) => {
      if (baselinePoints.length === 0) {
        return fallback;
      }

      const firstBaseline = baselinePoints[0];
      const lastBaseline = baselinePoints[baselinePoints.length - 1];

      if (x <= firstBaseline.x || x >= lastBaseline.x) {
        if (baselineRegression && x > 0) {
          const logX = Math.log(x);
          return Math.exp(baselineRegression.intercept + baselineRegression.slope * logX);
        }
        return x <= firstBaseline.x ? firstBaseline.y : lastBaseline.y;
      }

      for (let i = 0; i < baselinePoints.length - 1; i++) {
        const p1 = baselinePoints[i];
        const p2 = baselinePoints[i + 1];
        if (x >= p1.x && x <= p2.x) {
          const logX = Math.log(x);
          const logX1 = Math.log(p1.x);
          const logX2 = Math.log(p2.x);
          const t = (logX - logX1) / (logX2 - logX1);
          return p1.y + t * (p2.y - p1.y);
        }
      }

      return fallback;
    };

    const getRelativeMetrics = (value: number, x: number) => {
      const baselineBPB = getBaselineValue(x, value);
      if (baselineBPB <= 0 || value <= 0) {
        return { percent: 0, logRatio: undefined };
      }
      const ratio = value / baselineBPB;
      return {
        percent: (ratio - 1) * 100,
        logRatio: Math.log(ratio)
      };
    };

    const convertToRelative = (value: number, x: number) => {
      return getRelativeMetrics(value, x).percent;
    };

    const processedPoints =
      yAxis === 'relative'
        ? allPoints.map(point => ({
            ...point,
            y: convertToRelative(point.y, point.x)
          }))
        : allPoints;

    let yDomain: [number | string, number | string] = ['auto', 'auto'];
    if (yAxis === 'absolute' && processedPoints.length > 0) {
      const allYValues = processedPoints.map(p => p.y);
      const minYValue = Math.min(...allYValues);
      const maxYValue = Math.max(...allYValues);
      const range = maxYValue - minYValue;
      const margin = range * 0.05;
      yDomain = [minYValue - margin, maxYValue + margin];
    }

    if (trackId === 'scaling') {
      const groups: Record<string, Run[]> = {};
      filteredRuns.forEach(run => {
        const folder = getScalingGroupName(run.run_name);
        if (!groups[folder]) groups[folder] = [];
        groups[folder].push(run);
      });

      const scalingGroups: ScalingGroup[] = Object.entries(groups)
        .map(([folder, groupRuns]) => {
        const dataPoints = groupRuns
          .map(r => {
            const absoluteY = r.eval_paloma_c4_en_bpb;
            const xValue = r[xAxis];
            const relativeMetrics = getRelativeMetrics(absoluteY, xValue);
            const yValue = yAxis === 'relative' ? relativeMetrics.percent : absoluteY;

            const tokens =
              r.training_hardware_flops && r.model_flops
                ? Math.round(r.training_hardware_flops / (6 * r.model_flops))
                : undefined;

            return {
              x: r[xAxis],
              y: yValue,
              name: r.run_name,
              logX: Math.log(r[xAxis]),
              logAbsoluteY: Math.log(absoluteY),
              logRelativeY: relativeMetrics.logRatio,
              trainingFlops: r.training_hardware_flops,
              modelFlops: r.model_flops,
              modelSize: r.model_size,
              tokens,
              bpb: r.eval_paloma_c4_en_bpb
            };
          })
          .sort((a, b) => a.x - b.x);

        const regressionInputAbsolute = dataPoints.map(point => ({
          x: point.logX,
          y: point.logAbsoluteY
        }));

        const regressionInputRelative =
          yAxis === 'relative'
            ? dataPoints
                .filter(point => point.logRelativeY !== undefined)
                .map(point => ({
                  x: point.logX,
                  y: point.logRelativeY as number
                }))
            : [];

        const absoluteRegression =
          regressionInputAbsolute.length >= 2 ? computeLinearRegression(regressionInputAbsolute) : null;
        const relativeRegression =
          yAxis === 'relative' && regressionInputRelative.length >= 2
            ? computeLinearRegression(regressionInputRelative)
            : null;

        const displayRegression = yAxis === 'relative' ? relativeRegression ?? absoluteRegression : absoluteRegression;

        if (!displayRegression) {
          return {
            name: folder,
            data: dataPoints,
            regression: dataPoints,
            forecastedBpbAt1e22: Infinity,
            leaderboard: null
          };
        }

        const { slope, intercept } = displayRegression;
        let displaySlope = slope;
        let displayIntercept = intercept;

        if (yAxis === 'relative' && baselineRegression && dataPoints.length > 0) {
          const averageDelta =
            dataPoints.reduce((sum, point) => {
              const baselineLog =
                baselineRegression.intercept + baselineRegression.slope * point.logX;
              return sum + (point.logAbsoluteY - baselineLog);
            }, 0) / dataPoints.length;

          displaySlope = baselineRegression.slope;
          displayIntercept = baselineRegression.intercept + averageDelta;
        }

        const logX1e22 = Math.log(1e22);

        const forecastDataPoints = groupRuns
          .filter(r => r.model_flops > 0 && r.eval_paloma_c4_en_bpb > 0)
          .map(r => ({
            x: Math.log(r.model_flops),
            y: Math.log(r.eval_paloma_c4_en_bpb)
          }));

        const forecastRegression = computeLinearRegression(forecastDataPoints);
        const projected =
          forecastRegression &&
          Math.exp(forecastRegression.intercept + forecastRegression.slope * logX1e22);
        const forecastedBpbAt1e22 = projected ?? Infinity;

        const minX = Math.min(...dataPoints.map(p => p.x));
        const maxX = 1e22;
        const regressionPoints = [];
        const numPoints = 100;

        for (let i = 0; i < numPoints; i++) {
          const logX = Math.log(minX) + ((Math.log(maxX) - Math.log(minX)) * i) / (numPoints - 1);
          const x = Math.exp(logX);
          if (yAxis === 'relative' && relativeRegression) {
            const logRatio = intercept + slope * logX;
            const ratio = Math.exp(logRatio);
            const y = (ratio - 1) * 100;
            regressionPoints.push({ x, y, name: folder });
          } else {
            const logY = intercept + slope * logX;
            const absoluteY = Math.exp(logY);
            const y = absoluteY;
            regressionPoints.push({ x, y, name: folder });
          }
        }

        const firstRun = groupRuns[0];
        const leaderboardRegression = absoluteRegression;

        const leaderboard: ScalingGroupMeta | null = firstRun
          ? {
              author: firstRun.author.name,
              authorUrl: firstRun.author.url,
              affiliation: firstRun.author.affiliation,
              wandb: firstRun.wandb_link,
              filepath: firstRun.results_filepath.split('/').slice(0, -1).join('/'),
              date: firstRun.run_completion_timestamp,
              intercept: leaderboardRegression?.intercept ?? null,
              slope: leaderboardRegression?.slope ?? null,
              r2: leaderboardRegression?.r2 ?? null,
              projected: projected ?? null
            }
          : null;

        return {
          name: folder,
          data: dataPoints,
          regression: regressionPoints,
          forecastedBpbAt1e22,
          leaderboard
        };
      });

      const qualifiedScalingGroups = scalingGroups.filter(group =>
        group.data.some(point => (point.trainingFlops ?? 0) > MIN_SCALING_LEADERBOARD_FLOPS)
      );

      const sortedByForecast = [...qualifiedScalingGroups].sort(
        (a, b) => a.forecastedBpbAt1e22 - b.forecastedBpbAt1e22
      );
      const top3Groups = new Set(sortedByForecast.slice(0, 3).map(g => g.name));
      const scalingBackground = processedPoints.filter(p => p.inTrack);

      return {
        type: 'scaling' as const,
        background: scalingBackground,
        groups: sortedByForecast,
        yDomain,
        top3Groups
      };
    }

    return {
      type: 'standard' as const,
      all: processedPoints,
      highlighted: processedPoints,
      pareto:
        yAxis === 'relative'
          ? paretoPoints.map(p => ({ ...p, y: 0 }))
          : paretoPoints,
      yDomain
    };
  }, [runs, filteredRuns, trackId, xAxis, yAxis]);

  const xTicks = useMemo(() => {
    const allXValues: number[] = [];
    if (chartData.type === 'scaling') {
      chartData.groups.forEach(group => {
        group.data.forEach(point => allXValues.push(point.x));
      });
    } else {
      chartData.all.forEach(point => allXValues.push(point.x));
    }

    if (allXValues.length === 0) return [];

    const minX = Math.min(...allXValues);
    const maxX = 1e22;

    const minExp = Math.floor(Math.log10(minX));
    const maxExp = Math.ceil(Math.log10(maxX));

    const ticks: number[] = [];
    for (let exp = minExp; exp <= maxExp; exp++) {
      ticks.push(Math.pow(10, exp));
      ticks.push(Math.pow(10, exp + 0.5));
    }

    return ticks.filter(tick => tick >= minX * 0.8 && tick <= maxX * 1.2);
  }, [chartData]);

  const leaderboardRows = useMemo<LeaderboardRow[]>(() => {
    if (trackId === 'scaling' && chartData.type === 'scaling') {
      return chartData.groups
        .map(group => {
          if (!group.leaderboard) return null;

          const hasHighFlopsRun = group.data.some(
            point => (point.trainingFlops ?? 0) > MIN_SCALING_LEADERBOARD_FLOPS
          );
          if (!hasHighFlopsRun) return null;

          return {
            type: 'scaling' as const,
            name: group.name,
            ...group.leaderboard
          };
        })
        .filter((row): row is ScalingLeaderboardRow => row !== null);
    }

    return [...filteredRuns]
      .sort((a, b) => a.eval_paloma_c4_en_bpb - b.eval_paloma_c4_en_bpb)
      .map((run, idx) => ({
        type: 'standard' as const,
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
        date: run.run_completion_timestamp
      }));
  }, [chartData, filteredRuns, trackId]);

  return {
    chartData,
    nextLower,
    xTicks,
    leaderboardRows
  };
}
