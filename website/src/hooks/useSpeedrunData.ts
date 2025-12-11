import { useMemo } from 'react';
import { getScalingGroupName } from '../utils/scaling';
import { computeLinearRegression } from '../utils/regression';

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
      logY: number;
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

    const paretoPoints: typeof allPoints = [];
    const sorted = [...allPoints].sort((a, b) => a.x - b.x);
    let minY = Infinity;

    for (const point of sorted) {
      if (point.y < minY) {
        minY = point.y;
        paretoPoints.push(point);
      }
    }

    const processedPoints =
      yAxis === 'relative'
        ? allPoints.map(point => {
            let baselineBPB = point.y;

            if (baselinePoints.length > 0) {
              for (let i = 0; i < baselinePoints.length - 1; i++) {
                const p1 = baselinePoints[i];
                const p2 = baselinePoints[i + 1];
                if (point.x >= p1.x && point.x <= p2.x) {
                  const logX = Math.log(point.x);
                  const logX1 = Math.log(p1.x);
                  const logX2 = Math.log(p2.x);
                  const t = (logX - logX1) / (logX2 - logX1);
                  baselineBPB = p1.y + t * (p2.y - p1.y);
                  break;
                }
              }
              if (point.x < baselinePoints[0].x) baselineBPB = baselinePoints[0].y;
              if (point.x > baselinePoints[baselinePoints.length - 1].x)
                baselineBPB = baselinePoints[baselinePoints.length - 1].y;
            }

            return {
              ...point,
              y: ((point.y - baselineBPB) / baselineBPB) * 100
            };
          })
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

      const scalingGroups: ScalingGroup[] = Object.entries(groups).map(([folder, groupRuns]) => {
        const dataPoints = groupRuns
          .map(r => {
            let yValue = r.eval_paloma_c4_en_bpb;

            if (yAxis === 'relative' && baselinePoints.length > 0) {
              let baselineBPB = yValue;
              const xValue = r[xAxis];

              for (let i = 0; i < baselinePoints.length - 1; i++) {
                const p1 = baselinePoints[i];
                const p2 = baselinePoints[i + 1];
                if (xValue >= p1.x && xValue <= p2.x) {
                  const logX = Math.log(xValue);
                  const logX1 = Math.log(p1.x);
                  const logX2 = Math.log(p2.x);
                  const t = (logX - logX1) / (logX2 - logX1);
                  baselineBPB = p1.y + t * (p2.y - p1.y);
                  break;
                }
              }
              if (xValue < baselinePoints[0].x) baselineBPB = baselinePoints[0].y;
              if (xValue > baselinePoints[baselinePoints.length - 1].x)
                baselineBPB = baselinePoints[baselinePoints.length - 1].y;

              yValue = ((yValue - baselineBPB) / baselineBPB) * 100;
            }

            const tokens =
              r.training_hardware_flops && r.model_flops
                ? Math.round(r.training_hardware_flops / (6 * r.model_flops))
                : undefined;

            return {
              x: r[xAxis],
              y: yValue,
              name: r.run_name,
              logX: Math.log(r[xAxis]),
              logY: yAxis === 'relative' ? yValue : Math.log(yValue),
              trainingFlops: r.training_hardware_flops,
              modelFlops: r.model_flops,
              modelSize: r.model_size,
              tokens,
              bpb: r.eval_paloma_c4_en_bpb
            };
          })
          .sort((a, b) => a.x - b.x);

        const regressionInput = dataPoints.map(point => ({
          x: point.logX,
          y: point.logY
        }));

        const regressionResult = computeLinearRegression(regressionInput);
        if (!regressionResult || dataPoints.length < 2) {
          return {
            name: folder,
            data: dataPoints,
            regression: dataPoints,
            forecastedBpbAt1e22: Infinity,
            leaderboard: null
          };
        }

        const { slope, intercept } = regressionResult;

        const logX1e22 = Math.log(1e22);

        const forecastDataPoints = groupRuns
          .filter(r => r.training_hardware_flops > 0 && r.eval_paloma_c4_en_bpb > 0)
          .map(r => ({
            x: Math.log(r.training_hardware_flops),
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
          const logY = intercept + slope * logX;
          const y = yAxis === 'relative' ? logY : Math.exp(logY);
          regressionPoints.push({ x, y, name: folder });
        }

        const firstRun = groupRuns[0];
        const leaderboard: ScalingGroupMeta | null = firstRun
          ? {
              author: firstRun.author.name,
              authorUrl: firstRun.author.url,
              affiliation: firstRun.author.affiliation,
              wandb: firstRun.wandb_link,
              filepath: firstRun.results_filepath.split('/').slice(0, -1).join('/'),
              date: firstRun.run_completion_timestamp,
              intercept: regressionResult.intercept,
              slope: regressionResult.slope,
              r2: regressionResult.r2,
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

      const sortedByForecast = [...scalingGroups].sort(
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
