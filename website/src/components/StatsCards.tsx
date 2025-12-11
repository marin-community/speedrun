import { getScalingGroupName } from '../utils/scaling';
import type { ChartData } from '../hooks/useSpeedrunData';

interface Run {
  run_name: string;
  eval_paloma_c4_en_bpb: number;
  training_hardware_flops: number;
}

interface StatsCardsProps {
  runs: Run[];
  trackId: string;
  allRuns: Run[];
  chartData: ChartData;
}

export function StatsCards({ runs, trackId, allRuns, chartData }: StatsCardsProps) {
  const FLOPS_BUDGET = 1e22;
  
  let numRuns = 0;
  let bestFlopsHeader = "Best FLOPs in Track";
  let bestFlopsValue = "N/A";
  let bestBpbHeader = "Best C4-EN BPB";
  let bestBpbValue = "N/A";
  
  if (trackId === 'scaling') {
    const groupedRuns = runs.reduce((acc, run) => {
      const folder = getScalingGroupName(run.run_name);
      if (!acc[folder]) acc[folder] = [];
      acc[folder].push(run);
      return acc;
    }, {} as Record<string, Run[]>);
    
    numRuns = Object.keys(groupedRuns).length;
    
    const scalingGroups = chartData?.type === 'scaling' ? chartData.groups : [];
    const finiteProjections = scalingGroups
      .map(group => group.forecastedBpbAt1e22)
      .filter(value => Number.isFinite(value));
    if (finiteProjections.length > 0) {
      bestBpbValue = Math.min(...finiteProjections).toFixed(4);
    }

    const finiteSlopes = scalingGroups
      .map(group => group.leaderboard?.slope)
      .filter((s): s is number => typeof s === 'number' && Number.isFinite(s));
    if (finiteSlopes.length > 0) {
      const bestSlope = Math.min(...finiteSlopes);
      bestFlopsValue = bestSlope.toFixed(4);
    }
    bestBpbHeader = `Best Projected BPB @ ${FLOPS_BUDGET.toExponential(0)} FLOPs`;
    bestFlopsHeader = "Best Compute Scaling Term";
  } else {
    numRuns = runs.length;
    const bestFlops = runs.length > 0 
      ? Math.min(...runs.map(r => r.training_hardware_flops))
      : NaN;
    bestFlopsValue = isNaN(bestFlops) ? "N/A" : bestFlops.toExponential(3);
    
    const bestBpb = runs.length > 0
      ? Math.min(...runs.map(r => r.eval_paloma_c4_en_bpb))
      : NaN;
    bestBpbValue = isNaN(bestBpb) ? "N/A" : bestBpb.toFixed(4);
  }
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg text-gray-600 mb-2">Total Runs in Track</h3>
        <div className="text-2xl text-gray-900">{numRuns}</div>
      </div>
      
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg text-gray-600 mb-2">{bestFlopsHeader}</h3>
        <div className="text-2xl text-gray-900">{bestFlopsValue}</div>
      </div>
      
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg text-gray-600 mb-2">{bestBpbHeader}</h3>
        <div className="text-2xl text-gray-900">{bestBpbValue}</div>
      </div>
    </div>
  );
}
