import { useMemo } from 'react';

interface Run {
  run_name: string;
  author: {
    name: string;
    url: string;
    affiliation?: string;
  };
  eval_paloma_c4_en_bpb: number;
  training_hardware_flops: number;
  model_size: number;
  training_time: number;
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

interface LeaderboardTableProps {
  runs: Run[];
  trackId: string;
  currentTrack?: Track;
  allRuns: Run[];
}

export function LeaderboardTable({ runs, trackId, currentTrack }: LeaderboardTableProps) {
  const tableData = useMemo(() => {
    const sorted = [...runs].sort((a, b) => a.eval_paloma_c4_en_bpb - b.eval_paloma_c4_en_bpb);
    
    if (trackId === 'scaling') {
      const groups: Record<string, Run[]> = {};
      sorted.forEach(run => {
        const folder = run.run_name.split('/')[0];
        if (!groups[folder]) groups[folder] = [];
        groups[folder].push(run);
      });
      
      return Object.entries(groups).map(([folder, groupRuns]) => {
        // Calculate scaling law
        if (groupRuns.length >= 2) {
          const x = groupRuns.map(r => Math.log(r.training_hardware_flops));
          const y = groupRuns.map(r => Math.log(r.eval_paloma_c4_en_bpb));
          
          const n = x.length;
          const sumX = x.reduce((a, b) => a + b, 0);
          const sumY = y.reduce((a, b) => a + b, 0);
          const sumXY = x.reduce((a, b, i) => a + b * y[i], 0);
          const sumXX = x.reduce((a, b) => a + b * b, 0);
          
          const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
          const intercept = (sumY - slope * sumX) / n;
          
          // Calculate R²
          const yMean = sumY / n;
          const yPred = x.map(xi => intercept + slope * xi);
          const ssRes = y.reduce((a, yi, i) => a + Math.pow(yi - yPred[i], 2), 0);
          const ssTot = y.reduce((a, yi) => a + Math.pow(yi - yMean, 2), 0);
          const r2 = 1 - ssRes / ssTot;
          
          const FLOPS_BUDGET = 1e22;
          const projected = Math.exp(intercept + slope * Math.log(FLOPS_BUDGET));
          
          return {
            name: folder,
            author: groupRuns[0].author.name,
            authorUrl: groupRuns[0].author.url,
            affiliation: groupRuns[0].author.affiliation,
            intercept,
            slope,
            r2,
            projected,
            wandb: groupRuns[0].wandb_link,
            filepath: groupRuns[0].results_filepath.split('/').slice(0, -1).join('/'),
            date: groupRuns[0].run_completion_timestamp
          };
        }
        return null;
      }).filter(Boolean).sort((a, b) => (a?.projected || 0) - (b?.projected || 0));
    }
    
    return sorted.map((run, idx) => ({
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
  }, [runs, trackId]);

  const formatModelSize = (size: number) => {
    if (!size) return 'N/A';
    return size < 1e9 ? `${(size / 1e6).toFixed(1)} M` : `${(size / 1e9).toFixed(1)} B`;
  };

  const formatFlops = (flops: number) => {
    if (!flops) return 'N/A';
    return flops.toExponential(2);
  };

  const formatDate = (timestamp: string) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp.replace(' UTC', '')).toISOString().split('T')[0];
  };

  return (
    <div className="bg-white rounded-lg shadow p-4 md:p-6">
      <div className="flex items-center gap-2 mb-4">
        {currentTrack && trackId !== 'all' && (
          <img 
            src="https://raw.githubusercontent.com/marin-community/speedrun/refs/heads/website/src/assets/c50ff30aae7d4e504c6176c6fd540903d04b93f5.png" 
            alt="Marin logo"
            className="w-8 h-8"
          />
        )}
        <h3 className="text-2xl text-gray-900">
          {trackId === 'scaling' ? 'Scaling' : trackId === 'all' ? 'All Runs' : currentTrack?.name || ''} Leaderboard
        </h3>
      </div>

      {trackId !== 'all' && currentTrack?.target_bpb && (
        <p className="text-gray-600 mb-4">
          Runs achieving ≤ {currentTrack.target_bpb.toFixed(4)} C4-EN BPB, ranked by training efficiency
        </p>
      )}

      {/* Mobile Card View */}
      <div className="md:hidden space-y-4">
        {tableData.map((row: any, idx) => (
          <div key={idx} className="border border-gray-200 rounded-lg overflow-hidden bg-white shadow-sm">
            {/* Card Header */}
            <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
              <div className="flex items-center gap-2">
                {trackId !== 'scaling' && (
                  <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-semibold">
                    {row.rank}
                  </div>
                )}
                {trackId === 'scaling' && (
                  <img 
                    src="https://raw.githubusercontent.com/marin-community/speedrun/refs/heads/main/website/src/assets/marin-logo.png"
                    alt="Scaling track"
                    className="w-6 h-6 flex-shrink-0"
                  />
                )}
                <a 
                  href={`https://github.com/marin-community/marin/tree/main/${row.filepath}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-700 font-medium truncate"
                >
                  {row.name}
                </a>
              </div>
            </div>

            {/* Author Section */}
            <div className="px-4 py-2 bg-gray-25">
              <a 
                href={row.authorUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-700 text-sm"
              >
                {row.author}
              </a>
              {row.affiliation && (
                <span className="text-gray-500 text-sm"> ({row.affiliation})</span>
              )}
            </div>

            {/* Metrics Grid */}
            <div className="px-4 py-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <div>
                <span className="text-gray-500">Date Added:</span>
              </div>
              <div className="text-gray-900 font-medium">
                {formatDate(row.date)}
              </div>

              {trackId === 'scaling' ? (
                <>
                  <div>
                    <span className="text-gray-500">Intercept:</span>
                  </div>
                  <div className="text-gray-900 font-medium">
                    {row.intercept?.toFixed(3)}
                  </div>

                  <div>
                    <span className="text-gray-500">Slope:</span>
                  </div>
                  <div className="text-gray-900 font-medium">
                    {row.slope?.toFixed(3)}
                  </div>

                  <div>
                    <span className="text-gray-500">R²:</span>
                  </div>
                  <div className="text-gray-900 font-medium">
                    {row.r2?.toFixed(3)}
                  </div>

                  <div>
                    <span className="text-gray-500">Projected BPB:</span>
                  </div>
                  <div className="text-gray-900 font-medium">
                    {row.projected?.toFixed(3)}
                  </div>
                </>
              ) : (
                <>
                  <div>
                    <span className="text-gray-500">Model Size:</span>
                  </div>
                  <div className="text-gray-900 font-medium">
                    {formatModelSize(row.modelSize)}
                  </div>

                  <div>
                    <span className="text-gray-500">Training Time:</span>
                  </div>
                  <div className="text-gray-900 font-medium">
                    {(row.trainingTime / 60).toFixed(1)} m
                  </div>

                  <div>
                    <span className="text-gray-500">Total FLOPs:</span>
                  </div>
                  <div className="text-gray-900 font-medium">
                    {formatFlops(row.flops)}
                  </div>

                  <div>
                    <span className="text-gray-500">C4-EN BPB:</span>
                  </div>
                  <div className="text-gray-900 font-medium">
                    {row.bpb?.toFixed(3)}
                  </div>
                </>
              )}
            </div>

            {/* Links Footer */}
            <div className="px-4 py-3 bg-gray-50 border-t border-gray-200 flex gap-3 justify-center">
              <a 
                href={`https://github.com/marin-community/marin/tree/main/${row.filepath}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-3 py-2 bg-gray-800 text-white rounded hover:bg-gray-700 text-sm min-h-[44px]"
              >
                <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                </svg>
                GitHub
              </a>
              
              {row.wandb ? (
                <a 
                  href={row.wandb}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center px-3 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600 text-sm min-h-[44px]"
                >
                  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z" />
                  </svg>
                  W&B
                </a>
              ) : (
                <span className="inline-flex items-center px-3 py-2 bg-gray-300 text-gray-500 rounded text-sm min-h-[44px]">
                  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z" />
                  </svg>
                  W&B N/A
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Desktop Table View */}
      <div className="hidden md:block overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {trackId !== 'scaling' && <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">Rank</th>}
              <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">Run Name</th>
              <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">Author</th>
              <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">Date Added</th>
              {trackId === 'scaling' ? (
                <>
                  <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">Intercept</th>
                  <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">Slope</th>
                  <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">R²</th>
                  <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">Projected BPB</th>
                </>
              ) : (
                <>
                  <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">Model Size</th>
                  <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">Training Time</th>
                  <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">Total FLOPs</th>
                  <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">C4-EN BPB</th>
                </>
              )}
              <th className="px-4 py-3 text-left text-xs text-gray-500 uppercase tracking-wider">W&B Run</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {tableData.map((row: any, idx) => (
              <tr key={idx} className="hover:bg-gray-50">
                {trackId !== 'scaling' && <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">{row.rank}</td>}
                <td className="px-4 py-4 whitespace-nowrap text-sm">
                  <a 
                    href={`https://github.com/marin-community/marin/tree/main/${row.filepath}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-700"
                  >
                    {row.name}
                  </a>
                </td>
                <td className="px-4 py-4 text-sm">
                  <a 
                    href={row.authorUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-700"
                  >
                    {row.author}
                  </a>
                  {row.affiliation && (
                    <div className="text-xs text-gray-500">{row.affiliation}</div>
                  )}
                </td>
                <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">{formatDate(row.date)}</td>
                {trackId === 'scaling' ? (
                  <>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">{row.intercept?.toFixed(3)}</td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">{row.slope?.toFixed(3)}</td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">{row.r2?.toFixed(3)}</td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">{row.projected?.toFixed(3)}</td>
                  </>
                ) : (
                  <>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">{formatModelSize(row.modelSize)}</td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">{(row.trainingTime / 60).toFixed(1)} m</td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">{formatFlops(row.flops)}</td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">{row.bpb?.toFixed(3)}</td>
                  </>
                )}
                <td className="px-4 py-4 whitespace-nowrap text-sm">
                  {row.wandb ? (
                    <a 
                      href={row.wandb}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-700"
                    >
                      View Run
                    </a>
                  ) : (
                    <span className="text-gray-400">N/A</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {trackId !== 'scaling' && (
        <div className="mt-4 text-sm text-gray-500">
          <p>* Model size refers to the total number of trainable parameters</p>
          <p>* Total FLOPs refers to hardware FLOPs performed during training</p>
        </div>
      )}
    </div>
  );
}