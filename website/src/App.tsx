import { useState, useEffect, useMemo } from 'react';
import { Header } from './components/Header';
import { Introduction } from './components/Introduction';
import { TrackTabs } from './components/TrackTabs';
import { StatsCards } from './components/StatsCards';
import { SpeedrunChart } from './components/SpeedrunChart';
import { LeaderboardTable } from './components/LeaderboardTable';
import { useSpeedrunData } from './hooks/useSpeedrunData';

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
  run_name?: string;
}

export default function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [loading, setLoading] = useState(true);

  // Read initial track from URL
  const [selectedTrack, setSelectedTrack] = useState<string>(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('track') || 'scaling';
  });

  const [xAxis, setXAxis] = useState<'training_hardware_flops' | 'model_flops'>(() => {
    const params = new URLSearchParams(window.location.search);
    const xParam = params.get('xAxis');
    return xParam === 'training_hardware_flops' ? 'training_hardware_flops' : 'model_flops';
  });

  const [yAxis, setYAxis] = useState<'absolute' | 'relative'>(() => {
    const params = new URLSearchParams(window.location.search);
    const yParam = params.get('yAxis');
    return yParam === 'absolute' ? 'absolute' : 'absolute';
  });

  // Update URL when track changes
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set('track', selectedTrack);
    window.history.replaceState({}, '', `${window.location.pathname}?${params.toString()}`);
  }, [selectedTrack]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set('xAxis', xAxis);
    params.set('yAxis', yAxis);
    window.history.replaceState({}, '', `${window.location.pathname}?${params.toString()}`);
  }, [xAxis, yAxis]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [runsRes, tracksRes] = await Promise.all([
          fetch('https://raw.githubusercontent.com/marin-community/speedrun/refs/heads/main/data/runs.json'),
          fetch('https://raw.githubusercontent.com/marin-community/speedrun/refs/heads/main/data/tracks.json')
        ]);

        const runsData = await runsRes.json();
        const tracksData = await tracksRes.json();

        // Only keep "all" and "scaling" tracks
        const filteredTracks = tracksData.filter((t: Track) => t.id === 'all' || t.id === 'scaling');

        setRuns(runsData);
        setTracks(filteredTracks);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const currentTrack = useMemo(() =>
    tracks.find(t => t.id === selectedTrack),
    [tracks, selectedTrack]
  );

  const filteredRuns = useMemo(() => {
    if (selectedTrack === 'all') return runs;
    if (selectedTrack === 'scaling') {
      return runs.filter(r => r.run_name.includes('/'));
    }

    if (!currentTrack?.target_bpb) return runs;

    const sortedTracks = tracks
      .filter(t => t.id !== 'all' && t.target_bpb)
      .sort((a, b) => (b.target_bpb || 0) - (a.target_bpb || 0));

    const idx = sortedTracks.findIndex(t => t.id === selectedTrack);
    const nextLower = idx < sortedTracks.length - 1 ? sortedTracks[idx + 1].target_bpb || 0 : 0;

    return runs.filter(r =>
      r.eval_paloma_c4_en_bpb !== null &&
      r.eval_paloma_c4_en_bpb <= currentTrack.target_bpb! &&
      r.eval_paloma_c4_en_bpb > nextLower
    );
  }, [runs, selectedTrack, currentTrack, tracks]);

  const processedData = useSpeedrunData({
    runs,
    filteredRuns,
    trackId: selectedTrack,
    currentTrack,
    tracks,
    xAxis,
    yAxis
  });

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-xl text-gray-600">Loading Marin Speedrun data...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <Introduction />

        <TrackTabs
          tracks={tracks}
          selectedTrack={selectedTrack}
          onSelectTrack={setSelectedTrack}
        />

        <StatsCards
          runs={filteredRuns}
          trackId={selectedTrack}
          allRuns={runs}
          chartData={processedData.chartData}
        />

        <SpeedrunChart
          trackId={selectedTrack}
          currentTrack={currentTrack}
          chartData={processedData.chartData}
          nextLower={processedData.nextLower}
          xTicks={processedData.xTicks}
          xAxis={xAxis}
          yAxis={yAxis}
          setXAxis={setXAxis}
          setYAxis={setYAxis}
        />

        <LeaderboardTable
          trackId={selectedTrack}
          currentTrack={currentTrack}
          rows={processedData.leaderboardRows}
        />
      </div>
    </div>
  );
}
