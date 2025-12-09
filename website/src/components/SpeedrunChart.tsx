import { useMemo, useState, useEffect, useRef } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Line, LineChart, ReferenceArea, ReferenceLine } from 'recharts';
import domtoimage from 'dom-to-image-more';

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
}

interface Track {
  id: string;
  name: string;
  color: string;
  target_bpb?: number;
}

interface SpeedrunChartProps {
  runs: Run[];
  filteredRuns: Run[];
  trackId: string;
  currentTrack?: Track;
  tracks: Track[];
}

export function SpeedrunChart({ runs, filteredRuns, trackId, currentTrack, tracks }: SpeedrunChartProps) {
  // Read initial values from URL
  const [xAxis, setXAxis] = useState<'training_hardware_flops' | 'model_flops'>(() => {
    const params = new URLSearchParams(window.location.search);
    const xParam = params.get('xAxis');
    return xParam === 'training_hardware_flops' ? 'training_hardware_flops' : 'model_flops';
  });
  
  const [yAxis, setYAxis] = useState<'absolute' | 'relative'>(() => {
    const params = new URLSearchParams(window.location.search);
    const yParam = params.get('yAxis');
    return yParam === 'absolute' ? 'absolute' : 'relative';
  });

  // Track which legend items are visible
  const [hiddenLegendItems, setHiddenLegendItems] = useState<Set<string>>(new Set());

  // Track window size for responsive margins
  const [isMobile, setIsMobile] = useState(false);

  // Ref for chart container to enable downloading
  const chartRef = useRef<HTMLDivElement>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768); // md breakpoint
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Initialize hidden legend items when switching to scaling track - hide all except top 3
  useEffect(() => {
    if (trackId === 'scaling' && chartData.type === 'scaling') {
      const groupsToHide = chartData.groups
        .filter(g => !chartData.top3Groups.has(g.name))
        .map(g => g.name);
      // Also hide "All Runs" by default
      setHiddenLegendItems(new Set([...groupsToHide, 'All Runs']));
    } else {
      // Clear hidden items when switching to non-scaling track
      setHiddenLegendItems(new Set());
    }
  }, [trackId, xAxis, yAxis]); // Re-run when track OR axes change

  // Update URL when axis options change
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set('xAxis', xAxis);
    params.set('yAxis', yAxis);
    window.history.replaceState({}, '', `${window.location.pathname}?${params.toString()}`);
  }, [xAxis, yAxis]);

  // Custom tooltip component
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    
    const data = payload[0].payload;
    
    return (
      <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
        <p className="font-semibold mb-2 text-sm">{data.name}</p>
        <div className="text-xs space-y-1">
          {data.bpb !== undefined && (
            <p><span className="text-gray-600">Validation Loss (BPB):</span> <span className="font-medium">{data.bpb.toFixed(4)}</span></p>
          )}
          {data.trainingFlops !== undefined && (
            <p><span className="text-gray-600">Training FLOPs:</span> <span className="font-medium">{data.trainingFlops.toExponential(2)}</span></p>
          )}
          {data.modelFlops !== undefined && (
            <p><span className="text-gray-600">Model FLOPs:</span> <span className="font-medium">{data.modelFlops.toExponential(2)}</span></p>
          )}
          {data.modelSize !== undefined && (
            <p><span className="text-gray-600">Model Size:</span> <span className="font-medium">{(data.modelSize / 1e6).toFixed(1)}M params</span></p>
          )}
          {data.tokens !== undefined && (
            <p><span className="text-gray-600">Tokens:</span> <span className="font-medium">{data.tokens.toExponential(2)}</span></p>
          )}
        </div>
      </div>
    );
  };

  // Calculate next lower BPB for shading
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

  const chartData = useMemo(() => {
    // Use only filtered runs for display
    const allPoints = filteredRuns.map(r => {
      // Calculate tokens from training flops and model flops: tokens ≈ training_flops / (6 * model_flops)
      const tokens = r.training_hardware_flops && r.model_flops 
        ? Math.round(r.training_hardware_flops / (6 * r.model_flops))
        : undefined;
      
      return {
        x: r[xAxis],
        y: r.eval_paloma_c4_en_bpb,
        name: r.run_name,
        inTrack: true, // All points are in track now since we're only showing filtered runs
        // Include all run metadata for tooltip
        trainingFlops: r.training_hardware_flops,
        modelFlops: r.model_flops,
        modelSize: r.model_size,
        tokens: tokens,
        bpb: r.eval_paloma_c4_en_bpb
      };
    });

    // Find baseline runs (AdamW plain)
    const baselineRuns = runs.filter(r => r.run_name.toLowerCase().includes('adamw'));
    const baselinePoints = baselineRuns.map(r => ({
      x: r[xAxis],
      y: r.eval_paloma_c4_en_bpb,
      name: r.run_name
    })).sort((a, b) => a.x - b.x);

    // Calculate Pareto frontier (still needed for visualization)
    const paretoPoints: typeof allPoints = [];
    const sorted = [...allPoints].sort((a, b) => a.x - b.x);
    let minY = Infinity;
    
    for (const point of sorted) {
      if (point.y < minY) {
        minY = point.y;
        paretoPoints.push(point);
      }
    }

    // Apply relative baselining if needed (using AdamW baseline)
    const processedPoints = yAxis === 'relative' ? allPoints.map(point => {
      // Find baseline BPB at this compute level (interpolate from baseline runs)
      let baselineBPB = point.y;
      
      if (baselinePoints.length > 0) {
        for (let i = 0; i < baselinePoints.length - 1; i++) {
          const p1 = baselinePoints[i];
          const p2 = baselinePoints[i + 1];
          if (point.x >= p1.x && point.x <= p2.x) {
            // Log-linear interpolation
            const logX = Math.log(point.x);
            const logX1 = Math.log(p1.x);
            const logX2 = Math.log(p2.x);
            const t = (logX - logX1) / (logX2 - logX1);
            baselineBPB = p1.y + t * (p2.y - p1.y);
            break;
          }
        }
        // If before first baseline point, use first point's BPB
        if (point.x < baselinePoints[0].x) baselineBPB = baselinePoints[0].y;
        // If after last baseline point, use last point's BPB
        if (point.x > baselinePoints[baselinePoints.length - 1].x) baselineBPB = baselinePoints[baselinePoints.length - 1].y;
      }
      
      return {
        ...point,
        y: ((point.y - baselineBPB) / baselineBPB) * 100 // Percentage difference
      };
    }) : allPoints;

    // Calculate Y-axis domain with small margin for absolute mode
    let yDomain: [number | string, number | string] = ['auto', 'auto'];
    if (yAxis === 'absolute') {
      const allYValues = processedPoints.map(p => p.y);
      const minY = Math.min(...allYValues);
      const maxY = Math.max(...allYValues);
      const range = maxY - minY;
      const margin = range * 0.05; // 5% margin
      yDomain = [minY - margin, maxY + margin];
    }

    if (trackId === 'scaling') {
      const groups: Record<string, Run[]> = {};
      filteredRuns.forEach(run => {
        const folder = run.run_name.split('/')[0];
        if (!groups[folder]) groups[folder] = [];
        groups[folder].push(run);
      });

      const scalingGroups = Object.entries(groups).map(([folder, groupRuns]) => {
        const dataPoints = groupRuns.map(r => {
          // For scaling track, we need to apply the same baseline transformation
          let yValue = r.eval_paloma_c4_en_bpb;
          
          if (yAxis === 'relative' && baselinePoints.length > 0) {
            // Find baseline BPB at this compute level (interpolate from baseline runs)
            let baselineBPB = yValue;
            const xValue = r[xAxis];
            
            for (let i = 0; i < baselinePoints.length - 1; i++) {
              const p1 = baselinePoints[i];
              const p2 = baselinePoints[i + 1];
              if (xValue >= p1.x && xValue <= p2.x) {
                // Log-linear interpolation
                const logX = Math.log(xValue);
                const logX1 = Math.log(p1.x);
                const logX2 = Math.log(p2.x);
                const t = (logX - logX1) / (logX2 - logX1);
                baselineBPB = p1.y + t * (p2.y - p1.y);
                break;
              }
            }
            // If before first baseline point, use first point's BPB
            if (xValue < baselinePoints[0].x) baselineBPB = baselinePoints[0].y;
            // If after last baseline point, use last point's BPB
            if (xValue > baselinePoints[baselinePoints.length - 1].x) baselineBPB = baselinePoints[baselinePoints.length - 1].y;
            
            yValue = ((yValue - baselineBPB) / baselineBPB) * 100; // Percentage difference
          }
          
          // Calculate tokens from training flops and model flops
          const tokens = r.training_hardware_flops && r.model_flops 
            ? Math.round(r.training_hardware_flops / (6 * r.model_flops))
            : undefined;
          
          return {
            x: r[xAxis],
            y: yValue,
            name: r.run_name,
            logX: Math.log(r[xAxis]),
            logY: yAxis === 'relative' ? yValue : Math.log(yValue), // Don't log if already relative
            // Include metadata for tooltip
            trainingFlops: r.training_hardware_flops,
            modelFlops: r.model_flops,
            modelSize: r.model_size,
            tokens: tokens,
            bpb: r.eval_paloma_c4_en_bpb
          };
        }).sort((a, b) => a.x - b.x);

        // Compute log-log linear regression for scaling law
        const n = dataPoints.length;
        if (n < 2) {
          return { name: folder, data: dataPoints, regression: dataPoints, forecastedBpbAt1e22: Infinity };
        }

        const sumLogX = dataPoints.reduce((sum, p) => sum + p.logX, 0);
        const sumLogY = dataPoints.reduce((sum, p) => sum + p.logY, 0);
        const sumLogXY = dataPoints.reduce((sum, p) => sum + p.logX * p.logY, 0);
        const sumLogX2 = dataPoints.reduce((sum, p) => sum + p.logX * p.logX, 0);

        const slope = (n * sumLogXY - sumLogX * sumLogY) / (n * sumLogX2 - sumLogX * sumLogX);
        const intercept = (sumLogY - slope * sumLogX) / n;

        // Calculate forecasted BPB at 1e22 FLOPs
        // ALWAYS use absolute values AND training_hardware_flops for sorting purposes
        // This ensures consistent sort order regardless of axis selections
        const logX1e22 = Math.log(1e22);
        
        // Always calculate forecast based on training_hardware_flops for consistency
        const forecastDataPoints = groupRuns.map(r => ({
          logX: Math.log(r.training_hardware_flops),
          logY: Math.log(r.eval_paloma_c4_en_bpb)
        }));
        
        const forecastN = forecastDataPoints.length;
        const forecastSumLogX = forecastDataPoints.reduce((sum, p) => sum + p.logX, 0);
        const forecastSumLogY = forecastDataPoints.reduce((sum, p) => sum + p.logY, 0);
        const forecastSumLogXY = forecastDataPoints.reduce((sum, p) => sum + p.logX * p.logY, 0);
        const forecastSumLogX2 = forecastDataPoints.reduce((sum, p) => sum + p.logX * p.logX, 0);
        
        const forecastSlope = (forecastN * forecastSumLogXY - forecastSumLogX * forecastSumLogY) / (forecastN * forecastSumLogX2 - forecastSumLogX * forecastSumLogX);
        const forecastIntercept = (forecastSumLogY - forecastSlope * forecastSumLogX) / forecastN;
        const forecastedBpbAt1e22 = Math.exp(forecastIntercept + forecastSlope * logX1e22);

        // Generate regression line points
        const minX = Math.min(...dataPoints.map(p => p.x));
        const maxX = 1e22; // Forecast out to 1e22 FLOPs
        const regressionPoints = [];
        const numPoints = 100;
        
        for (let i = 0; i < numPoints; i++) {
          const logX = Math.log(minX) + (Math.log(maxX) - Math.log(minX)) * i / (numPoints - 1);
          const x = Math.exp(logX);
          const logY = intercept + slope * logX;
          const y = yAxis === 'relative' ? logY : Math.exp(logY); // Don't exp if relative
          regressionPoints.push({ x, y, name: folder });
        }

        return { name: folder, data: dataPoints, regression: regressionPoints, forecastedBpbAt1e22 };
      });

      // Sort by forecasted BPB (lower is better) and get top 3
      const sortedByForecast = [...scalingGroups].sort((a, b) => a.forecastedBpbAt1e22 - b.forecastedBpbAt1e22);
      const top3Groups = new Set(sortedByForecast.slice(0, 3).map(g => g.name));

      // For scaling track, only show scaling runs in background
      const scalingBackground = processedPoints.filter(p => p.inTrack);

      return {
        type: 'scaling' as const,
        background: scalingBackground,
        groups: sortedByForecast, // Return sorted groups so colors match forecast ranking
        yDomain: yDomain,
        top3Groups
      };
    } else {
      return {
        type: 'standard' as const,
        all: processedPoints, // Show only filtered runs
        highlighted: processedPoints, // All points are highlighted since they're all in the track
        pareto: yAxis === 'relative' 
          ? paretoPoints.map(p => ({ ...p, y: 0 })) // Pareto is baseline (0%)
          : paretoPoints,
        yDomain: yDomain
      };
    }
  }, [runs, filteredRuns, trackId, xAxis, yAxis]);

  const xAxisLabel = xAxis === 'training_hardware_flops' ? 'Training Hardware FLOPs' : 'Model FLOPs';
  const yAxisLabel = yAxis === 'absolute' ? 'C4-EN BPB' : '% Difference from Baseline';
  const colors = ['#1877F2', '#F0701A', '#5A24C7', '#E42C97', '#00487C', '#0EAC96', '#AB76FF', '#B50550', '#0099E6', '#22085F', '#783301'];

  // Generate custom x-axis ticks at 1eX and 5eX
  const generateXTicks = () => {
    // Get min and max x values from the data
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
      ticks.push(Math.pow(10, exp)); // 1
      ticks.push(Math.pow(10, exp+0.5)); // 1

    }
    
    // Filter to only include ticks within the data range (with some padding)
    return ticks.filter(tick => tick >= minX * 0.8 && tick <= maxX * 1.2);
  };

  const xTicks = generateXTicks();

  // Handle legend click to toggle visibility
  const handleLegendClick = (data: any) => {
    const itemName = data.value || data.dataKey;
    setHiddenLegendItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(itemName)) {
        newSet.delete(itemName);
      } else {
        newSet.add(itemName);
      }
      return newSet;
    });
  };

  // Download chart as PNG
  const handleDownloadChart = async () => {
    if (!chartRef.current) return;
    
    setIsDownloading(true);
    
    try {
      // Now capture with dom-to-image-more
      const dataUrl = await domtoimage.toPng(chartRef.current, {
        bgcolor: '#ffffff',
        width: chartRef.current.offsetWidth,
        height: chartRef.current.offsetHeight,
        style: {
          backgroundColor: '#ffffff'
        },
        filter: (node: any) => {
          // Filter out any tooltip elements that might be visible
          if (node.classList && node.classList.contains('recharts-tooltip-wrapper')) {
            return false;
          }
          
          // Remove borders and outlines from all elements during clone
          if (node.style) {
            node.style.border = 'none';
            node.style.outline = 'none';
            node.style.boxShadow = 'none';
          }
          
          return true;
        }
      });
      
      const link = document.createElement('a');
      const timestamp = new Date().toISOString().split('T')[0];
      link.download = `marin-speedrun-${trackId}-${xAxis}-${yAxis}-${timestamp}.png`;
      link.href = dataUrl;
      link.click();
    } catch (error) {
      console.error('Error downloading chart:', error);
      alert('Failed to download chart. Please try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  // Custom Legend component
  const CustomLegend = (props: any) => {
    const { payload } = props;
    if (!payload || !payload.length) return null;

    // Create a map of forecasted BPB values for sorting
    const forecastMap = new Map<string, number>();
    if (chartData.type === 'scaling') {
      chartData.groups.forEach(group => {
        forecastMap.set(group.name, group.forecastedBpbAt1e22);
      });
    }

    // Sort the payload
    const sortedPayload = [...payload]
      .filter((entry: any, index: number, self: any[]) => 
        self.findIndex(e => e.value === entry.value) === index
      )
      .sort((a: any, b: any) => {
        const aName = a.value;
        const bName = b.value;

        // "All Runs" always first
        if (aName === 'All Runs') return -1;
        if (bName === 'All Runs') return 1;

        // "Baseline (AdamW)" always second
        if (aName === 'Baseline (AdamW)') return -1;
        if (bName === 'Baseline (AdamW)') return 1;

        // Everything else sorted by forecasted BPB (lower is better)
        const aForecast = forecastMap.get(aName) ?? Infinity;
        const bForecast = forecastMap.get(bName) ?? Infinity;
        return aForecast - bForecast;
      });

    return (
      <div 
        style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          paddingTop: isMobile ? '16px' : '12px',
          paddingBottom: '0px'
        }}
      >
        <div style={{ 
          display: 'flex', 
          gap: '12px', 
          flexWrap: 'wrap',
          paddingLeft: '8px', 
          paddingRight: '8px',
          justifyContent: 'center',
          ...(isMobile ? {
            maxHeight: '150px', // Allow ~3 rows
            overflowY: 'auto'
          } : {})
        }}>
          {sortedPayload.map((entry: any, index: number) => {
            const isHidden = hiddenLegendItems.has(entry.value);
            return (
              <div
                key={`item-${index}`}
                onClick={() => handleLegendClick(entry)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  cursor: 'pointer',
                  opacity: isHidden ? 0.5 : 1,
                  whiteSpace: 'nowrap',
                  minHeight: isMobile ? '40px' : '32px', // Better touch target for mobile
                  padding: '4px 8px',
                  border: 'none',
                  outline: 'none',
                  boxShadow: 'none'
                }}
              >
                <svg width="14" height="14" style={{ marginRight: '8px', flexShrink: 0, border: 'none', outline: 'none' }}>
                  <circle cx="7" cy="7" r="6" fill={entry.color} />
                </svg>
                <span style={{ fontSize: '14px', border: 'none', outline: 'none' }}>{entry.value}</span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white rounded-lg shadow p-4 md:p-6 mb-2">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-2xl text-gray-900">Performance Chart</h3>
        <button
          onClick={handleDownloadChart}
          disabled={isDownloading}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {isDownloading ? (
            <>
              <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Downloading...
            </>
          ) : (
            <>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download Chart
            </>
          )}
        </button>
      </div>
      
      <div className="mb-6 flex flex-col md:flex-row md:flex-wrap gap-4 md:gap-6">
        <div className="w-full md:w-auto">
          <label className="block text-sm text-gray-700 mb-2">X-axis metric</label>
          <div className="flex flex-col md:flex-row gap-2 md:gap-4">
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                value="training_hardware_flops"
                checked={xAxis === 'training_hardware_flops'}
                onChange={(e) => setXAxis(e.target.value as 'training_hardware_flops')}
                className="mr-2"
              />
              Training Hardware FLOPs
            </label>
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                value="model_flops"
                checked={xAxis === 'model_flops'}
                onChange={(e) => setXAxis(e.target.value as 'model_flops')}
                className="mr-2"
              />
              Model FLOPs
            </label>
          </div>
        </div>
        <div className="w-full md:w-auto">
          <label className="block text-sm text-gray-700 mb-2">Y-axis metric</label>
          <div className="flex flex-col md:flex-row gap-2 md:gap-4">
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                value="absolute"
                checked={yAxis === 'absolute'}
                onChange={(e) => setYAxis(e.target.value as 'absolute')}
                className="mr-2"
              />
              Absolute BPB
            </label>
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                value="relative"
                checked={yAxis === 'relative'}
                onChange={(e) => setYAxis(e.target.value as 'relative')}
                className="mr-2"
              />
              Relative to Baseline
            </label>
          </div>
        </div>
      </div>

      <div ref={chartRef} style={{ position: 'relative' }}>
        {trackId === 'scaling' && (
          <div style={{
            position: 'absolute',
            top: isMobile ? '28px' : '32px',
            right: isMobile ? '40px' : '80px',
            zIndex: 10,
            pointerEvents: 'none',
            opacity: 0.85
          }}>
            <img
	    src="https://raw.githubusercontent.com/marin-community/speedrun/refs/heads/main/website/src/assets/marin-logo.png"  
              alt="" 
              style={{
                width: isMobile ? '24px' : '36px',
                height: isMobile ? '24px' : '36px',
                borderRadius: '50%',
                objectFit: 'cover'
              }}
            />
          </div>
        )}
        <ResponsiveContainer width="100%" height={500}>
          {trackId === 'scaling' && chartData.type === 'scaling' ? (
            <ScatterChart margin={{ 
              top: 20, 
              right: isMobile ? 10 : 60, 
              bottom: 20, 
              left: isMobile ? 40 : 80 
            }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                type="number" 
                dataKey="x" 
                name={xAxisLabel}
                scale="log"
                domain={['auto', 'auto']}
                label={{ value: xAxisLabel, position: 'insideBottom', offset: -10 }}
                tickFormatter={(value) => value.toExponential(0)}
                ticks={xTicks}
              />
              <YAxis 
                type="number" 
                dataKey="y" 
                name={yAxisLabel}
                domain={yAxis === 'absolute' ? ['auto', 'auto'] : ['auto', 'auto']}
                label={{ value: yAxisLabel, angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }}
                tickFormatter={(value) => yAxis === 'relative' ? `${value.toFixed(1)}%` : value.toFixed(3)}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend content={<CustomLegend />} />
              {/* Baseline reference line at 0% in relative mode */}
              {yAxis === 'relative' && (
                <ReferenceLine
                  y={0}
                  stroke="#000000"
                  strokeDasharray="5 5"
                  strokeWidth={2}
                />
              )}
              {/* Show all runs as background in gray */}
              <Scatter 
                name="All Runs" 
                data={hiddenLegendItems.has('All Runs') ? [] : chartData.background} 
                fill="rgba(156,163,175,0.3)" 
              />
              {/* Show baseline at 0% when in relative mode */}
              {yAxis === 'relative' && chartData.type === 'scaling' && (
                <Scatter 
                  name="Baseline (AdamW)" 
                  data={hiddenLegendItems.has('Baseline (AdamW)') ? [] : chartData.background.filter(p => p.name.toLowerCase().includes('adamw'))}
                  fill="#000000"
                />
              )}
              {/* Show scaling groups as scatter points (no lines) */}
              {chartData.groups.map((group, idx) => (
                <Scatter
                  key={group.name}
                  name={group.name}
                  data={hiddenLegendItems.has(group.name) ? [] : group.data}
                  fill={colors[idx % colors.length]}
                />
              ))}
              {/* Show regression lines */}
              {chartData.groups.map((group, idx) => (
                <Scatter
                  key={`${group.name}-fit`}
                  name={group.name}
                  data={hiddenLegendItems.has(group.name) ? [] : group.regression}
                  fill="none"
                  line={{ stroke: colors[idx % colors.length], strokeWidth: 2, strokeDasharray: '5 5' }}
                  shape={() => null}
                  legendType="none"
                />
              ))}
            </ScatterChart>
          ) : (
            <ScatterChart margin={{ 
              top: 20, 
              right: isMobile ? 10 : 60, 
              bottom: 20, 
              left: isMobile ? 10 : 60 
            }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                type="number" 
                dataKey="x" 
                name={xAxisLabel}
                scale="log"
                domain={['auto', 'auto']}
                label={{ value: xAxisLabel, position: 'insideBottom', offset: -10 }}
                tickFormatter={(value) => value.toExponential(0)}
                ticks={xTicks}
              />
              <YAxis 
                type="number" 
                dataKey="y" 
                name={yAxisLabel}
                domain={chartData.yDomain}
                label={{ value: yAxisLabel, angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }}
                tickFormatter={(value) => yAxis === 'relative' ? `${value.toFixed(1)}%` : value.toFixed(3)}
              />
              <Tooltip 
                formatter={(value: any) => {
                  if (typeof value === 'number') {
                    if (yAxis === 'relative') {
                      return `${value.toFixed(2)}%`;
                    }
                    return value < 1 ? value.toFixed(4) : value.toExponential(2);
                  }
                  return value;
                }}
              />
              <Legend content={<CustomLegend />} />
              {/* Baseline reference line at 0% in relative mode */}
              {yAxis === 'relative' && (
                <ReferenceLine
                  y={0}
                  stroke="#000000"
                  strokeDasharray="5 5"
                  strokeWidth={2}
                />
              )}
              {chartData.type === 'standard' && (
                <>
                  <Scatter 
                    name="Track Runs" 
                    data={hiddenLegendItems.has('Track Runs') ? [] : chartData.highlighted} 
                    fill={currentTrack?.color || '#3b82f6'} 
                  />
                  <Scatter 
                    name="Pareto Frontier" 
                    data={hiddenLegendItems.has('Pareto Frontier') ? [] : chartData.pareto} 
                    fill="#ef4444" 
                    line={{ stroke: '#ef4444', strokeWidth: 2 }} 
                  />
                </>
              )}
              {nextLower !== null && currentTrack?.target_bpb && (
                <>
                  <ReferenceArea
                    y1={nextLower}
                    y2={currentTrack.target_bpb}
                    fill={currentTrack.color}
                    fillOpacity={0.15}
                    stroke="none"
                  />
                  <ReferenceLine
                    y={currentTrack.target_bpb}
                    stroke={currentTrack.color}
                    strokeDasharray="5 5"
                    strokeWidth={2}
                    label={{ value: `Target: ${currentTrack.target_bpb.toFixed(2)}`, position: 'right' }}
                  />
                </>
              )}
            </ScatterChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}