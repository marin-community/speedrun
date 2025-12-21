import { useState, useEffect, useRef } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceArea,
  ReferenceLine
} from 'recharts';
import domtoimage from 'dom-to-image-more';
import { ChartData } from '../hooks/useSpeedrunData';

interface Track {
  id: string;
  name: string;
  color: string;
  target_bpb?: number;
}

interface SpeedrunChartProps {
  trackId: string;
  currentTrack?: Track;
  chartData: ChartData;
  nextLower: number | null;
  xTicks: number[];
  xAxis: 'training_hardware_flops' | 'model_flops';
  yAxis: 'absolute' | 'relative';
  setXAxis: (value: 'training_hardware_flops' | 'model_flops') => void;
  setYAxis: (value: 'absolute' | 'relative') => void;
}

export function SpeedrunChart({
  trackId,
  currentTrack,
  chartData,
  nextLower,
  xTicks,
  xAxis,
  yAxis,
  setXAxis,
  setYAxis
}: SpeedrunChartProps) {
  const [hiddenLegendItems, setHiddenLegendItems] = useState<Set<string>>(new Set());
  const [isMobile, setIsMobile] = useState(false);
  const chartRef = useRef<HTMLDivElement>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const prevTrackIdRef = useRef<string | null>(null);
  const groupColorMapRef = useRef<Map<string, string>>(new Map());

  const colors = ['#1877F2', '#F0701A', '#5A24C7', '#E42C97', '#00487C', '#0EAC96', '#AB76FF', '#B50550', '#0099E6', '#22085F', '#783301'];

  // Get a stable color for a group name - once assigned, a group keeps its color
  const getGroupColor = (groupName: string): string => {
    if (groupColorMapRef.current.has(groupName)) {
      return groupColorMapRef.current.get(groupName)!;
    }
    // Assign the next available color
    const usedColors = new Set(groupColorMapRef.current.values());
    const availableColor = colors.find(c => !usedColors.has(c)) || colors[groupColorMapRef.current.size % colors.length];
    groupColorMapRef.current.set(groupName, availableColor);
    return availableColor;
  };

  // Reset color mapping when track changes
  useEffect(() => {
    if (prevTrackIdRef.current !== trackId) {
      groupColorMapRef.current.clear();
    }
  }, [trackId]);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Only reset hidden legend items when trackId changes, not when chartData changes (e.g., axis changes)
  useEffect(() => {
    if (prevTrackIdRef.current !== trackId) {
      prevTrackIdRef.current = trackId;
      if (trackId === 'scaling' && chartData.type === 'scaling') {
        const groupsToHide = chartData.groups
          .filter(g => !chartData.top3Groups.has(g.name))
          .map(g => g.name);
        setHiddenLegendItems(new Set([...groupsToHide, 'All Runs']));
      } else {
        setHiddenLegendItems(new Set());
      }
    }
  }, [trackId, chartData]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;

    return (
      <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
        <p className="font-semibold mb-2 text-sm">{data.name}</p>
        <div className="text-xs space-y-1">
          {data.bpb !== undefined && (
            <p>
              <span className="text-gray-600">Validation Loss (BPB):</span>{' '}
              <span className="font-medium">{data.bpb.toFixed(4)}</span>
            </p>
          )}
          {data.trainingFlops !== undefined && (
            <p>
              <span className="text-gray-600">Training FLOPs:</span>{' '}
              <span className="font-medium">{data.trainingFlops.toExponential(2)}</span>
            </p>
          )}
          {data.modelFlops !== undefined && (
            <p>
              <span className="text-gray-600">Model FLOPs:</span>{' '}
              <span className="font-medium">{data.modelFlops.toExponential(2)}</span>
            </p>
          )}
          {data.modelSize !== undefined && (
            <p>
              <span className="text-gray-600">Model Size:</span>{' '}
              <span className="font-medium">{(data.modelSize / 1e6).toFixed(1)}M params</span>
            </p>
          )}
          {data.tokens !== undefined && data.tokens > 0 && (
            <p>
              <span className="text-gray-600">Tokens:</span>{' '}
              <span className="font-medium">{data.tokens.toExponential(2)}</span>
            </p>
          )}
        </div>
      </div>
    );
  };

  const xAxisLabel = xAxis === 'training_hardware_flops' ? 'Training Hardware FLOPs' : 'Model FLOPs';
  const yAxisLabel = yAxis === 'absolute' ? 'C4-EN BPB' : '% Difference from Baseline';

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

  const handleDownloadChart = async () => {
    if (!chartRef.current) return;

    setIsDownloading(true);

    try {
      const dataUrl = await domtoimage.toPng(chartRef.current, {
        bgcolor: '#ffffff',
        width: chartRef.current.offsetWidth,
        height: chartRef.current.offsetHeight,
        style: {
          backgroundColor: '#ffffff'
        },
        filter: (node: any) => {
          if (node.classList && node.classList.contains('recharts-tooltip-wrapper')) {
            return false;
          }

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

  const CustomLegend = (props: any) => {
    const { payload } = props;
    if (!payload || !payload.length) return null;

    const forecastMap = new Map<string, number>();
    if (chartData.type === 'scaling') {
      chartData.groups.forEach(group => {
        forecastMap.set(group.name, group.forecastedBpbAt1e22);
      });
    }

    const sortedPayload = [...payload]
      .filter((entry: any, index: number, self: any[]) => self.findIndex(e => e.value === entry.value) === index)
      .sort((a: any, b: any) => {
        const aName = a.value;
        const bName = b.value;

        if (aName === 'All Runs') return -1;
        if (bName === 'All Runs') return 1;

        if (aName === 'Baseline (AdamW)') return -1;
        if (bName === 'Baseline (AdamW)') return 1;

        const aForecast = forecastMap.get(aName) ?? Infinity;
        const bForecast = forecastMap.get(bName) ?? Infinity;
        return aForecast - bForecast;
      });

    const MAX_LEGEND_ROWS = 3;
    const legendItemGap = 12;
    const legendRowHeight = isMobile ? 40 : 32;
    const legendMaxHeight = MAX_LEGEND_ROWS * legendRowHeight + (MAX_LEGEND_ROWS - 1) * legendItemGap;

    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          paddingTop: isMobile ? '16px' : '12px',
          paddingBottom: '0px'
        }}
      >
        <div
          style={{
            display: 'flex',
            gap: '12px',
            flexWrap: 'wrap',
            paddingLeft: '8px',
            paddingRight: '8px',
            justifyContent: 'center',
            maxHeight: `${legendMaxHeight}px`,
            overflowY: 'scroll'
          }}
        >
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
                  minHeight: isMobile ? '40px' : '32px',
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
                onChange={() => setXAxis('training_hardware_flops')}
                className="mr-2"
              />
              Training Hardware FLOPs
            </label>
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                value="model_flops"
                checked={xAxis === 'model_flops'}
                onChange={() => setXAxis('model_flops')}
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
                onChange={() => setYAxis('absolute')}
                className="mr-2"
              />
              Absolute BPB
            </label>
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                value="relative"
                checked={yAxis === 'relative'}
                onChange={() => setYAxis('relative')}
                className="mr-2"
              />
              Relative to Baseline
            </label>
          </div>
        </div>
      </div>

      <div ref={chartRef} style={{ position: 'relative' }}>
        {trackId === 'scaling' && (
          <div
            style={{
              position: 'absolute',
              top: isMobile ? '28px' : '32px',
              right: isMobile ? '40px' : '80px',
              zIndex: 10,
              pointerEvents: 'none',
              opacity: 0.85
            }}
          >
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
            <ScatterChart
              margin={{
                top: 20,
                right: isMobile ? 10 : 60,
                bottom: 20,
                left: isMobile ? 40 : 80
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="x"
                name={xAxisLabel}
                scale="log"
                domain={['auto', 'auto']}
                label={{ value: xAxisLabel, position: 'insideBottom', offset: -10 }}
                tickFormatter={value => value.toExponential(0)}
                ticks={xTicks}
              />
              <YAxis
                type="number"
                dataKey="y"
                name={yAxisLabel}
                domain={['auto', 'auto']}
                label={{ value: yAxisLabel, angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }}
                tickFormatter={value => (yAxis === 'relative' ? `${value.toFixed(1)}%` : value.toFixed(3))}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend content={<CustomLegend />} />
              {yAxis === 'relative' && (
                <ReferenceLine y={0} stroke="#000000" strokeDasharray="5 5" strokeWidth={2} />
              )}
              <Scatter
                name="All Runs"
                data={hiddenLegendItems.has('All Runs') ? [] : chartData.background}
                fill="rgba(156,163,175,0.3)"
              />
              {yAxis === 'relative' && (
                <Scatter
                  name="Baseline (AdamW)"
                  data={
                    hiddenLegendItems.has('Baseline (AdamW)')
                      ? []
                      : chartData.background.filter(p => p.name.toLowerCase().includes('adamw'))
                  }
                  fill="#000000"
                />
              )}
              {chartData.groups.map((group) => (
                <Scatter
                  key={group.name}
                  name={group.name}
                  data={hiddenLegendItems.has(group.name) ? [] : group.data}
                  fill={getGroupColor(group.name)}
                />
              ))}
              {chartData.groups.map((group) => (
                <Scatter
                  key={`${group.name}-fit`}
                  name={group.name}
                  data={hiddenLegendItems.has(group.name) ? [] : group.regression}
                  fill="none"
                  line={{ stroke: getGroupColor(group.name), strokeWidth: 2, strokeDasharray: '5 5' }}
                  shape={() => null}
                  legendType="none"
                />
              ))}
            </ScatterChart>
          ) : (
            <ScatterChart
              margin={{
                top: 20,
                right: isMobile ? 10 : 60,
                bottom: 20,
                left: isMobile ? 10 : 60
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="x"
                name={xAxisLabel}
                scale="log"
                domain={['auto', 'auto']}
                label={{ value: xAxisLabel, position: 'insideBottom', offset: -10 }}
                tickFormatter={value => value.toExponential(0)}
                ticks={xTicks}
              />
              <YAxis
                type="number"
                dataKey="y"
                name={yAxisLabel}
                domain={chartData.yDomain}
                label={{ value: yAxisLabel, angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }}
                tickFormatter={value => (yAxis === 'relative' ? `${value.toFixed(1)}%` : value.toFixed(3))}
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
              {yAxis === 'relative' && (
                <ReferenceLine y={0} stroke="#000000" strokeDasharray="5 5" strokeWidth={2} />
              )}
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
