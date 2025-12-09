interface Track {
  id: string;
  name: string;
  color: string;
}

interface TrackTabsProps {
  tracks: Track[];
  selectedTrack: string;
  onSelectTrack: (trackId: string) => void;
}

export function TrackTabs({ tracks, selectedTrack, onSelectTrack }: TrackTabsProps) {
  return (
    <div className="flex justify-center mb-8">
      <div className="bg-white rounded-lg shadow p-1 inline-flex gap-1">
        {tracks.map(track => (
          <button
            key={track.id}
            onClick={() => onSelectTrack(track.id)}
            className={`px-6 py-3 rounded-md transition-all duration-150 capitalize ${
              selectedTrack === track.id
                ? 'bg-blue-600 text-white shadow-md'
                : 'text-gray-700 hover:bg-gray-100'
            }`}
          >
            {track.name}
          </button>
        ))}
      </div>
    </div>
  );
}
