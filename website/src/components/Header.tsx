export function Header() {
  return (
    <header className="bg-gradient-to-r from-gray-900 to-gray-800 text-white shadow-lg">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center gap-5 group hover:opacity-90 transition-opacity duration-150">
          <img 
            src="https://raw.githubusercontent.com/marin-community/speedrun/refs/heads/main/website/src/assets/marin-logo.png" 
            alt="Marin Logo" 
            className="h-14 w-14 object-contain"
          />
          <div className="flex flex-col justify-center">
            <h1 className="text-3xl leading-tight">Marin Speedrun - Leaderboard</h1>
            <p className="text-gray-300 text-sm mt-0.5">Community-driven model training leaderboard</p>
          </div>
        </div>
      </div>
    </header>
  );
}
