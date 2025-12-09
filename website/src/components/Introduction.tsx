export function Introduction() {
  return (
    <div className="bg-white rounded-lg shadow p-8 mb-8">
      <div className="flex flex-col md:flex-row justify-between items-start gap-4 mb-6">
        <h2 className="text-3xl text-gray-900">What is Speedrun?</h2>
        <a 
          href="https://github.com/marin-community/marin/blob/main/docs/tutorials/submitting-speedrun.md" 
          target="_blank" 
          rel="noopener noreferrer"
          className="inline-flex items-center px-6 py-3 border border-transparent rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-150 whitespace-nowrap"
        >
          Get started with Speedrun
        </a>
      </div>
      
      <div className="space-y-4 text-gray-600 leading-relaxed text-lg">
        <p>
          Speedrun is a community-driven initiative by the{' '}
          <a 
            href="https://marin.community/" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-700 transition-colors duration-150"
          >
            Marin project
          </a>
          {' '}to track and optimize the training efficiency of large language models. 
          Have a new architecture or training procedure that you think is more efficient? Participate in the Marin speedrun competition (inspired by 
          the{' '}
          <a 
            href="https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#world-record-history" 
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:text-blue-700 transition-colors duration-150"
          >
            nanogpt speedrun
          </a>
          ), pick your compute budget, and create the fastest method to train a model to a certain quality!
        </p>
        
        <p>
          On this page, you can find leaderboards for different speedrun tracks, each targeting a specific loss threshold. You can click on any run to view the code that
          generated it, or view the Weights & Biases link for the model! We also track the overall Pareto frontier of models, allowing us to track efficiency-performance tradeoffs across all tracks.
        </p>
        
        <p>
          We invite you to join us in the search for more performant and efficient training methods!
        </p>
      </div>
    </div>
  );
}
