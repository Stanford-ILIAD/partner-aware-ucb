# cooperative-decentralized-bandits
This code learns reward functions from human preferences in various tasks by actively generating batches of scenarios and querying a human expert.

Companion code to AAAI 2022 submission:  
Erdem Biyik, Anusha Lalitha, Rajarshi Saha, Andrea Goldsmith, Dorsa Sadigh. **"Partner-Aware Algorithms in Decentralized Cooperative Bandit Teams"**. *Thirty-Sixth AAAI Conference on Artificial Intelligence*, Feb. 2022. *(Submitted)*

## Dependencies
You need to have the following libraries with [Python3](http://www.python.org/downloads):
* [gym 0.17.1](https://gym.openai.com/)
* [NumPy 1.18.1](https://numpy.org/)

## Usage
You can run the following command to run the game once with the specified seed for randomness and the specified horizon.
```bash
  python main.py --seed 0 --horizon 2000
```

## Setting up the Multi-Agent MAB environment
In the [main.py](main.py), you can play with the agents' reward observabilities (line 12), number of actions (line 13). You can also experiment with more than 2 agents by having larger arrays.

By default, the reward matrix of the environment is random where each entry comes from a uniform distribution between 0 and 1. To experiment with a fixed reward matrix, you can enable line 21 of [main.py](main.py) to specify a matrix. Dimension i corresponds to the actions of agent i.

## Setting up the Agents
[agents.py](agents.py) includes a number of agent types we implemented. You can plug them in the [main.py](main.py) (line 24 and 25) to see how they perform together.

Currently, the following agents are available:
* **BernoulliGameHumanAgent**: A human operator plays the game using the command-line
* **BernoulliGameOptimalAgent**: Knows the true reward matrix and always takes the optimal action
* **BernoulliGameCentralizedThompsonSamplingAgent**: A central controller that implements Thompson sampling (all agents have to be the same kind, because it is centralized)
* **BernoulliGameCentralizedUCBAgent**: A central controller that implements UCB (all agents have to be the same kind, because it is centralized)
* **BernoulliGameNaiveThompsonSamplingAgent**: Naive extension of Thompson sampling to the decentralized multi-agent case
* **BernoulliGameNaiveUCBAgent**: Naive extension of UCB to the decentralized multi-agent case
* **BernoulliGamePartnerAwareUCBAgent**: Our partner-aware UCB algorithm
* **BernoulliGamePartnerAwareThompsonAgent**: Our partner-aware approach implemented with Thompson sampling instead of UCB

