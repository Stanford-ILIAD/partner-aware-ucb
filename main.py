import argparse
from bernoulli_game import *
from agents import *

def parse_args():
    parser = argparse.ArgumentParser("Cooperative Decentralized Bandits Experiments")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--horizon", type=int, default=2000, help="horizon of the episode")
    return parser.parse_args()
    
def main(arglist):
    p = [1.0, 0.5]
    num_actions = [10, 10]
    
    horizon = arglist.horizon
    num_agents = len(p)

    env = BernoulliEnv(horizon, num_actions, p)
    env.seed(arglist.seed)
    obs_n, done = env.reset(), False
    #env.theta = np.array([[0.8,0.4],[0.2,0.6]])

    agents = []
    agents.append(BernoulliGamePartnerAwareUCBAgent(0, env, W=1, L=2, c=0.025)) # leader (id 0)
    agents.append(BernoulliGamePartnerAwareUCBAgent(1, env, W=1, L=1, c=0.025)) # follower (id 1)
    
    for i in range(num_agents):
        agents[i].seed(arglist.seed + i)

    res = []
    while not done:
        actions = [agents[i](obs_n[i]) for i in range(num_agents)]
        obs_n, rews, done, _ = env.step(actions)
        for i in range(num_agents):
            agents[i].observe_actions(actions)
            agents[i].obtain_reward(rews[i])
        res.append(env.regret)
        
    print('Cumulative Team Regret = ' + str(env.regret))

if __name__ == '__main__':
    arglist = parse_args()
    main(arglist)