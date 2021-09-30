import numpy as np
from gym.utils import seeding
import copy
import os
import pickle
import pdb
from scipy.stats import mode


class Agent(): # abstract class
    def __init__(self, id, num_arms):
        super(Agent, self).__init__()
        self.id = id
        self.num_arms = num_arms
        
    def __call__(self, obs):
        raise NotImplementedError
        
    def obtain_reward(self, rew):
        pass
        
    def observe_actions(self, actions):
        pass
        
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
        
class BernoulliGameAgent(Agent): # abstract class
    def __init__(self, id, env):
        super(BernoulliGameAgent, self).__init__(id, env.num_arms)
        #self.p = env.p
        
        
class BernoulliGameHumanAgent(BernoulliGameAgent):
    def __init__(self, id, env):
        super(BernoulliGameHumanAgent, self).__init__(id, env)
        
    def __call__(self, obs):
        alphas = np.reshape(obs[:len(obs)//2], self.num_arms)
        betas  = np.reshape(obs[len(obs)//2:], self.num_arms)
        print('Current alpha values:')
        print(alphas)
        print('-----------------------\nCurrent beta values:')
        print(betas)
        print('-----------------------\nYou are agent ' + str(self.id))
        action = -1
        while not (0 <= action < self.num_arms[self.id]):
            action = int(input("Enter your action: "))
        return action
        
        
class BernoulliGameOptimalAgent(BernoulliGameAgent): # always takes the optimal action
    def __init__(self, id, env):
        super(BernoulliGameOptimalAgent, self).__init__(id, env)
        self.env = env
        
    def __call__(self, obs):
        return self.env.optimal_action[self.id]
        
        
class BernoulliGameCentralizedThompsonSamplingAgent(BernoulliGameAgent):
    # This class is full of hacks and the reason is the game was designed only for decentralized agents
    # Specifically, the agent exploits the environment to share information with other agents through env.hacky_rews and env.hacky_theta
    def __init__(self, id, env):
        super(BernoulliGameCentralizedThompsonSamplingAgent, self).__init__(id, env)
        self.seed(0) # to create self.np_random. Other seeds can be given externally
        self.env = env
        if id == 0: # 0th agent will do the computations, the others will follow
            self.centralized_alpha = copy.deepcopy(self.env.alpha[id])
            self.centralized_beta = copy.deepcopy(self.env.beta[id])
            self.env.hacky_rews = []
        
    def observe_actions(self, actions):
        if self.id == 0:
            self.last_actions = actions
        
    def obtain_reward(self, rew):
        if self.id == 0:
            self.env.hacky_rews = []
        self.env.hacky_rews.append(rew)
        
    def __call__(self, obs):
        if self.id == 0 and self.env.hacky_rews:
            rew_obtained = 1 if np.any(np.array(self.env.hacky_rews) > 0) else 0
            self.centralized_alpha[tuple(self.last_actions)] += rew_obtained
            self.centralized_beta[tuple(self.last_actions)] += 1 - rew_obtained
        if self.id == 0:
            self.env.hacky_theta = self.np_random.beta(self.centralized_alpha, self.centralized_beta)
        return np.unravel_index(self.env.hacky_theta.argmax(), self.env.hacky_theta.shape)[self.id]
        
        
class BernoulliGameCentralizedUCBAgent(BernoulliGameAgent):
    # This class is full of hacks and the reason is the game was designed only for decentralized agents
    # Specifically, the agent exploits the environment to share information with other agents through env.hacky_rews and env.hacky_theta
    def __init__(self, id, env, c):
        super(BernoulliGameCentralizedUCBAgent, self).__init__(id, env)
        self.seed(0) # to create self.np_random. Other seeds can be given externally
        self.env = env
        self.c = c
        self.action_counts = np.zeros(self.num_arms, dtype=int)
        if id == 0: # 0th agent will do the computations, the others will follow
            self.centralized_alpha = copy.deepcopy(self.env.alpha[id])
            self.centralized_beta = copy.deepcopy(self.env.beta[id])
            self.env.hacky_rews = []
        
    def observe_actions(self, actions):
        self.action_counts[tuple(actions)] += 1
        if self.id == 0:
            self.last_actions = actions
        
    def obtain_reward(self, rew):
        if self.id == 0:
            self.env.hacky_rews = []
        self.env.hacky_rews.append(rew)
        
    def __call__(self, obs):
        if self.id == 0 and self.env.hacky_rews:
            rew_obtained = 1 if np.any(np.array(self.env.hacky_rews) > 0) else 0
            self.centralized_alpha[tuple(self.last_actions)] += rew_obtained
            self.centralized_beta[tuple(self.last_actions)] += 1 - rew_obtained
        if np.isclose(self.action_counts.sum(), 0):
            return self.np_random.randint(self.num_arms[self.id])
        if self.id == 0:
            reward_estimates = self.centralized_alpha / (self.centralized_alpha + self.centralized_beta)
            self.env.hacky_theta = reward_estimates + np.sqrt(2 * self.c * np.log(np.sum(self.action_counts)) / np.maximum(1e-6,self.action_counts))
        return np.unravel_index(self.env.hacky_theta.argmax(), self.env.hacky_theta.shape)[self.id]


class BernoulliGameNaiveThompsonSamplingAgent(BernoulliGameAgent):
    def __init__(self, id, env):
        super(BernoulliGameNaiveThompsonSamplingAgent, self).__init__(id, env)
        self.seed(0) # to create self.np_random. Other seeds can be given externally
        
    def __call__(self, obs):
        alphas = np.reshape(obs[:len(obs)//2], self.num_arms)
        betas  = np.reshape(obs[len(obs)//2:], self.num_arms)
        theta = self.np_random.beta(alphas, betas)
        return np.unravel_index(theta.argmax(), theta.shape)[self.id]

class BernoulliGameNaiveUCBAgent(BernoulliGameAgent):
    def __init__(self, id, env, c):
        super(BernoulliGameNaiveUCBAgent, self).__init__(id, env)
        self.seed(0) # to create self.np_random. Other seeds can be given externally
        self.c = c
        self.action_counts = np.zeros(self.num_arms, dtype=int)
        
    def observe_actions(self, actions):
        self.action_counts[tuple(actions)] += 1
        
    def __call__(self, obs):
        if np.isclose(self.action_counts.sum(),0):
            return self.np_random.randint(self.num_arms[self.id])
        alphas = np.reshape(obs[:len(obs)//2], self.num_arms)
        betas  = np.reshape(obs[len(obs)//2:], self.num_arms)
        reward_estimates = alphas / (alphas + betas)
        obj = reward_estimates + np.sqrt(2 * self.c * np.log(np.sum(self.action_counts)) / np.maximum(1e-6,self.action_counts))
        return np.unravel_index(obj.argmax(), obj.shape)[self.id]


class BernoulliGamePartnerAwareUCBAgent(BernoulliGameAgent):
    def __init__(self, id, env, W=25, L=1, c=0.025):
        super(BernoulliGamePartnerAwareUCBAgent, self).__init__(id, env)
        self.seed(0) # to create self.np_random. Other seeds can be given externally
        self.env = env
        self.all_actions = np.zeros((0,self.env.n))
        self.W = W
        self.L = L
        self.c = c
        self.action_counts = np.zeros(self.num_arms, dtype=int)
        self.last_action = None
        
    def observe_actions(self, actions):
        self.all_actions = np.vstack((self.all_actions, actions))
        self.all_actions = self.all_actions[-self.W:]
        self.action_counts[tuple(actions)] += 1
        
    def __call__(self, obs):
        if self.env.t % self.L >= 1:
            return self.last_action
            
        if np.isclose(self.action_counts.sum(),0):
            self.last_action = self.np_random.randint(self.num_arms[self.id])
            return self.last_action
            
        alphas = np.reshape(obs[:len(obs)//2], self.num_arms)
        betas  = np.reshape(obs[len(obs)//2:], self.num_arms)
        act_counts = self.action_counts.copy()
        
        selfidx = 0
        for agent_id in range(self.env.n):
            if self.env.p[agent_id] > self.env.p[self.id] or (np.isclose(self.env.p[agent_id], self.env.p[self.id]) and agent_id < self.id):
                if len(self.all_actions) > 0:
                    p = [np.mean(self.all_actions[:,agent_id]==i) for i in range(self.num_arms[agent_id])]
                else:
                    p = [1./self.num_arms[agent_id]]*self.num_arms[agent_id]
                agent_arm = self.np_random.choice(self.num_arms[agent_id], p=p)
                alphas = alphas[agent_arm]
                betas = betas[agent_arm]
                act_counts = act_counts[agent_arm]
            else:
                alphas = alphas.transpose([*range(1,len(alphas.shape))] + [0])
                betas = betas.transpose([*range(1,len(betas.shape))] + [0])
                act_counts = act_counts.transpose([*range(1,len(act_counts.shape))] + [0])
                if agent_id < self.id:
                    selfidx += 1        
        
        rew_estimate = alphas / (alphas + betas)
        obj = rew_estimate + np.sqrt(2 * self.c * np.log(self.action_counts.sum()) / np.maximum(1e-6,act_counts))
        self.last_action = np.unravel_index(obj.argmax(), obj.shape)[selfidx]
        return self.last_action
        

class BernoulliGamePartnerAwareThompsonAgent(BernoulliGameAgent):
    def __init__(self, id, env, W=25, L=1):
        super(BernoulliGamePartnerAwareThompsonAgent, self).__init__(id, env)
        self.seed(0) # to create self.np_random. Other seeds can be given externally
        self.env = env
        self.all_actions = np.zeros((0,self.env.n))
        self.W = W
        self.L = L
        self.last_action = None
        
    def observe_actions(self, actions):
        self.all_actions = np.vstack((self.all_actions, actions))
        self.all_actions = self.all_actions[-self.W:]

    def __call__(self, obs):
        if self.env.t % self.L >= 1:
            return self.last_action
        
        alphas = np.reshape(obs[:len(obs)//2], self.num_arms)
        betas  = np.reshape(obs[len(obs)//2:], self.num_arms)
        
        selfidx = 0
        for agent_id in range(self.env.n):
            if self.env.p[agent_id] > self.env.p[self.id] or (np.isclose(self.env.p[agent_id], self.env.p[self.id]) and agent_id < self.id):
                if len(self.all_actions) > 0:
                    p = [np.mean(self.all_actions[:,agent_id]==i) for i in range(self.num_arms[agent_id])]
                else:
                    p = [1./self.num_arms[agent_id]]*self.num_arms[agent_id]
                agent_arm = self.np_random.choice(self.num_arms[agent_id], p=p)
                alphas = alphas[agent_arm]
                betas = betas[agent_arm]
            else:
                alphas = alphas.transpose([*range(1,len(alphas.shape))] + [0])
                betas = betas.transpose([*range(1,len(betas.shape))] + [0])
                if agent_id < self.id:
                    selfidx += 1

        theta = self.np_random.beta(alphas, betas)
        self.last_action = np.unravel_index(theta.argmax(), theta.shape)[selfidx]
        return self.last_action

