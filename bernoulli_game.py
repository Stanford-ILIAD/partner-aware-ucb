import numpy as np
import gym
from gym.utils import seeding

class BernoulliEnv(gym.Env):
    def __init__(self, horizon, num_arms = [2, 2], p = [1., 0.5], prior_alpha=1, prior_beta=1):
        # horizon is T (how many times each agent is going to make pulls)
        # p is the observability for each agent (1 is full observable)
        # num_arms is the sequence (|A_1|, |A_2|, ...)
        # prior_alpha and prior_beta are the parameters for the Beta distribution for theta values
        assert len(num_arms) == len(p), 'num_arms and p must have the same number of entries, equal to the number of agents'
        assert prior_alpha >= 0 and prior_beta >= 0 , 'prior_alpha and prior_beta must be nonnegative'
        
        self.horizon = horizon
        self.n = len(num_arms)
        self.num_arms = num_arms
        self.p = p
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        self.np_random, seed = seeding.np_random(0)
        self.reset()
        
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def reset(self):
        self.t = 0
        self.alpha = [self.prior_alpha * np.ones(self.num_arms) for _ in range(self.n)] # don't do []*2 because it will have the same reference
        self.beta = [self.prior_beta * np.ones(self.num_arms) for _ in range(self.n)]
        self.theta = self.np_random.beta(self.prior_alpha, self.prior_beta, size = self.num_arms)
        self.cumulative_team_regret = 0.
        return self._get_obs()
        
    @property
    def max_expected_team_reward(self):
        return np.max(self.theta) * np.mean(self.p)
        
    def _get_obs(self):
        # after each time step, the agents are going to observe the alpha and beta values at that time
        # the observation for each agent will be a vector of 2*|A_1|*|A_2|*... elements
        # the first |A_1|*|A_2|*... elements correspond to the alpha values, and the second to the beta values.
        # as an example, consider the following codelet.
        #
        # obs_n = self._get_obs()
        # alphas = np.reshape(obs_n[0][:len(obs)//2], self.num_arms)
        # betas  = np.reshape(obs_n[0][len(obs)//2:], self.num_arms)
        #
        # this code will give the matrix of alphas and betas for agent 0.
        # calling alphas[0,3], for example, gives the alpha value (for agent 0) for the 0th arm of agent 0 and 3rd arm of agent 1.
        
        return [np.concatenate((self.alpha[i].reshape(-1), self.beta[i].reshape(-1))) for i in range(self.n)]
        
    def step(self, actions):
        # actions is a list that contains the action for each agent
        assert len(actions) == self.n, 'actions must be a list that contains n elements where n is the number of agents'
        assert np.all([0 <= actions[i] < self.num_arms[i] for i in range(self.n)]), 'invalid action'
        
        self.t += 1
        effective_theta = self.theta[tuple(actions)]
        observable_reward = (self.np_random.rand() < effective_theta) + 0.
        rews = observable_reward * np.array([self.np_random.rand() < self.p[i] for i in range(self.n)])
        
        for i in range(self.n):
            self.alpha[i][tuple(actions)] += rews[i]
            self.beta[i][tuple(actions)] += 1 - rews[i]
            
        self.cumulative_team_regret += np.max(self.theta) - effective_theta
        return self._get_obs(), rews, self.t >= self.horizon, {}
        
        
     # the functions below are only for analysis
    @property
    def regret(self):
        return self.cumulative_team_regret
        
    @property
    def optimal_action(self):
        return np.unravel_index(self.theta.argmax(), self.theta.shape)
