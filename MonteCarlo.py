#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        G = 0
        for t in range(len(states) - 1, -1, -1):  # Iterate backwards through the episode
           G = self.gamma * G + rewards[t]  
           state, action = states[t], actions[t]
           if (state, action) not in self.Q_sa:
               self.Q_sa[(state, action)] = 0
           self.Q_sa[(state, action)] += self.learning_rate * (G - self.Q_sa[(state, action)])
           pass

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    for timestep in range(n_timesteps):
        states, actions, rewards = [], [], []
        state = env.reset()
        states.append(state)
        
     
        for t in range(max_episode_length):
            action = pi.select_action(state, policy, epsilon)  
            states.append(state)
            actions.append(action)
            
            next_state, reward, done = env.step(action)  
            rewards.append(reward)
            
            if done:
                break  
            state = next_state
            
        if timestep % eval_interval == 0:          
           eval_return = pi.evaluate(eval_env)  
           eval_timesteps.append(timestep)
           eval_returns.append(eval_return)
        if plot:
           env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

                 
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()
