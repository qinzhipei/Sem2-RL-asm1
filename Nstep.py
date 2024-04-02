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

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        T = len(states) - 1  # Length of the episode
        for t in range(T):
                G = sum([self.gamma ** i * rewards[t+i] for i in range(min(n, T - t))]) #n-step return
        #min(n, T - t)确保只累加最多n步的奖励，如果episode在这之前结束则提前停止'''
                if t + n < T: #n步后episode未结束
                   G += self.gamma ** n * np.max(self.Q_sa[states[t+n]]) #add the estimated Q value
                   #self.Q_sa[states[t+n], actions[t+n]] 
           
                # Update the Q value for the state-action pair at time t
                s, a = states[t], actions[t]
                self.Q_sa[s, a] += self.learning_rate * (G - self.Q_sa[s, a])        
                if done:
                   break
        pass

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, epsilon,
                   policy='egreedy',  temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    state = env.reset()
    for timestep in range(n_timesteps):
        
        action = pi.select_action(state)
        episode_states = [state]
        episode_actions = [action]
        episode_rewards = []

        for _ in range(max_episode_length):
            next_state, reward, done = env.step(action)
            episode_rewards.append(reward)
            episode_states.append(next_state)

            if done:
                break

            next_action = pi.select_action(next_state)
            episode_actions.append(next_action)
            action = next_action

        pi.update(episode_states, episode_actions, episode_rewards, done, n)

        if timestep % eval_interval == 0:
            eval_return = pi.evaluate(eval_env)
            eval_timesteps.append(timestep)
            eval_returns.append(eval_return)
    
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
        
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    epsilon=0.1
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy,  temp, plot,epsilon,n=5)
    
    
if __name__ == '__main__':
    test()
