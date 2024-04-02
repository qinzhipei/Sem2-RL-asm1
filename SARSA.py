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

class SarsaAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,a_next,done, alpha, gamma):
        Q_t = self.Q_sa[s, a]
        Q_t2 = self.Q_sa[s_next, a_next] if not done else 0
        target = r + gamma * Q_t2
        self.Q_sa[s, a] += alpha * (target - Q_t)
        pass

        
'''def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=0.1, temp=None, plot=True, eval_interval=500):
     runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep  
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your SARSA algorithm here!
    for timestep in range(n_timesteps):
        state = env.reset() # 获取环境的初始状态。
        action = pi.select_action(state, policy, epsilon, temp)  
        done = False # 设置一个标志，表示是否达到终止状态。
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = pi.select_action(next_state,policy, epsilon, temp)  # Select the next action

        # SARSA update
            pi.update(state, action, reward, next_state, next_action, done, learning_rate, gamma)

    # Check if the state is terminal
        if done:
            state = env.reset()  # Reset the environment
            action = pi.select_action(state, policy, epsilon, temp)  # Reset the action
        else:
            state = next_state  # Move to the next state
            action = next_action  # Move to the next action

        # Evaluate the policy every eval_interval steps
        if timestep % eval_interval == 0:
            eval_return = pi.evaluate(eval_env)  # Run evaluation episodes here
            eval_timesteps.append(timestep)
            eval_returns.append(eval_return)

        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution

    return np.array(eval_returns), np.array(eval_timesteps) '''


def sarsa(n_timesteps, learning_rate, gamma, eval_interval, policy='egreedy', epsilon=0.1, temp=None, plot=True):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    state = env.reset() # Initialize the state
    action = pi.select_action(state, policy, epsilon, temp) 
    for timestep in range(n_timesteps):
        
            next_state, reward, done = env.step(action)  # Take action and observe next state and reward
            next_action = pi.select_action(next_state, policy, epsilon, temp)  # Select next action
            pi.update(state, action, reward, next_state, next_action, done, learning_rate, gamma)  # SARSA update

            state = next_state if not done else env.reset()  # Update state or reset if done
            action = next_action if not done else pi.select_action(state, policy, epsilon, temp)
                
            if timestep % eval_interval == 0 or done:
                eval_return = pi.evaluate(eval_env)  # Evaluate the policy
                eval_timesteps.append(timestep)
                eval_returns.append(eval_return)
                
            if plot:
                env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1) 
                # Plot the Q-value estimates
            if done:  # Start new episode if previous one ended
               state = env.reset()
               action = pi.select_action(state, policy, epsilon, temp)
            print(eval_returns)
    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 5000
    eval_interval = 500
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    eval_returns, eval_timesteps = sarsa(n_timesteps, learning_rate, gamma, eval_interval, policy, epsilon, temp, plot)
    print(eval_returns,eval_timesteps)
    
if __name__ == '__main__':
    test()
