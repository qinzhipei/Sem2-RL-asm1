# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:50:57 2024

@author: Administrator
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax
import matplotlib.pyplot as plt

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.threshold = threshold ##
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s): #根据当前状态 s，选择具有最大 Q 值的动作作为最优动作，实现了贪婪策略
        ''' Returns the greedy best action in state s ''' 
        ## Implement the greedy policy: π(s) = arg maxa Q(s, a)
        best_action = np.argmax(self.Q_sa[s])
        return best_action
        
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas 
        # TO DO: Add own code
        pass'''
        ## the implementation of the Bellman equation for Q-value iteration in reinforcement learning.
        new_Q = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa,axis=1)))
        max_abs_error = np.abs(self.Q_sa[s, a] - new_Q) 
        self.Q_sa[s, a] = new_Q #将 Q 值更新到状态-动作值表 Q_sa 中。
        return max_abs_error #print the maximum absolute error after each full sweep
   #计算了单个状态-动作对在当前迭代中的Q值（即 new_Q）与上一次迭代中的Q值（即 self.Q_sa[s, a]）之间的差的绝对值
   


def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    #gamma:discount factor threshold:convergence threshold
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
    
    # Implement Q-value iteration
    max_error = threshold #使用threshold值初始化max_error以进入while循环
    iteration = 0
    while max_error >= threshold:
        max_error = 0  # Reset max error for this iteration
        # Sweep through the state space
        for s in range(env.n_states):
            for a in range(env.n_actions): #嵌套循环：覆盖每个状态的所有动作
                # Get the transition probabilities and rewards for the current state-action pair
                p_sas, r_sas = env.model(s, a)
                # Update Q-values using the Bellman equation
                error = QIagent.update(s, a, p_sas, r_sas)
                # Track the maximum absolute error for convergence check
                max_error = max(max_error, error)
        # Optional: Render the current policy and Q-values
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        print(f"Q-value iteration, iteration {iteration}, max error {max_error}")
        iteration += 1
        
    return QIagent
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
    
    # Implement Q-value iteration
    max_error = threshold
    iteration = 0
    while max_error >= threshold:
        max_error = 0  # Reset max error for this iteration
        # Sweep through the state space
        for s in range(env.n_states):
            for a in range(env.n_actions):
                #调用了StochasticWindyGridworld类，返回 transition probabilities and rewards for the current state-action pair
                p_sas, r_sas = env.model(s, a)
                # Update Q-values using the Bellman equation
                error = QIagent.update(s, a, p_sas, r_sas)
                # Track the maximum absolute error for convergence check
                max_error = max(max_error, error)
        # Optional: Render the current policy and Q-values
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        print(f"Q-value iteration, iteration {iteration}, max error {max_error}")
        iteration += 1
        
    return QIagent

'''def plot_Q_values_stages(Q_values_stages, title="Q-Value Progression"):
    num_stages = len(Q_values_stages)
    fig, axes = plt.subplots(1, num_stages, figsize=(num_stages * 5, 4))
    
    for i, Q_values in enumerate(Q_values_stages):
        cax = axes[i].matshow(Q_values, interpolation='nearest')
        fig.colorbar(cax, ax=axes[i])
        axes[i].set_title(f"Stage {i+1}")
    
    plt.suptitle(title)
    plt.savefig(r'E:\2024 Sem2\RL\Assignment\Figure\Qvalue.png')'''

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env, gamma, threshold)
    #print(QIagent.Q_sa)
    # Record Q-values at different stages
    Q_values_stages = [QIagent.Q_sa.copy()]  # Initial Q-values

    # Print initial Q-values
    #print("Initial Q-values:\n", Q_values_stages[0])

    # view optimal policy
    done = False
    s = env.reset()
    total_reward = 0
    timesteps = 0
    #midpoint_timesteps = (env.n_states * env.n_actions) // 2
    #print(midpoint_timesteps)
    
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        total_reward += r      
        timesteps += 1
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.5)
        s = s_next
        
        # Record midway Q-values
        if timesteps == 8:
            print(QIagent.Q_sa)
            Q_values_stages.append(QIagent.Q_sa.copy())
            # Print midway Q-values
            #print("Midway Q-values:\n", Q_values_stages[1])

    # Record final Q-values
    
    Q_values_stages.append(QIagent.Q_sa.copy())

    # Print final Q-values
    #print("Final Q-values:\n", Q_values_stages[-1])

    # Compute mean reward per timestep under the optimal policy
    mean_reward_per_timestep = total_reward / timesteps if timesteps else 0
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    
    # the converged optimal value at the start
    Q_values_converged = Q_values_stages[-1]
    converged_optimal_value = np.max(Q_values_converged[3])
    print(total_reward,timesteps)
    
    
    return timesteps, converged_optimal_value, total_reward
          
if __name__ == '__main__':
    reward_sum = 0
    optimal_value = 0
    step_sum = 0
    repetition = 20
    for i in range(repetition):
        timesteps, converged_optimal_value, total_reward = experiment()
        optimal_value += converged_optimal_value
        step_sum += timesteps
        reward_sum += total_reward
    print("converged optimal value:{}".format(optimal_value/repetition))
    print(optimal_value/step_sum)
    print(reward_sum/step_sum)
    #print("converged optimal value: {}".format(optimal_value/repetition))
    print("avg timesteps: {}".format(step_sum/repetition))