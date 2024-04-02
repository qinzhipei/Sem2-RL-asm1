# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:52:48 2024

@author: Administrator
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        G_t = r + self.gamma * np.max(self.Q_sa[s_next]) #compute gt
        self.Q_sa[s, a] += self.learning_rate * (G_t - self.Q_sa[s, a])# Update the Q value
        #self.gamma:discount factor self.alpha:learning rate
        pass

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    for timestep in range(n_timesteps):
        state = env.reset() # 获取环境的初始状态。
        done = False # 设置一个标志，表示是否达到终止状态。
        
        while not done:
            
            action = agent.select_action(state, policy, epsilon, temp) #根据当前policy选择行动          
            next_state, reward, done = env.step(action) #使用env.step函数，环境根据行动返回下一个状态、奖励和是否结束。           
            agent.update(state, action, reward, next_state, done) # 使用观察到的数据更新 Q 表。
            state = next_state if not done else env.reset() # 如果未结束，则状态更新为下一个状态，否则重置为初始状态。
        
        if timestep % eval_interval == 0: # 如果到了评估间隔，执行评估。         
            eval_return = agent.evaluate(eval_env)  #eval_returns:当前政策的平均回报。
            eval_timesteps.append(timestep)
            eval_returns.append(eval_return)
            # 存储评估时间步和回报。
    
        if plot:
           env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

        
    return np.array(eval_returns), np.array(eval_timesteps)   



def test():
    
    n_timesteps = 10001
    eval_interval= 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    
    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()

