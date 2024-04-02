#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

'''Environment.py：该文件生成环境。 
运行该文件以查看具有随机选择的操作的环境演示。 检查类方法并确保您理解它们。 
使用 render()，您可以在执行期间交互式地可视化环境。 
如果您提供 Q_sa（Q 值表），环境还将显示每个状态中每个操作的 Q 值估计，
同时切换plot_optimal_policy 还将显示最优策略的箭头。 
尝试这些设置，并确保您理解它们。'''

import matplotlib
matplotlib.use('Qt5Agg') # 'TkAgg'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle,Arrow

class StochasticWindyGridworld:
    ''' Stochastic version of WindyGridworld 
        (based on Sutton & Barto, Example 6.5 at page 130, see http://incompleteideas.net/book/RLbook2020.pdf)
        Compared to the book version, the vertical wind is now stochastic, and only blows 80% of times
    '''
    
    def __init__(self,initialize_model=True):
        self.height = 7 # 7
        self.width = 10 # 10
        self.shape = (self.width, self.height)
        self.n_states = self.height * self.width
        self.n_actions = 4
        self.action_effects = {
                0: (0, 1),  # up
                1: (1, 0),   # right
                2: (0, -1),   # down
                3: (-1, 0),  # left
                }
        self.start_location = (0,3)
        self.winds = (0,0,0,1,1,1,2,2,1,0)
        self.wind_blows_proportion = 0.9         

        self.reward_per_step = -1.0 # default reward on every step that does not reach a goal
        self.goal_locations = [[7,3]] # [[6,2]] a vector specifying the goal locations in [[x1,y1],[x2,y2]] format
        self.goal_rewards = [100] # a vector specifying the associated rewards with the goals in self.goal_locations, in [r1,r2] format
        
        # Initialize model
        self.initialize_model = initialize_model
        if self.initialize_model:
            self._construct_model()
            
        # Initialize figures
        self.fig = None
        self.Q_labels = None
        self.arrows = None
        
        # Set agent to the start location
        self.reset() 

    def reset(self):
        ''' set the agent back to the start location '''
        self.agent_location = np.array(self.start_location)
        s = self._location_to_state(self.agent_location)
        return s
    
    def step(self,a):
        ''' Forward the environment based on action a, really affecting the agent location  
        Returns the next state, the obtained reward, and a boolean whether the environment terminated '''
        self.agent_location += self.action_effects[a] # effect of action
        #基于代理选择的动作a（如上、下、左、右）来更新代理的位置
        self.agent_location = np.clip(self.agent_location,(0,0),np.array(self.shape)-1) # bound within grid
        #使用np.clip函数确保代理的位置仍然在网格世界的边界内
        if np.random.rand() < self.wind_blows_proportion: # 风效果应用apply effect of wind 
            self.agent_location[1] += self.winds[self.agent_location[0]] # effect of wind
        self.agent_location = np.clip(self.agent_location,(0,0),np.array(self.shape)-1) # bound within grid
        s_next = self._location_to_state(self.agent_location)    
        
        # Check reward and termination
        goal_present = np.any([np.all(goal_location == self.agent_location) for goal_location in self.goal_locations])
        if goal_present: #如果代理到达目标位置，则该时间步骤被标记为终止（done = True），并根据目标位置获得相应的奖励
            goal_index = np.where([np.all(goal_location == self.agent_location) for goal_location in self.goal_locations])[0][0]
            done = True
            r = self.goal_rewards[goal_index]
        else: 
            done = False
            r = self.reward_per_step           
            
        return s_next, r, done  #返回新的状态（s_next）、在这个时间步骤获得的奖励（r）以及一个布尔值表示是否达到了终止状态

    def model(self,s,a):
        ''' Returns vectors p(s'|s,a) and r(s,a,s') for given s and a.
        Only simulates, does not affect the current agent location '''
        if self.initialize_model: #当环境预先构建并存储所有的状态转移概率和奖励信息时model才可用
            return self.p_sas[s,a], self.r_sas[s,a]
        #r:奖励函数 执行动作a从状态s转移到状态's时所得到的预期奖励
        else:
            raise ValueError("set initialize_model=True when creating Environment")
            

    def render(self,Q_sa=None,plot_optimal_policy=False,step_pause=0.001):
        ''' Plot the environment 
        if Q_sa is provided, it will also plot the Q(s,a) values for each action in each state
        if plot_optimal_policy=True, it will additionally add an arrow in each state to indicate the greedy action '''
        # Initialize figure
        if self.fig == None:
            self._initialize_plot() #如果是首次调用，它会初始化一个图形界面，显示网格世界的布局，包括起点、终点和风的影响
        
        '''如果提供了Q值（Q_sa），该方法会在每个格子中显示对应的Q值。
        这些值代表了在给定状态下采取不同动作的预期回报。
        通过显示这些值，可以直观地看到代理认为哪些动作在某状态下是有利的。'''
        # Add Q-values to plot
        if Q_sa is not None:
            # Initialize labels
            if self.Q_labels is None:
                self._initialize_Q_labels()
            # Set correct values of labels
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    self.Q_labels[state][action].set_text(np.round(Q_sa[state,action],1))

        # Add arrows of optimal policy
        if plot_optimal_policy and Q_sa is not None:
            self._plot_arrows(Q_sa)
            
        # Update agent location
        self.agent_circle.center = self.agent_location+0.5
            
        # Draw figure
        plt.pause(step_pause)  
        #通过step_pause参数，该方法在每次更新后会暂停一小段时间，使得变化可以被人眼观察到  

    '''方便地根据需要使用状态索引(整数）或位置坐标(x, y)坐标'''
    def _state_to_location(self,state):
        ''' bring a state index to an (x,y) location of the agent '''
        return np.array(np.unravel_index(state,self.shape))
    
    def _location_to_state(self,location):
        ''' bring an (x,y) location of the agent to a state index '''
        return np.ravel_multi_index(location,self.shape)
        
    def _construct_model(self):
        ''' Constructs full p(s'|s,a) and r(s,a,s') arrays
            Stores these in self.p_sas and self.r_sas '''
            
        # Initialize transition and reward functions
        #构建并初始化整个环境的状态转移概率矩阵p_sas和奖励函数矩阵r_sas
        p_sas = np.zeros((self.n_states,self.n_actions,self.n_states))
        r_sas = np.zeros((self.n_states,self.n_actions,self.n_states)) + self.reward_per_step # set all rewards to the default value
        
        #遍历所有状态和动作：对于环境中的每个状态（s）和在该状态下可能采取的每个动作（a），计算动作的效果，包括风的随机效应
        for s in range(self.n_states):
            for a in range(self.n_actions):
                s_location = self._state_to_location(s)  
                #如果当前状态是目标（终结）状态之一，那么任何动作都将只导致自身状态的重复（自环），并且不会有额外的奖励（或者说是特定的奖励设置，比如0奖励），以表示游戏结束或达成目标。  
                # if s is goal state (terminal) make it a self-loop without rewards
                #goal_locations = [[7,3]]
                state_is_a_goal = np.any([np.all(goal_location == s_location) for goal_location in self.goal_locations])
                if state_is_a_goal: 
                    # Make actions from this state a self-loop with 0 reward.
                    p_sas[s,a,s] = 1.0 
                    r_sas[s,a,] = np.zeros(self.n_states)  
                else: #模拟了风吹动和不吹动两种情况下代理可能达到的下一状态
                    # check what happens if the wind blows:
                    next_location_with_wind = np.copy(s_location) 
                    next_location_with_wind += self.action_effects[a] # effect of action
                    next_location_with_wind = np.clip(next_location_with_wind,(0,0),np.array(self.shape)-1) # bound within grid
                    next_location_with_wind[1] += self.winds[next_location_with_wind[0]] # Apply effect of wind
                    next_location_with_wind = np.clip(next_location_with_wind,(0,0),np.array(self.shape)-1) # bound within grid
                    next_state_with_wind = self._location_to_state(next_location_with_wind)   
                    
                    # Update p_sas and r_sas
                    p_sas[s,a,next_state_with_wind] += self.wind_blows_proportion
                    for (i,goal) in enumerate(self.goal_locations):
                        if np.all(next_location_with_wind == goal): # reached a goal!
                            r_sas[s,a,next_state_with_wind]  = self.goal_rewards[i]
                    
                    # check what happens if the wind does not blow:
                    next_location_without_wind = np.copy(s_location)
                    next_location_without_wind += self.action_effects[a] # effect of action
                    next_location_without_wind = np.clip(next_location_without_wind,(0,0),np.array(self.shape)-1) # bound within grid
                    next_state_without_wind = self._location_to_state(next_location_without_wind)
    
                    # Update p_sas and r_sas
                    p_sas[s,a,next_state_without_wind] += (1-self.wind_blows_proportion) #如果风没有吹动，代理将直接根据动作a的效果（没有额外的风效果）移动到新的位置
                    for (i,goal) in enumerate(self.goal_locations):
                        if np.all(next_state_without_wind == goal): # reached a goal!
                            r_sas[s,a,next_state_without_wind]  = self.goal_rewards[i] 

        self.p_sas = p_sas
        self.r_sas = r_sas
        return 

    def _initialize_plot(self): #环境可视化
        self.fig,self.ax = plt.subplots()#figsize=(self.width, self.height+1)) # Start a new figure
        self.ax.set_xlim([0,self.width]) #10
        self.ax.set_ylim([0,self.height]) #7
        self.ax.axes.xaxis.set_visible(False) #隐藏坐标轴标签
        self.ax.axes.yaxis.set_visible(False)

        for x in range(self.width):
            for y in range(self.height):
                self.ax.add_patch(Rectangle((x, y),1,1, linewidth=0, facecolor='k',alpha=self.winds[x]/4)) #矩形的颜色深度（alpha值）基于相应位置的风强度       
                self.ax.add_patch(Rectangle((x, y),1,1, linewidth=0.5, edgecolor='k', fill=False))     

        self.ax.axvline(0,0,self.height,linewidth=5,c='k')
        self.ax.axvline(self.width,0,self.height,linewidth=5,c='k')
        self.ax.axhline(0,0,self.width,linewidth=5,c='k')
        self.ax.axhline(self.height,0,self.width,linewidth=5,c='k')

        # Indicate start state 表示起始位置
        self.ax.add_patch(Rectangle(self.start_location,1.0,1.0, linewidth=0, facecolor='b',alpha=0.2))
        self.ax.text(self.start_location[0]+0.05,self.start_location[1]+0.75, 'S', fontsize=20, c='b')

        # Indicate goal states 标示目标位置
        for i in range(len(self.goal_locations)): 
            if self.goal_rewards[i] >= 0:
                colour = 'g'
                text = '+{}'.format(self.goal_rewards[i])
            else:
                colour = 'r'
                text = '{}'.format(self.goal_rewards[i])
            self.ax.add_patch(Rectangle(self.goal_locations[i],1.0,1.0, linewidth=0, facecolor=colour,alpha=0.2))
            self.ax.text(self.goal_locations[i][0]+0.05,self.goal_locations[i][1]+0.75,text, fontsize=20, c=colour)

        # Add agent 表示agent在环境中的位置
        self.agent_circle = Circle(self.agent_location+0.5,0.3)
        self.ax.add_patch(self.agent_circle)
        
    def _initialize_Q_labels(self): 
        #为环境的每个状态和对应的动作初始化Q值标签，并将这些标签添加到可视化界面上
        #Q值：动作价值函数的估计
        self.Q_labels = []
        for state in range(self.n_states):
            state_location = self._state_to_location(state) #对于环境中的每个状态，计算该状态在网格中的位置（(x, y)坐标）
            self.Q_labels.append([])
            for action in range(self.n_actions): #为每个动作添加Q值标签
            #计算该动作对应的Q值标签应该放置的位置。这个位置基于状态的位置加上一个偏移量，以确保Q值标签不会重叠
                plot_location = np.array(state_location) + 0.42 + 0.35 * np.array(self.action_effects[action])
                next_label = self.ax.text(plot_location[0],plot_location[1]+0.03,0.0,fontsize=5)
                self.Q_labels[state].append(next_label)
    
    '''通过分析给定的Q值矩阵Q_sa来确定每个状态的最优动作（或动作们，如果有多个动作具有相同的最大Q值），
    然后在相应的状态位置上绘制一个或多个箭头，指示这些最优动作的方向'''
    def _plot_arrows(self,Q_sa):
        if self.arrows is not None: 
            for arrow in self.arrows:
                arrow.remove() # Clear all previous arrows
        self.arrows=[]
        for state in range(self.n_states):
            plot_location = np.array(self._state_to_location(state)) + 0.5 #计算该状态的中心位置，以此作为箭头的起始点
            max_actions = full_argmax(Q_sa[state]) #通过分析给定的Q值矩阵Q_sa来确定每个状态的最优动作（或动作们，如果有多个动作具有相同的最大Q值），然后在相应的状态位置上绘制一个或多个箭头，指示这些最优动作的方向
            for max_action in max_actions: #计算该状态的中心位置，以此作为箭头的起始点
                new_arrow = arrow = Arrow(plot_location[0],plot_location[1],self.action_effects[max_action][0]*0.2,
                                          self.action_effects[max_action][1]*0.2, width=0.05,color='k')
                ax_arrow = self.ax.add_patch(new_arrow)
                self.arrows.append(ax_arrow)

def full_argmax(x):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max '''
    return np.where(x == np.max(x))[0]            

def test():
    # Hyperparameters
    n_test_steps = 25
    step_pause = 0.01
    
    # Initialize environment and Q-array
    env = StochasticWindyGridworld()
    s = env.reset()
    Q_sa = np.zeros((env.n_states,env.n_actions)) # Q-value array of flat zeros

    # Test
    for t in range(n_test_steps):
        a = np.random.randint(4) # sample random action    
        s_next,r,done = env.step(a) # execute action in the environment
        p_sas,r_sas = env.model(s,a)
        print("State {}, Action {}, Reward {}, Next state {}, Done {}, p(s'|s,a) {}, r(s,a,s') {}".format(s,a,r,s_next,done,p_sas,r_sas))
        env.render(Q_sa=Q_sa,plot_optimal_policy=False,step_pause=step_pause) # display the environment
        if done: 
            s = env.reset()
        else: 
            s = s_next
    
if __name__ == '__main__':
    test()
