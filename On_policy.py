# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 01:21:12 2019

@author: YingjieSHAO
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:26:23 2019

@author: user
"""

import gym
from gym import spaces
import numpy as np

import argparse
from itertools import count
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

class Wave(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
     
    def __init__(self):
        self.meshsize = 512
        self.winddir = 90
        self.patchsize = self.meshsize/50
        self.windSpeed = 2
        self.A = 1
        self.g = 9.81
        self.xLim_min = -1
        self.xLim_max = 1
        self.yLim_min = -1
        self.yLim_max = 1
        self.zLim_min = -2e-4
        self.zLim_max = 2e-4
        self.h = 0
        self.sea_depth = 5
        self.air_depth = 5
        self.n1 = 1
        self.n2 = 1.34
        self.beamsize = 10        
        
        self.diameter = 10
        self.angle = 20
        self.action_space = [[np.pi/5000,np.pi/5000],[np.pi/5000,-np.pi/5000],[np.pi/5000,0],[-np.pi/5000,np.pi/5000],[-np.pi/5000,-np.pi/5000],[-np.pi/5000,0],[0,np.pi/5000],[0,-np.pi/5000],[0,0]]
        self.observation_space = spaces.Box(np.array([-self.diameter,-self.diameter]),np.array([-self.diameter,-self.diameter]))
        self.state = None
        np.random.seed(9999)
        self.MEMS_offset_x = 0.001
        self.MEMS_offset_y = 0.001
        
    
    def step(self, action):
        
        self.timestep = 0.01
        self.t0 = self.t0 + self.timestep
        
        #MEMS control
        #If the angle excess the maximum or the minimum angle, teh anlge will be set as the allowed one
        
        self.MEMS_offset_x = 0#self.action_space[action][0]
        self.MEMS_offset_y = 0#self.action_space[action][1]
        if self.MEMS_offset_x<self.MEMS_offset_x_min :
            self.MEMS_offset_x = 0#self.MEMS_offset_x_min
        if self.MEMS_offset_x>self.MEMS_offset_x_max :
            self.MEMS_offset_x = 0#self.MEMS_offset_x_max
        if self.MEMS_offset_y<self.MEMS_offset_y_min :
            self.MEMS_offset_y = 0#self.MEMS_offset_y_min
        if self.MEMS_offset_y>self.MEMS_offset_y_max :
            self.MEMS_offset_y = 0#self.MEMS_offset_y_max
            
            
        #The position of light spot after propogation through the air    
        self.Opt_offset_x = self.MEMS_position_x + 2*np.tan(self.MEMS_offset_x)*self.air_depth#unit:m
        self.Opt_offset_y = self.MEMS_position_y + 2*np.tan(self.MEMS_offset_y)*self.air_depth
        self.Opt_grid_x = np.around(self.Opt_offset_x*(2/self.meshsize) + self.meshsize/2).astype(int)#the index of the meshgrid
        self.Opt_grid_y = np.around(self.Opt_offset_y*(2/self.meshsize) + self.meshsize/2).astype(int)
        
        #calc_wave
        self.Gx,self.Gy = np.meshgrid(np.linspace(1,self.meshsize,self.meshsize),np.linspace(1,self.meshsize,self.meshsize))
        self.grid_sign = np.ones((self.meshsize,self.meshsize))
        self.grid_sign[np.mod(self.Gx+self.Gy,2)==0] = -1
        self.wt = np.exp(1j*self.W*self.t0)
        self.Ht = self.H0 * self.wt + np.conjugate(np.rot90(self.H0,2))*np.conjugate(self.wt)
        self.Z = np.real(np.fft.ifft2(self.Ht)*self.grid_sign)
        self.Z = self.Z*10000
        
        #The slope at the position of the light spot                        
        self.slope_x = (self.Z[self.Opt_grid_x][self.Opt_grid_y+1]-self.Z[self.Opt_grid_x][self.Opt_grid_y])/(2/self.meshsize)
        self.slope_y = (self.Z[self.Opt_grid_x+1][self.Opt_grid_y]-self.Z[self.Opt_grid_x][self.Opt_grid_y])/(2/self.meshsize) 
        self.slope_x_temp = self.slope_x/40
        self.slope_y_temp = self.slope_y/40
        #The output direction after the air-water interface
        self.A1_x=np.arctan(self.slope_x_temp) - self.MEMS_offset_x*2
        self.Ai_x=(0.5*np.pi)-self.A1_x
        self.Ao_x=np.arcsin(np.sin(self.A1_x)*self.n1/self.n2)
        self.slope_X=np.arcsin(self.slope_x_temp) - self.Ao_x#self.Ao_x + self.Ai_x - np.pi/2;
    
        self.A1_y=np.arctan(self.slope_y_temp) - self.MEMS_offset_y*2#np.arctan(self.slope_y_temp) - (np.pi/2 - self.MEMS_offset_y*2);  
        self.Ai_y=(0.5*np.pi)-self.A1_y; 
        self.Ao_y=np.arcsin(np.sin(self.A1_y)*self.n1/self.n2)
        self.slope_Y=np.arctan(self.slope_y_temp) - self.Ao_y#self.Ao_y + self.Ai_y - np.pi/2;
  
        #Light reflected by cubic reflector
        self.Z_temp = self.Z[self.Opt_grid_x][self.Opt_grid_y]
    
        self.thrwave_a=self.slope_X*(self.sea_depth+self.Z_temp)
        self.thrwave_b=self.slope_Y*(self.sea_depth+self.Z_temp)
        self.Opt_position_x = self.Opt_offset_x + self.thrwave_a
        self.Opt_position_y = self.Opt_offset_y + self.thrwave_b
    
        #Offset in the POSITION SENSITIVE DETECTOR
        self.back_offset_x = 2*(self.Cubic_position_x - self.Opt_position_x)
        self.back_offset_y = 2*(self.Cubic_position_y - self.Opt_position_y)        
        self.PSD_offset = [self.back_offset_x,self.back_offset_y]
        
        self.reward = 1/(np.square(self.back_offset_x)+np.square(self.back_offset_y))
        if self.reward <= 20:
           self.reward = 0
           
        #self.reward = np.log10(self.reward)
           
        if self.reward >= 20000:
           self.reward = 20000   
           
        if self.MEMS_offset_x==self.MEMS_offset_x_min :
            self.MEMS_offset_x = 0
        if self.MEMS_offset_x==self.MEMS_offset_x_max :
            self.MEMS_offset_x = 0
        if self.MEMS_offset_y==self.MEMS_offset_y_min :
            self.MEMS_offset_y = 0
        if self.MEMS_offset_y==self.MEMS_offset_y_max :
            self.MEMS_offset_y = 0
            
        return self.back_offset_x, self.back_offset_y, self.reward
    
    def reset(self):
        #iniiial wave
        self.meshLim = np.pi * self.meshsize / self.patchsize
        self.N = np.linspace(-self.meshLim,self.meshLim,self.meshsize)
        self.M = np.linspace(-self.meshLim,self.meshLim,self.meshsize)
        self.Kx, self.Ky = np.meshgrid(self.N,self.M)
        self.K = np.sqrt(np.square(self.Kx)+np.square(self.Ky))
        self.W = np.sqrt(self.K*self.g)
        self.windx = np.cos(np.deg2rad(self.winddir))
        self.windy = np.sin(np.deg2rad(self.winddir))
        
        #phillips
        self.K_sq = np.square(self.Kx)+np.square(self.Ky)
        self.L = np.square(self.windSpeed)/self.g
        self.k_norm = np.sqrt(self.K_sq)
        self.WK = self.Kx/self.k_norm*self.windx+self.Ky/self.k_norm*self.windy
        self.P = self.A/np.square(self.K_sq)*np.exp(-1/(self.K_sq*np.square(self.L)))*np.square(self.WK)
        self.P[self.K_sq==0] = 0
        self.P[self.WK<0] = 0
        self.H0 = 1/np.sqrt(2)*(np.random.randn(self.meshsize,self.meshsize)+1j*np.random.randn(self.meshsize,self.meshsize))*np.sqrt(self.P)
        self.t0 = 0
        
        self.Gx,self.Gy = np.meshgrid(np.linspace(1,self.meshsize,self.meshsize),np.linspace(1,self.meshsize,self.meshsize))
        self.grid_sign = np.ones((self.meshsize,self.meshsize))
        self.grid_sign[np.mod(self.Gx+self.Gy,2)==0] = -1
        self.wt = np.exp(1j*self.W*self.t0)
        self.Ht = self.H0 * self.wt + np.conjugate(np.rot90(self.H0,2))*np.conjugate(self.wt)
        self.Z = np.real(np.fft.ifft2(self.Ht)*self.grid_sign)
        self.Z = self.Z*10000
        
        #Light propogation from air to water
        self.slope_x = np.ones((self.meshsize-1,self.meshsize-1))
        self.slope_y = np.ones((self.meshsize-1,self.meshsize-1))
        for i in range(0,self.meshsize-2):
            for k in range(0,self.meshsize-2):
                self.slope_x[i][k] = (self.Z[i][k+1]-self.Z[i][k])/(2/self.meshsize)/40
                self.slope_y[i][k] = (self.Z[i+1][k]-self.Z[i][k])/(2/self.meshsize)/40
                
        self.slope = np.square(self.slope_x)+np.square(self.slope_y)
        self.indx,self.indy = np.unravel_index(np.argmin(self.slope, axis=None), self.slope.shape)
        
        #Determine the position of cubic reflector and MEMS
        #The inital position has the minimum slope       unit:m
        self.Cubic_position_x =0 #(self.indx - self.meshsize/2)*(2/self.meshsize) #
        self.Cubic_position_y =0#(self.indy - self.meshsize/2)*(2/self.meshsize)  #
        self.MEMS_position_x = 0#(self.indx - self.meshsize/2)*(2/self.meshsize)  #
        self.MEMS_position_y = 0#(self.indy - self.meshsize/2)*(2/self.meshsize)  #
        
        #Determine the maximum angle can be adjusted by MEMS unit:rad
        self.MEMS_offset_x_min = -np.arctan((self.MEMS_position_x+1)/self.air_depth)/2
        self.MEMS_offset_x_max = np.arctan((1-self.MEMS_position_x)/self.air_depth)/2
        self.MEMS_offset_y_min = -np.arctan((self.MEMS_position_y+1)/self.air_depth)/2
        self.MEMS_offset_y_max = np.arctan((1-self.MEMS_position_y)/self.air_depth)/2
        
        
        
    def render(self):
        return None
        
    def close(self):
        return None

parser = argparse.ArgumentParser(description='PyTorch policy gradient example at openai-gym pong')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99')
parser.add_argument('--decay_rate', type=float, default=0.99, metavar='G',
                    help='decay rate for RMSprop (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=20, metavar='G',
                    help='Every how many episodes to da a param update')
parser.add_argument('--seed', type=int, default=87, metavar='N',
                    help='random seed (default: 87)')
args = parser.parse_args()

class Policy(nn.Module):
    def __init__(self, num_actions=9):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 200)
        self.affine2 = nn.Linear(200, num_actions) # action 1: static, action 2: move up, action 3: move down
        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []
        
       
       
      

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


    def select_action(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        probs = self.forward(x)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs.append(probs)
        return action

# built policy network
policy = Policy()

## check & load pretrain model
#if os.path.isfile('sypg_params2.pkl'):
#    print('Load Policy Network parametets ...')
#    policy.load_state_dict(torch.load('sypg_params2.pkl'))


# construct a optimal function  
optimizer = optim.RMSprop(policy.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(- log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    
    policy_loss.backward()
    optimizer.step()

    # clean rewards and saved_actions
    del policy.rewards[:]
    del policy.saved_log_probs[:]





#Training   
#while True:
env = Wave()
env.reset()
action = 0
OFFSET_X,OFFSET_Y,reward=env.step(action)
reward_save = []
running_reward_save = []
Offset_x = []
Offset_y = []
#OFFSET_X = 0.510734434132928
#OFFSET_Y = 0.2169636324219617
# Main loop
running_reward = None
reward_sum = 0
prev_x = None

while reward<20:
    action = 0
    OFFSET_X,OFFSET_Y,reward=env.step(action)

for i_episode in count(1):
    #env.reset()
    for t in range(10000):
        #if render: env.render()
        #cur_x = prepro(state)
        #x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        #prev_x = cur_x
        x = [OFFSET_X,OFFSET_Y]
        x = np.array(x)
        action = policy.select_action(x)
        #action = action.data.numpy()
        
        OFFSET_X,OFFSET_Y,reward= env.step(action)
        reward_sum += reward
        Offset_x.append(OFFSET_X)
        Offset_y.append(OFFSET_Y)

        policy.rewards.append(reward)
        if t==100:
            # tracking log
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('REINFORCE ep %03d done. reward: %f. reward running mean: %f' % (i_episode, reward_sum, running_reward))
            reward_save.append(reward_sum)
            running_reward_save.append(running_reward)

            reward_sum = 0
            break


    # use policy gradient update model weights
    if i_episode % args.batch_size == 0:
        finish_episode()

    # Save model in every 50 episode
    if i_episode % 10 == 0:
        print('ep %d: model saving...' % (i_episode))
        torch.save(policy.state_dict(), 'onpolicy.pkl')
        test = pd.DataFrame(reward_save)
        test.to_csv('reward_sum.csv')
        test = pd.DataFrame(running_reward_save)
        test.to_csv('runing_reward.csv')
        test = pd.DataFrame(Offset_x)
        test.to_csv('Offset_x.csv')
        test = pd.DataFrame(Offset_y)
        test.to_csv('Offset_y.csv')
        

