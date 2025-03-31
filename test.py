import connection as cn
import numpy as np
import random
import os

PORT = 2037
s = cn.connect(PORT)
print("Connected to game server")

state, reward = cn.get_state_reward(s, 'jump')
platform = int(state[:7], 2) 
direction = int(state[2:], 2)
print("state[-7:]: ", state[:7])
print("state[2:]: ", state[2:])
print(f"State: {state}, Reward: {reward} platform: {platform} direction: {direction}")


state, reward = cn.get_state_reward(s, 'left')
platform = int(state[:7], 2) 
direction = int(state[2:], 2)
print("state[-7:]: ", state[:7])
print("state[2:]: ", state[-2:])
print(f"State: {state}, Reward: {reward} platform: {platform} direction: {direction}")

state, reward = cn.get_state_reward(s, 'jump')
platform = int(state[:7], 2)    
direction = int(state[2:], 2)
print("state[-7:]: ", state[:7])
print("state[2:]: ", state[-2:]) 
print(f"State: {state}, Reward: {reward} platform: {platform} direction: {direction}")

state, reward = cn.get_state_reward(s, 'jump')
platform = int(state[:7], 2) 
direction = int(state[2:], 2)
print("state[-7:]: ", state[:7])
print("state[2:]: ", state[-2:])
print(f"State: {state}, Reward: {reward} platform: {platform} direction: {direction}")

