import connection as cn
import numpy as np
import random
import os

# Constants
PORT = 2037



ACTIONS = ['left', 'right', 'jump']
NUM_PLATFORMS = 24
NUM_DIRECTIONS = 4
NUM_STATES = NUM_PLATFORMS * NUM_DIRECTIONS
NUM_ACTIONS = len(ACTIONS)

# Q-learning parameters

DISCOUNT_FACTOR = 0.92
EPSILON = 0.1
LEARNING_RATE = 0.1
EPISODES = 200
EPISODES_CYCLE = 100000
MAX_STEPS = 50

def binary_to_state_index(binary_state):
    """Convert binary state string to state index (0-95)"""
    platform = int(binary_state[:7], 2)  # First 5 bits for platform (0-23)
    direction = int(binary_state[-2:], 2) 

    return platform * NUM_DIRECTIONS + direction

def get_platform(binary_state):
    return int(binary_state[:7], 2)

def get_direction(binary_state):
    return int(binary_state[-2:], 2)

# Initialize Q-table (96 states × 3 actions)
if os.path.exists(f'resultado.txt'):
    q_table = np.loadtxt(f'resultado.txt')
    print("Q-table carregada com sucesso!")
else:
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))
    print("Criando uma nova Q-table...")

# Connect to the game
s = cn.connect(PORT)
print("Connected to game server")

BEST_EPISODE = 0
BEST_REWARD = 0
BEST_STEPS = 0
GOAL_REACHED_COUNT = 0
BEST_REWARD_REACHED = 0
PLATFORM_REACHED = 0

# Training loop
count = 0
state=-1
next_state_idx = -1
steps_same_platform = 0
try:
    while count < EPISODES_CYCLE:
        count += 1
        EPSILON = 0.1
        print("Saving Q-table to resultado.txt")
        np.savetxt(f'resultado.txt', q_table, fmt='%.6f')
        for episode in range(EPISODES):
            print(f"\nEpisode {episode + 1 + ((count-1) * EPISODES)} cycle {count}")

            LEARNING_RATE = max(0.01, 0.1 * (0.99 ** episode))
            EPSILON = max(0.01, EPSILON - (0.1 / (EPISODES*10)))
            print(f"Learning rate: {LEARNING_RATE} e Epsilon: {EPSILON}")

            # Get initial state
            state, reward = cn.get_state_reward(s, 'jump')  # Initial action
            print(f"State: {state}, Reward: {reward}")

            done = False
            steps = 0
            total_reward = 0

            while not done and steps < MAX_STEPS:  # Limit steps per episode
                current_platform = get_platform(state)
                current_direction = get_direction(state)
                state_idx = binary_to_state_index(state)

                # Epsilon-greedy action selection
                if random.random() < EPSILON:
                    print("Action: Random")
                    action_idx = random.randint(0, NUM_ACTIONS - 1)
                else:
                    action_idx = np.argmax(q_table[state_idx])

                # Take action
                # while next_state_idx == state_idx:
                next_state, reward = cn.get_state_reward(s, ACTIONS[action_idx])
                next_state_idx = binary_to_state_index(next_state)
                next_platform = get_platform(next_state)
                next_direction = get_direction(next_state)
                
                while next_state_idx == state_idx:
                    next_state, reward = cn.get_state_reward(s, ACTIONS[action_idx])
                    next_state_idx = binary_to_state_index(next_state)
                    next_platform = get_platform(next_state)
                    next_direction = get_direction(next_state)

                if(next_platform == current_platform):
                    steps_same_platform += 1
                    reward -= 6 + (2 * steps_same_platform)
                elif(next_platform < current_platform or (next_platform == 13 and current_platform == 23) or (next_platform == 23 and current_platform == 13)):
                    reward -= 10
                elif( next_platform> current_platform ):
                    reward += 10
                    steps_same_platform = 0
            
                # Q-learning update
                old_value = q_table[state_idx, action_idx]
                next_max = np.max(q_table[next_state_idx])
                new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
                q_table[state_idx, action_idx] = new_value

                total_reward += reward
                state = next_state
                steps += 1

                print(f"Step {steps}: Action={ACTIONS[action_idx]}, State={state}, Reward={reward} platform={get_platform(state)}")

                # Check if episode is done
                if reward > 250:  # Reached goal
                    q_table[state_idx, action_idx] = reward
                    print("Goal reached!")
                    GOAL_REACHED_COUNT += 1
                    done = True
                elif reward < -60:  # Failed
                    print("Failed - resetting episode")
                    done = True

            
            print(f"Episode {episode + 1} finished after {steps} steps. Total reward: {total_reward}")
            episode += 1
            print(f"Goal reached count: {GOAL_REACHED_COUNT}")
            if (reward > BEST_REWARD_REACHED) or BEST_REWARD_REACHED == 0:
                BEST_EPISODE = episode + (EPISODES * (count-1))
                BEST_REWARD = total_reward 
                BEST_STEPS = steps
                BEST_REWARD_REACHED =reward
                PLATFORM_REACHED = get_platform(state)
                print(f"New best episode: {BEST_EPISODE}, Reward: {BEST_REWARD} Steps: {BEST_STEPS} Reward reached: {BEST_REWARD_REACHED} Platform reached: {PLATFORM_REACHED}")
            elif(episode % 100 == 0):
                print(f"Best episode: {BEST_EPISODE}, Reward: {BEST_REWARD} Steps: {BEST_STEPS} Reward reached: {BEST_REWARD_REACHED} Platform reached: {PLATFORM_REACHED}")

            while reward != -100:

                next_state, reward = cn.get_state_reward(s, 'jump')
                
                state = next_state
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
finally:
    # Save Q-table
    print("Saving Q-table to resultado.txt")
    np.savetxt(f'resultado.txt', q_table, fmt='%.6f')
    print(f"Best episode: {BEST_EPISODE}, Reward: {BEST_REWARD} Steps: {BEST_STEPS} Reward reached: {BEST_REWARD_REACHED} Platform reached: {PLATFORM_REACHED}")
    print(f"Goal reached count: {GOAL_REACHED_COUNT}")
    print(f"Platform reached: {PLATFORM_REACHED}")
    print("Training complete!")
