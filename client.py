import connection as cn
import numpy as np
import random
import os

# Constants
PORT = int(input("Porta padrão: 2037\nDigite número da porta: "))



ACTIONS = ['left', 'right', 'jump']
NUM_PLATFORMS = 24
NUM_DIRECTIONS = 4
NUM_STATES = NUM_PLATFORMS * NUM_DIRECTIONS
NUM_ACTIONS = len(ACTIONS)

# Q-learning parameters

DISCOUNT_FACTOR = 0.9222
EPSILON = 0.1
LEARNING_RATE = 0.1
EPISODES = 300
Episodes_Cycle = 1000
MAX_STEPS = 20

def binary_to_state_index(binary_state):
    """Convert binary state string to state index (0-95)"""
    platform = int(binary_state[:5], 2)  # First 5 bits for platform (0-23)
    direction = int(binary_state[5:], 2)  # Last 2 bits for direction (0-3)
    return platform * NUM_DIRECTIONS + direction

# Initialize Q-table (96 states × 3 actions)
if os.path.exists(f'resultado-{PORT}.txt'):
    q_table = np.loadtxt(f'resultado-{PORT}.txt')
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
try:
    while count < Episodes_Cycle:
        count += 1
        EPSILON = 0.1
        for episode in range(EPISODES):
            print(f"\nEpisode {episode + 1} cycle {count}")

            LEARNING_RATE = max(0.01, 0.1 * (0.99 ** episode))
            EPSILON = max(0.01, EPSILON - (0.1 / EPISODES))
            print(f"Learning rate: {LEARNING_RATE} e Epsilon: {EPSILON}")

            # Get initial state
            state, reward = cn.get_state_reward(s, 'jump')  # Initial action
            print(f"State: {state}, Reward: {reward}")

            done = False
            steps = 0
            total_reward = 0

            while not done and steps < MAX_STEPS:  # Limit steps per episode
                state_idx = binary_to_state_index(state)

                # Epsilon-greedy action selection
                if random.random() < EPSILON:
                    action_idx = random.randint(0, NUM_ACTIONS - 1)
                else:
                    action_idx = np.argmax(q_table[state_idx])

                # Take action
                next_state, reward = cn.get_state_reward(s, ACTIONS[action_idx])
                next_state_idx = binary_to_state_index(next_state)

                # Q-learning update
                old_value = q_table[state_idx, action_idx]
                next_max = np.max(q_table[next_state_idx])
                new_value = (1 - LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
                q_table[state_idx, action_idx] = new_value

                total_reward += reward
                state = next_state
                steps += 1

                print(f"Step {steps}: Action={ACTIONS[action_idx]}, State={state}, Reward={reward}")

                # Check if episode is done
                if reward == -1:  # Reached goal
                    q_table[state_idx, action_idx] = 100
                    print("Goal reached!")
                    GOAL_REACHED_COUNT += 1
                    done = True
                elif reward == -100:  # Failed
                    print("Failed - resetting episode")
                    done = True

            print(f"Episode {episode + 1} finished after {steps} steps. Total reward: {total_reward}")
            episode += 1
            print(f"Goal reached count: {GOAL_REACHED_COUNT}")
            if (reward >= BEST_REWARD_REACHED) or episode == 1:
                BEST_EPISODE = episode + EPISODES * count
                BEST_REWARD = total_reward 
                BEST_STEPS = steps
                BEST_REWARD_REACHED =reward
                PLATFORM_REACHED = state
                print(f"New best episode: {BEST_EPISODE}, Reward: {BEST_REWARD} Steps: {BEST_STEPS} Reward reached: {BEST_REWARD_REACHED} Platform reached: {PLATFORM_REACHED}")
            elif(episode % 100 == 0):
                print(f"Best episode: {BEST_EPISODE}, Reward: {BEST_REWARD} Steps: {BEST_STEPS} Reward reached: {BEST_REWARD_REACHED} Platform reached: {PLATFORM_REACHED}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
finally:
    # Save Q-table
    print("Saving Q-table to resultado.txt")
    np.savetxt(f'resultado-{PORT}.txt', q_table, fmt='%.6f')
    print(f"Best episode: {BEST_EPISODE}, Reward: {BEST_REWARD} Steps: {BEST_STEPS} Reward reached: {BEST_REWARD_REACHED}")
    print(f"Goal reached count: {GOAL_REACHED_COUNT}")
    print(f"Platform reached: {PLATFORM_REACHED}")
    print("Training complete!")
