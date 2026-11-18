
import numpy as np
import matplotlib.pyplot as plt
import random

# 0 = free cell, 1 = wall
maze = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [1, 1, 0, 0]
])

start = (0, 0)
goal = (3, 3)

rows, cols = maze.shape

# Actions: up, right, down, left
actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
arrows = ['↑', '→', '↓', '←']

# Q-table: states x actions
Q = np.zeros((rows * cols, 4))

alpha = 0.1
gamma = 0.9
epsilon = 1.0

episodes = 1000
rewards = []

# Convert position into state number
def get_state(pos):
    return pos[0] * cols + pos[1]

# Perform an action in the maze
def move(pos, action):
    r = pos[0] + actions[action][0]
    c = pos[1] + actions[action][1]

    # Wall or boundary penalty
    if r < 0 or r >= rows or c < 0 or c >= cols or maze[r, c] == 1:
        return pos, -5, False

    new_pos = (r, c)

    # Goal reward
    if new_pos == goal:
        return new_pos, 50, True

    return new_pos, -1, False

# Training
for ep in range(episodes):
    pos = start
    total_reward = 0

    for step in range(100):
        state = get_state(pos)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])

        new_pos, reward, done = move(pos, action)
        next_state = get_state(new_pos)

        # Q-learning update
        target = reward if done else reward + gamma * np.max(Q[next_state])
        Q[state, action] += alpha * (target - Q[state, action])

        pos = new_pos
        total_reward += reward

        if done:
            break

    rewards.append(total_reward)

    # Reduce exploration gradually
    epsilon = max(0.05, epsilon * 0.995)

# Plot episode rewards
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q-learning Training")
plt.show()

# Display learned policy
print("Learned Maze Policy:")

for r in range(rows):
    for c in range(cols):
        if maze[r, c] == 1:
            print("#", end=" ")
        elif (r, c) == goal:
            print("G", end=" ")
        else:
            state = get_state((r, c))
            print(arrows[np.argmax(Q[state])], end=" ")
    print()
