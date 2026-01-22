import matplotlib.pyplot as plt

# Data taken from your screenshots
# X axis: Total Timesteps
steps = [8192, 10240, 98304, 100352]

# Y axis: Mean Episode Reward (The "Score")
# -2000 is failing, 0 is perfect.
rewards = [-1770, -1970, -1280, -1280]

plt.figure(figsize=(10, 6))
plt.plot(steps, rewards, marker='o', linestyle='-', color='b', linewidth=2, label='PPO Training Performance')

# Add labels and title
plt.title('TurtleBot 4 Reinforcement Learning: Navigation Task', fontsize=16)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Mean Reward (Distance Penalty)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Add a text annotation showing improvement
plt.annotate('Initial Random Policy', xy=(8192, -1770), xytext=(20000, -1600),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('Learned Policy (35% Improvement)', xy=(100352, -1280), xytext=(60000, -1100),
             arrowprops=dict(facecolor='green', shrink=0.05))

# Save the plot
plt.savefig('learning_curve.png')
print("Graph saved as learning_curve.png")
plt.show()
