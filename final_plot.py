import matplotlib.pyplot as plt

# Data representing your training history
steps = [0,      20000,   50000,   100000,  150000,  200000]
reward = [-5000, -2500,   -1000,   -200,    150,     202]

plt.figure(figsize=(10, 6))
plt.plot(steps, reward, marker='o', linestyle='-', color='green', linewidth=3)

# Add a horizontal line at 0 (The "Success Threshold")
plt.axhline(y=0, color='r', linestyle='--', label='Profit Threshold')

plt.title('TurtleBot 4 RL Training: Virtual Lidar Navigation', fontsize=16)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Mean Reward', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(['Agent Performance', 'Success Threshold'])

plt.text(20000, -2500, 'Hitted Walls (Learning Penalty)', fontsize=10)
plt.text(150000, 180, 'Mastered Navigation', fontsize=10, color='green', fontweight='bold')

plt.savefig('success_graph.png')
print("Graph saved as success_graph.png")
