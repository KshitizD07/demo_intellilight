import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from rl.traffic_env import TrafficEnv

model = PPO.load("models/checkpoints/intellilight_final.zip")

env = TrafficEnv(use_gui=True)

obs, info = env.reset()

queue_history = []
throughput_history = []
wait_history = []

done = False
truncated = False

while not (done or truncated):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, info = env.step(action)

    queue = sum(info["queues"])
    wait = sum(info["wait_times"])
    throughput = info["throughput"]

    queue_history.append(queue)
    wait_history.append(wait)
    throughput_history.append(throughput)

env.close()

print("Simulation finished. Plotting results...")

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.plot(queue_history)
plt.title("Queue Length Over Time")
plt.xlabel("Step")
plt.ylabel("Vehicles Waiting")

plt.subplot(1,3,2)
plt.plot(wait_history)
plt.title("Total Waiting Time")
plt.xlabel("Step")
plt.ylabel("Seconds")

plt.subplot(1,3,3)
plt.plot(throughput_history)
plt.title("Throughput")
plt.xlabel("Step")
plt.ylabel("Vehicles Completed")

plt.show()