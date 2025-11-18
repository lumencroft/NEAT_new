import gym
import imageio
import numpy as np

RECORD_SECONDS = 3
FPS = 30
TOTAL_FRAMES_TO_CAPTURE = RECORD_SECONDS * FPS
DURATION_MS = 1000 / FPS  # Calculate duration in milliseconds

env = gym.make('CartPole-v1')
obs = env.reset()

frames = []
print(f"Recording for {RECORD_SECONDS} seconds ({TOTAL_FRAMES_TO_CAPTURE} frames)...")

for _ in range(TOTAL_FRAMES_TO_CAPTURE):
    frame = env.render(mode='rgb_array')
    frames.append(frame)
    
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()

env.close()

print(f"\nSaving {len(frames)} frames as test_record.gif...")

# Use duration (in ms) instead of fps
imageio.mimsave('test_record.gif', frames, duration=DURATION_MS)

print("Test GIF saved successfully.")