import gym
import slimevolleygym
import pickle
import numpy as np
import imageio
import collections  # Import collections to use deque

# --- Constants ---
GENOME_PATH = 'champion_genome.pkl'
FPS = 30
DURATION_MS = 1000 / FPS
FRAME_LIMIT = 1000  # Keep only the last 300 frames

def act_from_output(out):
    left = out[0] < -0.33
    right = out[0] > 0.33
    jump = out[1] > 0.5
    return np.array([left, right, jump], dtype=np.int8)

with open(GENOME_PATH, 'rb') as f:
    champion_genome = pickle.load(f)

env = gym.make('SlimeVolley-v0')
obs = env.reset()

# Use a deque to automatically store only the last 300 frames
frames = collections.deque(maxlen=FRAME_LIMIT)
my_score = 0
opponent_score = 0

print("\n--- Starting New Game (will stop at first point) ---")

while True:
    frame = env.render(mode='rgb_array')
    frames.append(frame)
    
    out = champion_genome.forward(np.asarray(obs, dtype=np.float32))
    action = act_from_output(out)
    
    obs, reward, done, info = env.step(action)
    
    if reward == 1:
        my_score += 1
        print(f"Agent scores! Score: {my_score} - {opponent_score}")
    elif reward == -1:
        opponent_score += 1
        print(f"Opponent scores. Score: {my_score} - {opponent_score}")

    # Stop as soon as the score is 1-0 or 0-1
    if my_score == 1 or opponent_score == 1:
        break

print(f"Game Over. Final Score: {my_score} - {opponent_score}")

if my_score == 1:
    print(f"Agent Wins! Saving last {len(frames)} frames as GIF...")
    imageio.mimsave('winning_clip.gif', frames, duration=DURATION_MS)
    print("GIF saved as winning_clip.gif")
else:
    print("Agent Lost. GIF will not be saved.")

env.close()