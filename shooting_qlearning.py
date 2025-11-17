# shooting_qlearning.py
"""
Grid-based Shooting Game (Option A) with Q-learning.

Environment:
- Grid: WIDTH x HEIGHT (default 10x10)
- Player at x=0, can move up/down, shoot, or do nothing.
- Single stationary enemy at x=WIDTH-1, random row each episode.
- State: (player_row, enemy_row)
- Actions: 0=Up, 1=Down, 2=Shoot, 3=No-op

Q-learning tabular agent trains on this environment.

Author: ChatGPT
"""

import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import pickle
import argparse

# ---------- Environment ----------
class ShootingEnv:
    def __init__(self, width=10, height=10, max_steps=50):
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Action space: Up, Down, Shoot, No-op
        self.action_space_n = 4

        # State: (player_row, enemy_row) - both in range [0, height-1]
        self.player_x = 0
        self.enemy_x = width - 1

        self.seed()
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        # Player starts vertically centered
        self.player_row = self.height // 2
        # Enemy spawns at rightmost column, random row
        self.enemy_row = np.random.randint(0, self.height)
        self.steps = 0
        self.done = False
        # For Option A, enemy does not move
        return self._get_obs()

    def _get_obs(self):
        # return a small tuple-like state
        return (self.player_row, self.enemy_row)

    def step(self, action):
        """
        action: 0=Up,1=Down,2=Shoot,3=No-op
        Returns: obs, reward, done, info
        """
        if self.done:
            raise RuntimeError("Step called on terminated episode. Call reset().")

        reward = 0.0
        info = {}
        self.steps += 1

        # Movement
        if action == 0:  # Up
            if self.player_row > 0:
                self.player_row -= 1
        elif action == 1:  # Down
            if self.player_row < self.height - 1:
                self.player_row += 1
        elif action == 2:  # Shoot
            # Bullet travels horizontally from player_x to enemy_x.
            # In Option A, enemy is at enemy_x and stationary; a hit occurs if rows match.
            if self.player_row == self.enemy_row:
                # Hit
                reward += 15.0  # hitting enemy
                reward += 30.0  # killing (for Option A, hit == kill)
                self.done = True
            else:
                # Missed shot
                reward -= 2.0
        elif action == 3:  # No-op
            pass
        else:
            raise ValueError("Invalid action.")

        # Small step penalty to encourage efficiency
        reward -= 0.1

        # End episode if max steps reached
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, info

    def render(self):
        """Simple ASCII rendering"""
        grid = [[" ." for _ in range(self.width)] for _ in range(self.height)]
        # Player
        grid[self.player_row][self.player_x] = " P"
        # Enemy
        grid[self.enemy_row][self.enemy_x] = " E"
        # Print grid with row indices
        print("=" * (self.width * 3))
        for r in range(self.height):
            print("".join(grid[r]))
        print("=" * (self.width * 3))

# ---------- Q-learning ----------
def state_to_index(player_row, enemy_row, height):
    return player_row * height + enemy_row

def index_to_state(index, height):
    player_row = index // height
    enemy_row = index % height
    return player_row, enemy_row

def train_q_learning(env,
                     num_episodes=2000,
                     alpha=0.1,
                     gamma=0.99,
                     epsilon_start=1.0,
                     epsilon_min=0.05,
                     epsilon_decay=0.999,
                     render_every=0,
                     verbose=True):
    n_states = env.height * env.height  # player_row x enemy_row
    n_actions = env.action_space_n

    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    episode_rewards = []
    epsilons = []
    success_rates = []
    recent_success = deque(maxlen=100)

    epsilon = epsilon_start

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        state_idx = state_to_index(obs[0], obs[1], env.height)
        total_reward = 0.0
        done = False

        while not done:
            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                action = int(np.argmax(Q[state_idx]))

            next_obs, reward, done, _ = env.step(action)
            next_state_idx = state_to_index(next_obs[0], next_obs[1], env.height)

            # Q update
            best_next = np.max(Q[next_state_idx])
            Q[state_idx, action] += alpha * (reward + gamma * best_next - Q[state_idx, action])

            state_idx = next_state_idx
            total_reward += reward

        # decays
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_min)

        episode_rewards.append(total_reward)
        epsilons.append(epsilon)

        # success if enemy killed (positive large reward), detect by checking if total_reward > threshold
        success = total_reward > 10.0  # heuristic: if Q got kill reward (+45 minus penalties)
        recent_success.append(1 if success else 0)
        success_rates.append(np.mean(recent_success))

        if verbose and (ep % max(1, num_episodes // 10) == 0 or ep <= 10):
            print(f"Episode {ep}/{num_episodes} | "
                  f"Reward: {total_reward:+6.2f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Recent success rate: {success_rates[-1]:.2f}")

        if render_every and (ep % render_every == 0):
            env.render()

    return Q, episode_rewards, epsilons, success_rates

# ---------- Evaluation ----------
def evaluate_policy(env, Q, episodes=100, render=False):
    n_actions = env.action_space_n
    successes = 0
    rewards = []
    for ep in range(episodes):
        obs = env.reset()
        state_idx = state_to_index(obs[0], obs[1], env.height)
        total_reward = 0.0
        done = False
        while not done:
            action = int(np.argmax(Q[state_idx]))
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            state_idx = state_to_index(next_obs[0], next_obs[1], env.height)
            if render:
                env.render()
        rewards.append(total_reward)
        if total_reward > 10.0:
            successes += 1
    return successes / episodes, np.mean(rewards), np.std(rewards)

# ---------- Utilities ----------
def plot_training(rewards, epsilons, success_rates, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.subplot(1, 3, 2)
    plt.plot(epsilons)
    plt.title("Epsilon")
    plt.xlabel("Episode")
    plt.subplot(1, 3, 3)
    plt.plot(success_rates)
    plt.title("Success Rate (rolling 100)")
    plt.xlabel("Episode")
    plt.tight_layout()
    fname = os.path.join(outdir, "training_plots.png")
    plt.savefig(fname)
    print(f"Saved training plots to {fname}")
    plt.close()

def save_qtable(Q, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(Q, f)
    print(f"Saved Q-table to {filename}")

def load_qtable(filename="q_table.pkl"):
    with open(filename, "rb") as f:
        Q = pickle.load(f)
    print(f"Loaded Q-table from {filename}")
    return Q

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000, help="Training episodes")
    parser.add_argument("--width", type=int, default=10, help="Grid width")
    parser.add_argument("--height", type=int, default=10, help="Grid height")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--save", type=str, default="q_table.pkl", help="File to save Q-table")
    parser.add_argument("--plot-dir", type=str, default="results", help="Directory to save plots")
    args = parser.parse_args()

    env = ShootingEnv(width=args.width, height=args.height, max_steps=args.max_steps)

    print("Training Q-learning agent on ShootingEnv")
    Q, rewards, epsilons, success_rates = train_q_learning(
        env,
        num_episodes=args.episodes,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        render_every=0,
        verbose=True
    )

    plot_training(rewards, epsilons, success_rates, outdir=args.plot_dir)
    save_qtable(Q, filename=args.save)

    print("Evaluating policy (100 episodes)")
    success_rate, mean_reward, std_reward = evaluate_policy(env, Q, episodes=100, render=False)
    print(f"Eval success rate: {success_rate:.2f}, mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Example: interactive play
    print("\nExample interactive episode (rendered):")
    obs = env.reset()
    done = False
    state_idx = state_to_index(obs[0], obs[1], env.height)
    steps = 0
    while not done and steps < 50:
        action = int(np.argmax(Q[state_idx]))
        obs, reward, done, _ = env.step(action)
        state_idx = state_to_index(obs[0], obs[1], env.height)
        env.render()
        steps += 1

if __name__ == "__main__":
    main()
