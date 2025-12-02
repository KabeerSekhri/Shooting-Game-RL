# shooting_qlearning_option_b_fixed.py
"""
Grid-based Shooting Game — Option B (fixed)
- Compact discrete state consistent with Q-table
- Proper enemy removal on hit (no leftover state between episodes)
- Hit only the nearest alive enemy on the shot row
- Rendering ignores removed enemies
Author: ChatGPT (GPT-5 Thinking mini) — fixes applied
"""

import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import pickle
import argparse

# ---------- Environment ----------
class ShootingEnvB:
    def __init__(self,
                 width=10,
                 height=10,
                 max_steps=60,
                 ammo_capacity=5,
                 enemy_behavior="random",  # "bounce" or "random" or "static"
                 seed: int | None = None):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.ammo_capacity = ammo_capacity
        self.enemy_behavior = enemy_behavior

        # player x fixed at left
        self.player_x = 0

        # enemies
        self.max_num_enemies = 3
        self.enemies = []

        # actions
        self.action_space_n = 4  # up, down, shoot, noop

        # rendering helpers
        self.last_shot_row = None
        self.last_shot_active = False
        self.enemy_hit = False

        # seed then reset
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        # player starts vertically centered
        self.player_y = self.height // 2

        # enemy direction: +1 (down) or -1 (up); store as int
        self.enemy_dir = 1

        # spawn fresh enemies each episode
        self.num_enemies = self.max_num_enemies
        self.enemies = []
        for i in range(self.max_num_enemies):
            enemy_x = np.random.randint(self.width // 2, self.width)
            enemy_y = np.random.randint(0, self.height)
            self.enemies.append([enemy_x, enemy_y, False])  # [x, y, hit_flag]

        # ammo
        self.ammo = self.ammo_capacity
        # step counter
        self.steps = 0
        self.done = False

        # rendering flags
        self.last_shot_active = False
        self.enemy_hit = False

        return self._get_obs()

    def _get_obs(self):
        """
        Return compact observation compatible with state_to_index:
         (player_y, nearest_enemy_y_or_height, ammo, enemy_dir)
         - nearest_enemy_y_or_height: the Y of the nearest alive enemy to the right
           If no alive enemy exists, return self.height (special value).
        """
        # find nearest alive enemy (smallest x) among alive enemies
        alive = [e for e in self.enemies if not e[2] and e[0] >= 0 and e[1] >= 0]
        if len(alive) == 0:
            enemy_y = self.height  # special "no enemy" value
        else:
            # choose nearest enemy by x coordinate
            nearest = min(alive, key=lambda ee: ee[0])
            enemy_y = nearest[1]

        return (self.player_y, enemy_y, self.ammo, self.enemy_dir)

    def step(self, action):
        """
        action: 0=Up,1=Down,2=Shoot,3=No-op
        Order:
          1) execute player's action
          2) handle shooting/hit
          3) move enemy
          4) step count & terminal checks
        """
        if self.done:
            raise RuntimeError("Step called on terminated episode. Call reset().")

        reward = 0.0
        info = {}
        self.steps += 1
        self.enemy_hit = False
        self.last_shot_active = False

        # ---------- Player action ----------
        if action == 0:  # Up
            if self.player_y > 0:
                self.player_y -= 1
            reward -= 0.1  # small movement cost
        elif action == 1:  # Down
            if self.player_y < self.height - 1:
                self.player_y += 1
            reward -= 0.1
        elif action == 2:  # Shoot
            # Visual: mark shot row for render (bullet trail)
            self.last_shot_row = self.player_y
            self.last_shot_active = True

            if self.ammo > 0:
                self.ammo -= 1
                reward -= 1.0  # ammo usage cost

                # Hit detection: find alive enemies on the player's row to the right
                candidates = [e for e in self.enemies if (not e[2]) and e[1] == self.player_y and e[0] > self.player_x]
                if candidates:
                    # hit the nearest such enemy (smallest x)
                    target = min(candidates, key=lambda ee: ee[0])
                    # mark hit and remove from play
                    target[2] = True
                    target[0] = -1
                    target[1] = -1
                    self.num_enemies -= 1
                    reward += 20.0
                    self.enemy_hit = True
                    if self.num_enemies == 0:
                        self.done = True
                # else: shot misses (no extra reward)
            else:
                # shooting with no ammo — heavier penalty
                reward -= 10.0
        elif action == 3:  # No-op
            reward -= 0.05
        else:
            raise ValueError("Invalid action.")

        # ---------- Enemy movement ----------
        for e in self.enemies:
            if not self.done and not e[2]:
                if self.enemy_behavior == "random":
                    move = np.random.choice([-1, 0, 1])  # allow stay for smoother motion
                    e[1] = max(0, min(e[1] + move, self.height - 1))
                    if move > 0:
                        self.enemy_dir = 1
                    elif move < 0:
                        self.enemy_dir = -1
                elif self.enemy_behavior == "bounce":
                    # simplistic bounce: use enemy_dir to move all alive enemies vertically
                    # flip direction if any would go out of bounds
                    # (this is a simple shared-direction bounce)
                    next_ys = [e[1] + self.enemy_dir for e in self.enemies if not e[2]]
                    if any(ny < 0 or ny >= self.height for ny in next_ys):
                        self.enemy_dir *= -1
                    for ee in self.enemies:
                        if not ee[2]:
                            ee[1] = max(0, min(ee[1] + self.enemy_dir, self.height - 1))
                else:
                    # static: do nothing
                    pass

        # ---------- Terminal conditions ----------
        if self.steps >= self.max_steps or self.ammo == 0:
            self.done = True
            # penalty for remaining enemies
            for e in self.enemies:
                if not e[2]:
                    reward -= 10.0

        return self._get_obs(), reward, self.done, info

    def render(self, show_info=True):
        """ASCII rendering with bullet trail and hit effect."""
        grid = [[" ." for _ in range(self.width)] for _ in range(self.height)]

        # Place player
        grid[self.player_y][self.player_x] = " P"

        # Place enemy symbol (only for alive enemies with valid coords)
        for e in self.enemies:
            ex, ey, hit = e
            if ex >= 0 and ey >= 0:
                grid[ey][ex] = " E" if not hit else " *"

        # Optionally show bullet trail (show '|' on shot row to the right of player)
        if self.last_shot_active and self.last_shot_row is not None:
            r = self.last_shot_row
            for c in range(self.player_x + 1, self.width):
                # don't overwrite player or hit marker
                if grid[r][c] == " .":
                    grid[r][c] = " |"

        # Print the grid
        print("=" * (self.width * 3))
        for r in range(self.height):
            print("".join(grid[r]))
        print("=" * (self.width * 3))

        if show_info:
            print(f"Step: {self.steps}/{self.max_steps} | Player row: {self.player_y} | "
                  f"Enemy left: {self.num_enemies} (dir: {self.enemy_dir}) | Ammo: {self.ammo}")
            print()

# ---------- State <-> Index mapping ----------
def state_to_index(player_y, enemy_y, ammo, enemy_dir, height, ammo_capacity):
    """
    enemy_y in [0..height-1] or height == special 'no enemy' value
    enemy_dir: -1 -> 0, +1 -> 1
    shape: player_y (height) x enemy_y (height+1) x ammo (ammo_capacity+1) x dir (2)
    """
    dir_idx = 0 if enemy_dir <= 0 else 1
    enemy_range = height + 1
    return (((player_y * enemy_range + enemy_y) * (ammo_capacity + 1) + ammo) * 2 + dir_idx)

def index_to_state(index, height, ammo_capacity):
    dir_idx = index % 2
    index //= 2
    ammo = index % (ammo_capacity + 1)
    index //= (ammo_capacity + 1)
    enemy_range = height + 1
    enemy_y = index % enemy_range
    player_y = index // enemy_range
    enemy_dir = -1 if dir_idx == 0 else 1
    return player_y, enemy_y, ammo, enemy_dir

# ---------- Q-learning ----------
def train_q_learning(env,
                     num_episodes=10000,
                     alpha=0.1,
                     gamma=0.99,
                     epsilon_start=1.0,
                     epsilon_min=0.05,
                     epsilon_decay=0.995,
                     render_every=0,
                     verbose=True):
    height = env.height
    ammo_capacity = env.ammo_capacity
    enemy_range = height + 1
    n_states = height * enemy_range * (ammo_capacity + 1) * 2
    n_actions = env.action_space_n

    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    episode_rewards = []
    epsilons = []
    success_rates = []
    recent_success = deque(maxlen=100)

    epsilon = epsilon_start

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        state_idx = state_to_index(obs[0], obs[1], obs[2], obs[3], height, ammo_capacity)
        total_reward = 0.0
        done = False

        while not done:
            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                action = int(np.argmax(Q[state_idx]))

            next_obs, reward, done, _ = env.step(action)
            next_state_idx = state_to_index(next_obs[0], next_obs[1], next_obs[2], next_obs[3], height, ammo_capacity)

            # Q update (SARSA-less / Q-learning)
            best_next = np.max(Q[next_state_idx])
            Q[state_idx, action] += alpha * (reward + gamma * best_next - Q[state_idx, action])

            state_idx = next_state_idx
            total_reward += reward

        # decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_min)

        episode_rewards.append(total_reward)
        epsilons.append(epsilon)

        # success detection: we consider >10 reward as success heuristic
        success = total_reward > 10.0
        recent_success.append(1 if success else 0)
        success_rates.append(np.mean(recent_success))

        if verbose and (ep % max(1, num_episodes // 10) == 0 or ep <= 10):
            print(f"Episode {ep}/{num_episodes} | Reward: {total_reward:+6.2f} | "
                  f"Epsilon: {epsilon:.3f} | Recent success rate: {success_rates[-1]:.2f}")

        if render_every and (ep % render_every == 0):
            env.render()

    return Q, episode_rewards, epsilons, success_rates

# ---------- Evaluation ----------
def evaluate_policy(env, Q, episodes=200, render=False):
    height = env.height
    ammo_capacity = env.ammo_capacity
    successes = 0
    rewards = []
    for ep in range(episodes):
        obs = env.reset()
        state_idx = state_to_index(obs[0], obs[1], obs[2], obs[3], height, ammo_capacity)
        total_reward = 0.0
        done = False
        while not done:
            action = int(np.argmax(Q[state_idx]))
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            state_idx = state_to_index(next_obs[0], next_obs[1], next_obs[2], next_obs[3], height, ammo_capacity)
            if render:
                env.render()
        rewards.append(total_reward)
        if total_reward > 10.0:
            successes += 1
    return successes / episodes, np.mean(rewards), np.std(rewards)

# ---------- Utilities ----------
def plot_training(rewards, epsilons, success_rates, outdir="results_option_b"):
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
    fname = os.path.join(outdir, "training_plots_option_b.png")
    plt.savefig(fname)
    print(f"Saved training plots to {fname}")
    plt.close()

def save_qtable(Q, filename="q_table_option_b.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(Q, f)
    print(f"Saved Q-table to {filename}")

def load_qtable(filename="q_table_option_b.pkl"):
    with open(filename, "rb") as f:
        Q = pickle.load(f)
    print(f"Loaded Q-table from {filename}")
    return Q

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000, help="Training episodes")
    parser.add_argument("--width", type=int, default=10, help="Grid width")
    parser.add_argument("--height", type=int, default=10, help="Grid height")
    parser.add_argument("--max-steps", type=int, default=60, help="Max steps per episode")
    parser.add_argument("--ammo", type=int, default=5, help="Ammo capacity per episode")
    parser.add_argument("--enemy-behavior", type=str, default="bounce", choices=["bounce", "random", "static"], help="Enemy movement behavior")
    parser.add_argument("--save", type=str, default="q_table_option_b.pkl", help="File to save Q-table")
    parser.add_argument("--plot-dir", type=str, default="results_option_b", help="Directory to save plots")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    env = ShootingEnvB(width=args.width,
                       height=args.height,
                       max_steps=args.max_steps,
                       ammo_capacity=args.ammo,
                       enemy_behavior=args.enemy_behavior,
                       seed=args.seed)

    print("Training Q-learning agent on ShootingEnvB (Option B)")
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

    print("Evaluating policy (200 episodes)")
    success_rate, mean_reward, std_reward = evaluate_policy(env, Q, episodes=200, render=False)
    print(f"Eval success rate: {success_rate:.2f}, mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Example interactive episode (rendered)
    print("\nExample interactive episode (rendered):")
    obs = env.reset()
    done = False
    height = env.height
    ammo_capacity = env.ammo_capacity
    state_idx = state_to_index(obs[0], obs[1], obs[2], obs[3], height, ammo_capacity)
    steps = 0
    while not done and steps < env.max_steps:
        action = int(np.argmax(Q[state_idx]))
        obs, reward, done, _ = env.step(action)
        state_idx = state_to_index(obs[0], obs[1], obs[2], obs[3], height, ammo_capacity)
        env.render()
        steps += 1

if __name__ == "__main__":
    main()
