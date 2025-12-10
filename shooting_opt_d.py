# shooting_qlearning_option_d_fixed.py
"""
Grid-based Shooting Game — Option B (fixed)
- Compact discrete state consistent with Q-table
- Proper enemy removal on hit (no leftover state between episodes)
- Hit only the nearest alive enemy on the shot row
- Rendering ignores removed enemies
"""

import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import pickle
import argparse
import pygame

# ---------- Environment ----------
class ShootingEnvB:
    def __init__(self,
                 width=10,
                 height=10,
                 max_steps=60,
                 ammo_capacity=5,
                 enemy_behavior="random",  # "random" or "static"
                 seed: int | None = None):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.ammo_capacity = ammo_capacity
        self.enemy_behavior = enemy_behavior

        # player x fixed at left
        self.player_x = 0
        self.player_dead = False

        # enemies
        self.max_num_enemies = 3
        self.enemies = []

        # actions
        self.action_space_n = 4  # up, down, shoot, noop

        self.enemy_shots = []   # list of (x, y)
        self.enemy_shot_speed = 1
        self.enemy_shot_chance = 0.95  # enemy fires 5% of turns
        
        # rendering helpers
        self.last_shot_row = None
        self.last_shot_active = False
        self.enemy_hit = False

        # obstacles
        self.obstacles = set()
        for i in range(random.randint(1, 4)):
            y = random.randint(0, self.height-1)
            x = random.randint(2, self.width-2)
            self.obstacles.add((x, y))
            
        # levels
        self.level = 1
        self.max_level = 5
        self.level_rewards = {1: 100, 2: 200, 3: 400, 4: 800, 5: 1600}
        self.bonus_for_finishing_all = 2000

        # seed then reset
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def apply_level_settings(self):
        # Default resets
        self.enemy_shots = []

        if self.level == 1:
            self.enemy_behavior = "static"
            self.enemy_shot_chance = 0.0
            self.enemy_shot_speed = 1
            self.max_num_enemies = 3
            self.ammo_capacity = 30
            self.obstacles_enabled = False

        elif self.level == 2:
            self.enemy_behavior = "random"   # moving
            self.enemy_shot_chance = 0.0
            self.enemy_shot_speed = 1
            self.max_num_enemies = 4
            self.ammo_capacity = 25
            self.obstacles_enabled = False

        elif self.level == 3:
            self.enemy_behavior = "random"
            self.enemy_shot_chance = 0.05    # enemies shoot
            self.enemy_shot_speed = 2
            self.max_num_enemies = 5
            self.ammo_capacity = 20
            self.obstacles_enabled = False

        elif self.level == 4:
            self.enemy_behavior = "random"
            self.enemy_shot_chance = 0.07
            self.enemy_shot_speed = 2
            self.max_num_enemies = 6
            self.ammo_capacity = 15
            self.obstacles_enabled = True    # obstacles appear

        elif self.level == 5:
            self.enemy_behavior = "random"    # FULL AI PACKAGE
            self.enemy_shot_chance = 0.10
            self.enemy_shot_speed = 3
            self.max_num_enemies = 7
            self.ammo_capacity = 10
            self.obstacles_enabled = True



    def reset(self, level=1):
        self.level = level
        self.apply_level_settings()

        self.player_y = self.height // 2
        self.player_dead = False
        self.enemy_dir = 1

        self.enemy_shots = []  # important
        self.enemies = []

        self.num_enemies = self.max_num_enemies
        for _ in range(self.max_num_enemies):
            ex = np.random.randint(self.width // 2, self.width)
            ey = np.random.randint(0, self.height)
            self.enemies.append([ex, ey, False])

        self.ammo = self.ammo_capacity
        self.steps = 0
        self.done = False

        # rendering flags
        self.last_shot_active = False
        self.enemy_hit = False

        return self._get_obs()


    def _get_obs(self):

        # find nearest alive enemy (smallest x) among alive enemies
        alive = [e for e in self.enemies if not e[2] and e[0] >= 0 and e[1] >= 0]
        if len(alive) == 0:
            enemy_y = self.height  # special "no enemy" value
        else:
            # choose nearest enemy by x coordinate
            nearest = min(alive, key=lambda ee: ee[0])
            enemy_y = nearest[1]

        return (self.player_y, enemy_y, self.ammo, self.enemy_dir)

    def is_obstacle(self, x, y):
        return (x, y) in self.obstacles

    # Environment helpers
    def get_compact_state(self):
        """
        Compute a compact state representation from the environment internals.
        Returns a tuple of integer features in a stable order:
        (level_idx,
        player_row,
        ammo_bucket,
        nearest_enemy_dx_bucket,
        nearest_enemy_dy_bucket,
        enemy_in_sight_flag,
        bullet_threat_flag,
        obstacle_ahead_flag)
        """

        # 1) level (0..4)
        level_idx = max(1, getattr(self, "level", 1)) - 1

        # 2) player_row (0..height-1)
        player_row = int(self.player_y)

        # 3) ammo bucket (0..5) - cap for tabular Q
        # buckets: 0,1,2,3,4,5+ -> 6 buckets
        ammo = int(getattr(self, "ammo", 0))
        ammo_bucket = ammo if ammo <= 5 else 5

        # 4) nearest enemy dx bucket (0..4)
        # dx = min positive (enemy_x - player_x) where enemy is alive and to the right
        player_x = int(getattr(self, "player_x", 0))
        nearest_dx = None
        nearest_dy = None
        for e in self.enemies:
            ex, ey, hit = e
            if hit:
                continue
            dx = ex - player_x
            if dx <= 0:
                continue
            if nearest_dx is None or dx < nearest_dx:
                nearest_dx = dx
                nearest_dy = int(ey - self.player_y)

        # dx buckets: 0 = adjacent (dx==1), 1 = dx 2-3, 2 = dx 4-6, 3 = dx 7+, 4 = no enemy
        if nearest_dx is None:
            dx_bucket = 4
            dy_bucket = 3  # "no enemy" code
        else:
            if nearest_dx == 1:
                dx_bucket = 0
            elif nearest_dx <= 3:
                dx_bucket = 1
            elif nearest_dx <= 6:
                dx_bucket = 2
            else:
                dx_bucket = 3

            # dy bucket: -inf.. -> map to {-1,0,1,2} : above, same, below, unknown
            if nearest_dy < 0:
                dy_bucket = 0
            elif nearest_dy == 0:
                dy_bucket = 1
            else:
                dy_bucket = 2

        # 5) enemy_in_sight_flag: any enemy in same row with no obstacle between
        enemy_in_sight = 0
        for e in self.enemies:
            ex, ey, hit = e
            if hit:
                continue
            if ey == self.player_y and ex > player_x:
                # check obstacles in between
                blocked = any(self.is_obstacle(x, self.player_y)
                            for x in range(player_x + 1, ex))
                if not blocked:
                    enemy_in_sight = 1
                    break

        # 6) bullet_threat_flag: any enemy_shot on player's row and x >= player_x
        bullet_threat = 0
        for sx, sy in getattr(self, "enemy_shots", []):
            if sy == self.player_y and sx >= player_x:
                bullet_threat = 1
                break

        # 7) obstacle_ahead_flag: obstacle immediately to player's right (useful for movement)
        obstacle_ahead = 0
        if player_x + 1 < getattr(self, "width", 0):
            if self.is_obstacle(player_x + 1, self.player_y):
                obstacle_ahead = 1

        return (level_idx, player_row, ammo_bucket, dx_bucket, dy_bucket,
                enemy_in_sight, bullet_threat, obstacle_ahead)

    # ---------- State <-> Index mapping ---------
    def compact_state_sizes(self):
        """
        Return the dimension sizes for each compact feature in the same order as get_compact_state().
        Used to compute mixed-radix index (state -> integer).
        """
        # level: 5 (levels 1..5) -> indexes 0..4
        level_bins = getattr(self, "max_level", 5)
        player_row_bins = self.height  # 0..height-1
        ammo_bins = 6  # 0..5
        dx_bins = 5    # 0..4
        dy_bins = 4    # 0..3 (0 above,1 same,2 below,3 none)
        in_sight_bins = 2
        bullet_threat_bins = 2
        obstacle_bins = 2

        return (level_bins, player_row_bins, ammo_bins, dx_bins, dy_bins,
                in_sight_bins, bullet_threat_bins, obstacle_bins)


    def compact_state_to_index(self, compact_state):
        """
        Convert compact_state tuple to a single integer index using mixed-radix.
        """
        sizes = self.compact_state_sizes()
        idx = 0
        multiplier = 1
        for val, base in zip(reversed(compact_state), reversed(sizes)):
            if val < 0 or val >= base:
                print("compact_state:", compact_state)
                print("sizes:", sizes)
                raise ValueError(f"compact state value {val} out of range for base {base}")
            idx += val * multiplier
            multiplier *= base

        return int(idx)


    def index_to_compact_state(self, idx):
        """
        Convert index -> compact_state tuple (inverse mapping).
        """
        sizes = self.compact_state_sizes()
        vals = []
        for base in reversed(sizes):
            vals.append(int(idx % base))
            idx //= base
        vals = list(reversed(vals))
        return tuple(vals)


    def step(self, action):
        """
        action: 0=Up,1=Down,2=Shoot,3=No-op
        Order:
          1) execute player's action
          2) handle shooting/hit
          3) move enemy
          4) enemy shooting
          5) step count & terminal checks
        """
        if self.done:
            raise RuntimeError("Step called on terminated episode. Call reset().")

        reward = 0.0
        info = {}
        self.steps += 1
        self.last_shot_active = False
        self.player_dead = False
        self.enemy_hit = False

        # ---------- Player action ----------
        if action == 0:  # Up
            if self.player_y > 0 and not self.is_obstacle(self.player_x, self.player_y - 1):
                self.player_y -= 1
            reward -= 0.1  # small movement cost
        elif action == 1:  # Down
            if self.player_y < self.height - 1 and not self.is_obstacle(self.player_x, self.player_y + 1):
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
                    
                    # check obstacles in between
                    blocked = any(self.is_obstacle(x, self.player_y) 
                                for x in range(self.player_x + 1, target[0]))
                    
                    if blocked:
                        target = None
                    elif target:
                        # mark hit and remove from play
                        target[2] = True
                        target[0] = -1
                        target[1] = -1
                        self.num_enemies -= 1
                        reward += 20.0
                        self.enemy_hit = True
                        if self.num_enemies == 0:
                            self.done = True
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
                    new_y = max(0, min(e[1] + move, self.height - 1))
                    if not self.is_obstacle(e[0], new_y):
                        e[1] = new_y
                    if move > 0:
                        self.enemy_dir = 1
                    elif move < 0:
                        self.enemy_dir = -1
                elif self.enemy_behavior == "static":
                    # static: do nothing
                    pass
                elif self.enemy_behavior == "smart":
                    # (optional) place your smart logic here
                    pass

        # ---------- Enemy shooting ----------
        alive_enemies = [e for e in self.enemies if not e[2]]
        if alive_enemies and random.random() < (1 - self.enemy_shot_chance):
            shooter = random.choice(alive_enemies)
            sx, sy, _ = shooter
            # spawn a single left-going bullet from the shooter (only once)
            self.enemy_shots.append([sx - 1, sy])

        # Advance enemy bullets and handle collisions with obstacles or player
        new_shots = []
        for sx, sy in self.enemy_shots:
            new_x = sx - self.enemy_shot_speed
            # bullet hits obstacle → disappears
            if new_x >= 0 and self.is_obstacle(new_x, sy):
                continue
            if new_x >= 0:
                new_shots.append([new_x, sy])
        self.enemy_shots = new_shots

        # check bullet hits player
        for sx, sy in list(self.enemy_shots):
            if sx == self.player_x and sy == self.player_y:
                reward -= 20
                self.player_dead = True
                # optionally remove the bullet after hit:
                try:
                    self.enemy_shots.remove([sx, sy])
                except ValueError:
                    pass



        # ---------- Terminal conditions ----------
        if self.steps >= self.max_steps or self.ammo == 0 or self.player_dead==True:
            self.done = True
            # penalty for remaining enemies
            for e in self.enemies:
                if not e[2]:
                    reward -= 10.0

        # compute whether level was cleared: no alive enemies and player alive
        level_cleared = (all(e[2] for e in self.enemies) and not self.player_dead)

        # award or penalize depending on outcome
        info = {}
        info["level_cleared"] = bool(level_cleared)

        if level_cleared:
            # award level completion reward immediately (and final bonus if last level)
            # make sure level_rewards / bonus exist on the env
            reward += getattr(self, "level_rewards", {}).get(self.level, 0)
            if getattr(self, "max_level", None) is not None and self.level == self.max_level:
                reward += getattr(self, "bonus_for_finishing_all", 0)
            # mark done true if you want episode to end on clear (recommended)
            self.done = True
        else:
            # only apply 'remaining enemies' penalty if the episode ended but level not cleared
            if self.done and not level_cleared:
                for e in self.enemies:
                    if not e[2]:
                        reward -= 10.0

        # include remaining useful info
        info["player_dead"] = bool(self.player_dead)
        info["ammo"] = int(self.ammo)
        info["remaining_enemies"] = int(sum(1 for e in self.enemies if not e[2]))

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

# ---------- Rendering UI with Pygame ----------
class ShootingUI:
    def __init__(self, env, cell_size=40):
        self.env = env
        self.cell_size = cell_size

        pygame.init()
        self.screen = pygame.display.set_mode(
            (env.width * cell_size, env.height * cell_size)
        )
        pygame.display.set_caption("Shooting Game UI")
        self.clock = pygame.time.Clock()

        # ---------------------------
        # Load your sprites here
        # ---------------------------
        self.player_sprite = pygame.transform.scale(
            pygame.image.load("assets/player.png"),
            (cell_size, cell_size)
        )

        self.enemy_sprite = pygame.transform.scale(
            pygame.image.load("assets/enemy.png"),
            (cell_size, cell_size)
        )

        self.enemy_hit_sprite = pygame.transform.scale(
            pygame.image.load("assets/enemy_hit.png"),
            (cell_size, cell_size)
        )

        self.bullet_sprite = pygame.transform.scale(
            pygame.image.load("assets/bullet.png"),
            (cell_size, cell_size // 2)
        )

        self.enemy_bullet_sprite = pygame.transform.scale(
            pygame.image.load("assets/enemy_bullet.png"),
            (cell_size, cell_size // 2)
        )
        
        self.obstacle_sprite = pygame.transform.scale(
            pygame.image.load("assets/obstacle.png"),
            (cell_size, cell_size // 2)
        )

    def draw_grid(self):
        for y in range(self.env.height):
            for x in range(self.env.width):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, (40, 40, 40), rect, 1)

    def draw_player(self):
        x = self.env.player_x
        y = self.env.player_y
        self.screen.blit(
            self.player_sprite,
            (x * self.cell_size, y * self.cell_size)
        )


    def draw_enemies(self):
        for ex, ey, hit in self.env.enemies:
            if ex < 0:
                continue

            sprite = self.enemy_hit_sprite if hit else self.enemy_sprite

            self.screen.blit(
                sprite,
                (ex * self.cell_size, ey * self.cell_size)
            )


    def draw_shot(self):
        if not self.env.last_shot_active:
            return

        y = self.env.last_shot_row
        for x in range(self.env.player_x + 1, self.env.width):
            self.screen.blit(
                self.bullet_sprite,
                (x * self.cell_size, y * self.cell_size + self.cell_size//4)
            )

    def draw_enemy_shots(self):
        for sx, sy in self.env.enemy_shots:
            sprite = self.enemy_bullet_sprite   # load like others
            self.screen.blit(sprite, (sx*self.cell_size, sy*self.cell_size))

    def draw_obstacles(self):
        for (ox, oy) in self.env.obstacles:
            self.screen.blit(self.obstacle_sprite,
                            (ox*self.cell_size, oy*self.cell_size))
    

    def render_ui(self, fps=10):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        self.screen.fill((0, 0, 0))

        self.draw_grid()
        self.draw_player()
        self.draw_enemies()
        self.draw_shot()
        self.draw_enemy_shots()
        self.draw_obstacles()

        pygame.display.flip()
        self.clock.tick(fps)


# ---------- Q-learning ----------
def compute_n_states_from_env(env):
    sizes = env.compact_state_sizes()
    n = 1
    for s in sizes:
        n *= s
    return int(n)
def train_q_learning_levels(env,
                            max_episodes_per_level=3000,
                            alpha=0.1,
                            gamma=0.99,
                            epsilon_start=0.6,
                            epsilon_min=0.02,
                            epsilon_decay=0.9992,
                            success_window=200,
                            success_threshold=0.75,
                            verbose=True):
    """
    Q-learning using compact state encoding from env.get_compact_state().
    Curriculum per level: the trainer will move to next level when success rate
    over success_window reaches success_threshold, or when max_episodes_per_level reached.
    """
    # Prepare Q table
    n_states = compute_n_states_from_env(env)
    n_actions = env.action_space_n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    rewards_all = []
    epsilons = []
    success_rates = []

    for level in range(1, env.max_level + 1):
        if verbose:
            print("\n" + "="*60)
            print(f"       TRAINING LEVEL {level}")
            print("="*60)

        # per-level epsilon reset (fresh exploration)
        epsilon = epsilon_start
        recent_success = deque(maxlen=success_window)

        ep = 0
        while True:
            ep += 1
            obs = env.reset(level=level)

            # use compact state from env
            compact = env.get_compact_state()
            s_idx = env.compact_state_to_index(compact)

            done = False
            total_reward = 0.0

            while not done:
                # epsilon-greedy
                if np.random.rand() < epsilon:
                    a = np.random.randint(n_actions)
                else:
                    a = int(np.argmax(Q[s_idx]))

                next_obs, reward, done, info = env.step(a)

                next_compact = env.get_compact_state()
                ns_idx = env.compact_state_to_index(next_compact)

                # Q update
                Q[s_idx, a] += alpha * (reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, a])

                s_idx = ns_idx
                total_reward += reward

            # episode finished
            success = bool(info.get("level_cleared", False))
            recent_success.append(1 if success else 0)
            rewards_all.append(total_reward)
            epsilons.append(epsilon)
            success_rates.append(np.mean(recent_success))

            # decay epsilon inside level
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if verbose and ep % 200 == 0:
                print(f"[L{level}] ep {ep} | recent_success {np.mean(recent_success):.3f} | eps {epsilon:.3f} | reward {total_reward:+5.1f}")

            # curriculum condition
            if len(recent_success) == success_window and np.mean(recent_success) >= success_threshold:
                print(f"Level {level} solved (window avg {np.mean(recent_success):.3f}). Moving to next level.")
                break

            if ep >= max_episodes_per_level:
                print(f"Max episodes reached for level {level} ({ep}). Moving on.")
                break

        # end level
        print(f"Level {level} final success rate -> {np.mean(recent_success):.3f}")

    print("\nTraining complete.")
    return Q, rewards_all, epsilons, success_rates



# ---------- Evaluation ----------
def evaluate_policy(env, Q, episodes=200, render=False):
    successes = 0
    rewards = []
    
    for ep in range(episodes):
        obs = env.reset()
        # convert raw obs to compact state
        compact_obs = env.get_compact_state()
        state_idx = env.compact_state_to_index(compact_obs)
        total_reward = 0.0
        done = False

        while not done:
            action = int(np.argmax(Q[state_idx]))
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            # convert next_obs to compact state
            compact_next_obs = env.get_compact_state()
            state_idx = env.compact_state_to_index(compact_next_obs)

            if render:
                env.render()

        rewards.append(total_reward)
        if total_reward > 10.0:
            successes += 1

    return successes / episodes, np.mean(rewards), np.std(rewards)


# ---------- Utilities ----------
def plot_training(rewards, epsilons, success_rates, outdir="results_d"):
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
    fname = os.path.join(outdir, "training_plots_option_d.png")
    plt.savefig(fname)
    print(f"Saved training plots to {fname}")
    plt.close()

def save_qtable(Q, filename="q_table_option_d.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(Q, f)
    print(f"Saved Q-table to {filename}")

def load_qtable(filename="q_table_option_d.pkl"):
    with open(filename, "rb") as f:
        Q = pickle.load(f)
    print(f"Loaded Q-table from {filename}")
    return Q

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes")
    parser.add_argument("--width", type=int, default=10, help="Grid width")
    parser.add_argument("--height", type=int, default=10, help="Grid height")
    parser.add_argument("--max-steps", type=int, default=60, help="Max steps per episode")
    parser.add_argument("--ammo", type=int, default=5, help="Ammo capacity per episode")
    parser.add_argument("--enemy-behavior", type=str, default="bounce", choices=["bounce", "random", "static"], help="Enemy movement behavior")
    parser.add_argument("--save", type=str, default="q_table_option_d.pkl", help="File to save Q-table")
    parser.add_argument("--plot-dir", type=str, default="results_option_d", help="Directory to save plots")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    env = ShootingEnvB(width=args.width,
                       height=args.height,
                       max_steps=args.max_steps,
                       ammo_capacity=args.ammo,
                       enemy_behavior=args.enemy_behavior,
                       seed=args.seed)

    print("Training Q-learning agent on ShootingEnvB (Option D)")
    Q, rewards, epsilons, success_rates = train_q_learning_levels(
        env,
        max_episodes_per_level=args.episodes,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        verbose=True
    )

    plot_training(rewards, epsilons, success_rates, outdir=args.plot_dir)
    save_qtable(Q, filename=args.save)

    print("Evaluating policy (200 episodes)")
    success_rate, mean_reward, std_reward = evaluate_policy(env, Q, episodes=200, render=False)
    print(f"Eval success rate: {success_rate:.2f}, mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    print("Q shape:", Q.shape)

    # Example interactive episode (rendered)
    print("\nExample interactive episode (rendered):")
    obs = env.reset()
    done = False
    compact_obs = env.get_compact_state()
    state_idx =env.compact_state_to_index(compact_obs)
    steps = 0
    while not done and steps < env.max_steps:
        action = int(np.argmax(Q[state_idx]))
        obs, reward, done, _ = env.step(action)
        compact_obs = env.get_compact_state()
        state_idx = env.compact_state_to_index(compact_obs)
        steps += 1
    env.render()

if __name__ == "__main__":
    main()
