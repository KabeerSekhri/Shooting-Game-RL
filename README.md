Here’s a **brief, clean, student-friendly README** for **Option A (Basic Vertical Shooter RL Environment)**.
You can directly use this in your project repo.

---

# **Vertical Shooter RL Environment — Basic Version (Option A)**

### **Overview**

This project implements a simple custom **Reinforcement Learning environment** inspired by classic vertical-shooter arcade games.
The agent controls a player that can **move up/down** on the left side of the grid and **shoot** at an enemy placed on the right side.

The goal is to train an RL agent (e.g., Q-Learning or DQN) to **hit the enemy efficiently** using minimal movement and ammo.

---

## **Features**

* 2D grid world (configurable size, e.g., 10×10).
* Player can:

  * Move **up**
  * Move **down**
  * **Shoot** toward the right
  * **Do nothing**
* One stationary enemy placed at a fixed column.
* Simple ASCII-based **render**:

  * `P` = Player
  * `E` = Enemy
  * `e` or `̶E̶` (strikethrough) = Enemy hit moment
  * `.` = Empty cell

---

## **Rewards**

You can tune these based on desired agent behavior:

| Event                         | Reward       |
| ----------------------------- | ------------ |
| Hitting the enemy             | **+10**      |
| Shooting but missing          | **–1**       |
| Moving (up/down)              | **–0.1**     |
| Episode termination after hit | Ends episode |

---

## **State Representation**

State is currently represented as a simple tuple:

```
(player_row, enemy_row)
```

This keeps the environment very small, so Q-Learning converges quickly.

---

## **Actions**

```
0 → Move Up
1 → Move Down
2 → Shoot
3 → Stay Still
```

---

## **Why This Is a Good RL Project**

* Very easy to implement & train.
* Still grounded in core RL theory:

  * Markov Decision Processes (MDPs)
  * Action–value methods (Q-Learning)
  * Exploration vs. exploitation
  * Sparse vs. dense reward shaping
* Can be extended gradually (Option B, C...) to show meaningful improvement.

---

## **Training**

A simple Q-Learning agent can learn the optimal policy in **a few seconds** because:

* State space is tiny
* Only one enemy
* Reward is clear and sparse

---

## **How to Run**

Basic pipeline:

```
env = ShooterEnv()
obs = env.reset()

for episode in range(num_episodes):
    done = False
    while not done:
        action = agent.choose_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.update(obs, action, reward, next_obs)
        obs = next_obs
```

Render example:

```
........E
.........
P........
.........
```

---

## **Next Steps (Option B & beyond)**

Once this version works, we can expand it with:

* Moving enemies
* Obstacles
* Multiple enemies
* Non-combatants (negative reward for shooting them)
* Ammo limits
* Partial observability
* DQN or PPO training

---

If you're ready, we can now move to **Option B (Moving Enemy + Ammo + Better Rendering)**.
