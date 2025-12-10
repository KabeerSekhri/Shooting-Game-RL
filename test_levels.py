from shooting_opt_d import ShootingEnvB, ShootingUI
import pygame
import pickle
import numpy as np

def play_trained_agent(q_table_path="q_table_option_d.pkl"):
    # Load trained Q-table
    with open(q_table_path, "rb") as f:
        Q = pickle.load(f)

    env = ShootingEnvB()
    ui = ShootingUI(env)  # GUI rendering

    max_level = getattr(env, "max_level", 5)  # default 5 levels

    for level in range(1, max_level + 1):
        print(f"\n=== Playing Level {level} ===")
        env.level = level
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Compute compact state index
            compact_obs = env.get_compact_state()
            state_idx = env.compact_state_to_index(compact_obs)
            
            # Pick best action from Q-table
            action = int(np.argmax(Q[state_idx]))
            
            # Take step
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Render GUI
            ui.render_ui(fps=8)

        print(f"Level {level} finished | Reward: {total_reward} | Ammo left: {getattr(obs, 'ammo', obs[2])}")

    print("\nAll levels completed!")

if __name__ == "__main__":
    play_trained_agent()
