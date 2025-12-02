from shooting_qlearning_opt_b import ShootingEnvB, ShootingUI, state_to_index
import pygame


if __name__ == "__main__":
    import pickle
    import numpy as np

    # load trained agent
    with open("q_table_option_b.pkl", "rb") as f:
        Q = pickle.load(f)

    env = ShootingEnvB()
    ui = ShootingUI(env)  # GUI with rectangles or sprites

    obs = env.reset()
    done = False

    while not done:
        # pick best action from Q-table
        height = env.height
        ammo_capacity = env.ammo_capacity
        state_idx = state_to_index(obs[0], obs[1], obs[2], obs[3], height, ammo_capacity)
        action = np.argmax(Q[state_idx])



        # env step
        obs, reward, done, info = env.step(action)

        # draw GUI
        ui.render(fps=8)

    pygame.time.wait(1000)
