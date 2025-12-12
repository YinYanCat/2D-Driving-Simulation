from src.Environment import Environment
from src.DeepSarsa import DeepSarsa

import numpy as np
import pygame


def player_main(verbose=0):

    env = Environment(max_steps=4000)
    env.reset()
    pygame.init()

    clock = pygame.time.Clock()
    running = True
    state_dim, action_dim = env.get_dim()
    total_reward = 0
    step = 0
    while running:
        dt = clock.tick(60) / 1000  # segundos por frame
        step += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        keys = pygame.key.get_pressed()

        action = np.zeros(action_dim, dtype=int)

        if keys[pygame.K_r]:
            env.reset()
            total_reward = 0
            step = 0

            continue

        if keys[pygame.K_1]:
            action[0] = 0
        if keys[pygame.K_2]:
            action[0] = 1
        if keys[pygame.K_3]:
            action[0] = 2
        if keys[pygame.K_4]:
            action[0] = 3
        if keys[pygame.K_SPACE]:
            action[1] = 1
        elif keys[pygame.K_LSHIFT]:
            action[1] = 2
        else:
            action[1] = 0


        if keys[pygame.K_d]:
            action[2] = 0
        elif keys[pygame.K_a]:
            action[2] = 2
        elif keys[pygame.K_w]:
            action[2] = 1
        else:
            action[2] = 3

        image, state, reward, done = env.step(action, dt=dt)
        if not done:
            total_reward += reward
            if verbose >= 1:
                print(f"Step: {step}    Reward: {reward:.2f}    Total Reward: {total_reward:.2f}")


def main():
    play = True
    train_from_load = False
    env = Environment(render=play, max_steps=4000)
    env.reset()
    learning_rate = 0.001
    discount_factor = 0.875

    # Crear agente
    agent = DeepSarsa(learning_rate, discount_factor, env)

    if play: # evaluación
        if agent.load():
            agent.play(n_episodes=1, n_steps=4000, verbose=0)
        else:
            return
    else: # entrenado
        if train_from_load:
            agent.load()
        # Ojala epsilon_dec = 1*x^(n_steps/2)=0.222...
        agent.train(600, 2000, epsilon_dec=0.995,verbose=1)
        agent.save()


if __name__ == '__main__':
    main()
    #player_main(verbose=1)