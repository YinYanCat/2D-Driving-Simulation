class DeepSarsa:
    def __init__(self, state_dim, action_dim, learning_rate, discount_factor, environment):
        agent = DQNagent(state_dim, action_dim, learning_rate, discount_factor)
        self.env = environment

    def train(self, n_episodes, n_steps):
        for eps in range(n_episodes):
            state_0 = ...
            action = ...
            total_reward = ...
            stop = ...
            if stop:
                pass
            else:
                for step in range(n_steps):
                    state_1 = ...
                    curr_reward = ...
                    stop = ...
                    if stop:

                        break

            agent.update()




