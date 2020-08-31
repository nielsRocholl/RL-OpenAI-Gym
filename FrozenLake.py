import gym
import numpy as np
import random
import time
from IPython.display import clear_output


class Env:

    def __init__(self):
        self.env = gym.make("FrozenLake-v0")
        self.action_space_size = self.env.action_space.n
        self.state_space_size = self.env.observation_space.n


def train():
    # create environment
    env = Env()
    #   create Q-table
    q_table = np.zeros((env.state_space_size, env.action_space_size))

    # set parameters
    episodes = 10000
    steps = 100

    learning_rate = 0.1
    discount_rate = 0.99

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.001
    exploration_decay = 0.001

    rewards_all_episodes = []

    for episode in range(episodes):
        state = env.env.reset()

        # done = False
        rewards_current_episode = 0

        for step in range(steps):

            # exploration vs exploitation
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state, :])
            else:
                action = env.env.action_space.sample()

            new_state, reward, done, info, = env.env.step(action)

            # update the q-table
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * \
                                     (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            rewards_current_episode += reward

            if done:
                break

        # exploration rate decay
        exploration_rate = min_exploration_rate + \
                           (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay * episode)
        rewards_all_episodes.append(rewards_current_episode)

    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), episodes / 1000)
    count = 1000

    print("********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r / 1000)))
        count += 1000

    print(q_table)
    return q_table


def play(q_table):
    env = Env()
    for episode in range(3):
        state = env.env.reset()
        print("***** EPISODE ", episode + 1, "*****\n\n\n")
        time.sleep(1)

        for step in range(100):
            clear_output(wait=True)
            env.env.render()
            time.sleep(0.3)
            action = np.argmax(q_table[state, :])
            new_sate, reward, done, info = env.env.step(action)

            if done:
                clear_output(wait=True)
                env.env.render()
                if reward == 1:
                    print("***** You reached the goal! *****")
                    time.sleep(3)
                else:
                    print("***** You fell in the water! *****")
                    time.sleep(3)
                clear_output(wait=True)
                break

            state = new_sate

    env.env.close()


def main():
    q_table = train()
    play(q_table)


if __name__ == '__main__':
    main()
