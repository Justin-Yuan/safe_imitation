"""
Generate trajectory data randomly from environment using random action
sequences and ground truth cost function to generate cost/safety labels

See example command before for generating data:

python scripts/generate_data.py
"""
import gym
import numpy as np


def generate_trajectories(env,
                          num_train_rollouts=10000,
                          num_test_rollouts=100,
                          render=True):

    train_trajectories = []
    for i in range(num_train_rollouts):
        state = env.reset(random_start=True)
        rollout = []

        # Perform the same action repeatedly to move in a straight line
        action = env.action_space.sample()
        for _ in range(10):
            state, reward, done, info = env.step((action))

            if render:
                env.render()

            rollout.append((state, action, reward, info['cost']))

        train_trajectories.append(rollout)

    np.save('data/train.npy', np.asarray(train_trajectories, dtype=object))

    test_trajectories = []
    for i in range(num_test_rollouts):
        state = env.reset(random_start=True)
        rollout = []

        # Perform the same action repeatedly to move in a straight line
        action = env.action_space.sample()
        for _ in range(10):
            state, reward, done, info = env.step((action))

            if render:
                env.render()

            rollout.append((state, action, reward, info['cost']))

        test_trajectories.append(rollout)

    np.save('data/test.npy', np.asarray(test_trajectories, dtype=object))


if __name__ == '__main__':
    env = gym.make('safe_il:SimpleNavigation-v0')
    generate_trajectories(env, render=False)
