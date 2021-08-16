#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gym_minigrid.window import Window


def redraw(img, env, window):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)


def reset(env, window):
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs, env, window)


def step(env, window, action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset(env, window)
    else:
        redraw(obs, env, window)


def main(args):
    env = gym.make(args.env)

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    window = Window('gym_minigrid - ' + args.env)

    def key_handler(event):
        print('pressed', event.key)

        if event.key == 'escape':
            window.close()
            return

        if event.key == 'backspace':
            reset()
            return

        if event.key == 'left':
            step(env, window, env.actions.left)
            return
        if event.key == 'right':
            step(env, window, env.actions.right)
            return
        if event.key == 'up':
            step(env, window, env.actions.forward)
            return

        # Spacebar
        if event.key == ' ':
            step(env, window, env.actions.toggle)
            return
        if event.key == 'pageup':
            step(env, window, env.actions.pickup)
            return
        if event.key == 'pagedown':
            step(env, window, env.actions.drop)
            return

        if event.key == 'enter':
            step(env, window, env.actions.done)
            return

    window.reg_key_handler(key_handler)

    reset(env, window)

    # Blocking event loop
    window.show(block=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        help="size at which to render tiles",
        default=32
    )
    parser.add_argument(
        '--agent_view',
        default=False,
        help="draw the agent sees (partially observable view)",
        action='store_true'
    )
    args = parser.parse_args()
    main(args)
