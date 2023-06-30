import gym
import mujoco_py
import torch
from stable_baselines3 import PPO, DDPG, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
#from sb3_contrib import TRPO
import os
import argparse
import numpy as np
from typing import Callable


def main():

    ALGOS = {"trpo": PPO, "ddpg": DDPG, "ppo": PPO, "td3": TD3, "sac": SAC}

    parser = argparse.ArgumentParser()
    parser.add_argument("-time_steps", help="time steps", default=4e6, type=int)
    parser.add_argument("-pi_arch", help="actor architecure", default=6, nargs="+", type=int)
    parser.add_argument("-crit_arch", help="critic architecture", default=[256, 256], nargs="+", type=int)
    parser.add_argument("--env", help="environment ID", type=str, default="Ant-v3")
    parser.add_argument("--algo", help="RL Algorithm", default="sac", type=str, required=False, choices=ALGOS)
    parser.add_argument("-seed", help="Random seed", default=8, type=int)
    parser.add_argument("-lr", help="learning rate", default=None, type=float)
    parser.add_argument("-gamma", help="discount factor", default=0.99, type=float)
    parser.add_argument("-batch", help="batch size", default=None, type=int)
    parser.add_argument("-buffer", help="buffer size", default=1000000, type=int)
    parser.add_argument("-clip", help="clip range", default=0.2, type=float)
    parser.add_argument("-grad_steps", help="gradient steps", default=-1, type=int)
    parser.add_argument("-act_noise", help="noise", default=0.1, type=float)
    parser.add_argument("-train_freq", help="evaluation frequency", default=1, type=tuple)
    parser.add_argument("-eval_freq", help="evaluation frequency", default=10000, type=int)
    parser.add_argument("-eval_eps", help="number of evaluation episodes", default=10, type=int)
    args = parser.parse_args()

    algo = args.algo
    seed = int(args.seed)
    act_noise = float(args.act_noise)
    eval_freq = int(args.eval_freq)
    eval_eps = int(args.eval_eps)
    time_steps = int(args.time_steps)
    buffer = int(args.buffer)
    pi_arch = args.pi_arch
    crit_arch = args.crit_arch

    if isinstance(pi_arch, int):
        pi_arch = [pi_arch]

    if isinstance(crit_arch, int):
        crit_arch = [crit_arch]

    # configure directory
    save_to = './' + args.env + '/' + args.algo + '/' + 'x'.join(map(str, pi_arch)) + '/' + str(seed) + '/'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    # create environment
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    env.seed(seed)
    eval_env.seed(seed)

    # The noise objects for off-policy algorithms
    if act_noise > 0:
        n_actions = env.action_space.shape[-1] or 1
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=act_noise * np.ones(n_actions))
    else:
        action_noise = None

    # scheduler for lr and clip
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining ** 2 * initial_value
        return func

    # On-policy or off-policy?
    if algo in ["ppo", "trpo"]:
        batch_size = args.batch or 64
        lr = args.lr or 0.0003
        model = ALGOS[algo]('MlpPolicy', env, seed=seed, gamma=args.gamma, batch_size=batch_size,
                            clip_range=args.clip, learning_rate=lr,
                            policy_kwargs=dict(net_arch=[dict(pi=pi_arch, vf=crit_arch)], activation_fn=torch.nn.ReLU))
    else:
        batch_size = args.batch or 100
        lr = args.lr or 0.0003
        model = ALGOS[algo]('MlpPolicy', env, seed=seed, learning_rate=lr, gamma=args.gamma, batch_size=batch_size,
                            learning_starts=10000, action_noise=action_noise, gradient_steps=args.grad_steps, verbose=0,
                            buffer_size=buffer, #device='mps', # ent_coef=0.2,
                            policy_kwargs=dict(net_arch=dict(pi=pi_arch, qf=crit_arch), activation_fn=torch.nn.ReLU))

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_to, log_path=save_to, eval_freq=eval_freq,
                                 deterministic=True, render=False, n_eval_episodes=eval_eps)

    # Train then, saving best and final models
    model.learn(time_steps, callback=eval_callback)
    model.save(save_to + 'model')


if __name__ == "__main__":
    main()

