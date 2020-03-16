import os
import sys
import argparse
import pkg_resources
import importlib
import warnings
import numpy as np
from tqdm import tqdm

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
import numpy as np
try:
    import highway_env
except ImportError:
    highway_env = None
import stable_baselines
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams, find_saved_model

# Fix for breaking change in v2.6.0
if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
    sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
    stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer

# Sample one trajectory (until trajectory end)
def traj_1_generator(model, env, deterministic):

    ob = env.reset()
    done = False  
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    dones = []
    acs = []

    while True:
        action, _ = model.predict(ob, deterministic=deterministic)
        # Random Agent
        # action = [env.action_space.sample()]
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)

        obs.append(ob)
        dones.append(done)
        acs.append(action)

        ob, reward, done, infos = env.step(action) # update ob and done.
        rews.append(reward)  

        cur_ep_ret += reward
        cur_ep_len += 1
        if done:
            break

    obs = np.array(obs)
    rews = np.array(rews)
    dones = np.array(dones)
    acs = np.array(acs)

    # log out expert trajs information
    print("Episode Reward:{}".format(cur_ep_ret))
    print("Episode Length:{}".format(cur_ep_len))

    assert infos
    episode_infos = infos[0].get('episode')
    episode_infos_return = None
    episode_infos_length = None
    if episode_infos:
        print("ale.lives: {}".format(infos[0].get('ale.lives')))
        print("Atari Episode Score: {}".format(episode_infos['r']))
        print("Atari Episode Length:{}".format(episode_infos['l']))
        episode_infos_return = episode_infos['r']
        episode_infos_length = episode_infos['l']
    # assert episode_infos['l'] == cur_ep_len

    traj = {"ob": obs, "rew": rews, "done": dones, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}

    return traj, episode_infos_return, episode_infos_length


def runner(env, env_id, model, number_trajs, deterministic, save=True, save_dir=None):

    if deterministic:
        print('using deterministic policy.')
    else:
        print('using stochastic policy.')

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    info_score_list = []
    info_length_list = []
    lives_num = 0
    lives_countable = True

    for _ in tqdm(range(number_trajs)):
        traj, episode_infos_score, episode_infos_length = traj_1_generator(model, env, deterministic)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
        if lives_countable:
            lives_num += 1
        if episode_infos_score:
            info_score_list.append(episode_infos_score)
            lives_countable = False
        if episode_infos_length:
            info_length_list.append(episode_infos_length)

    if save:
        filename = save_dir + "/" + env_id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))

    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    avg_info_score = sum(info_score_list)/len(info_score_list)
    avg_info_length = sum(info_length_list)/len(info_length_list)
    print("Live number of this game:", lives_num)
    print("Transitions:", sum(len_list))
    print("Expert Rewards:", avg_info_score/lives_num)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    print('Average info score', avg_info_score)
    print('Average info length', avg_info_length)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
    # parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
                        # type=int)
    parser.add_argument('-n', '--n-episodes', help='number of episodes to collect', default=20,
                        type=int)                 
    parser.add_argument('--n-envs', help='number of environments', default=1,
                        type=int)
    parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=-1,
                        type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--no-render', action='store_true', default=False,
                        help='Do not render the environment (useful for tests)')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', default=False,
                        help='Use stochastic actions (for DDPG/DQN/SAC)')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='Normalize reward if applicable (trained with VecNormalize)')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env 
    algo = args.algo
    folder = args.folder 

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print('Loading latest experiment, id={}'.format(args.exp_id))

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, args.exp_id))
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)
    log_dir = args.reward_log if args.reward_log != '' else None

    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
            args.n_envs = 1
    set_global_seeds(args.seed)
    is_atari = 'NoFrameskip' in env_id

    env = create_test_env(env_id, n_envs=args.n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                          should_render=not args.no_render,
                          hyperparams=hyperparams)

    model_path = find_saved_model(algo, log_path, env_id)
    model = ALGOS[algo].load(model_path, env=env)

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not args.stochastic
    save_dir = os.path.join("expert_trajs", algo)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    runner(env, env_id, model, args.n_episodes, deterministic, save=True, save_dir=save_dir)


if __name__ == '__main__':
    main()
