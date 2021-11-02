import gym
env_fn = gym.make('BipedalWalker-v3')
from test_policy import load_policy_and_env, run_policy
fpath = 'data/BipedalWalkerv3'
env, get_action = load_policy_and_env(fpath,itr ='last',deterministic=False)
run_policy(env_fn, get_action, max_ep_len=0, num_episodes=10, render=True) 