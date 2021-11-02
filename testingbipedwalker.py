#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from spinup import td3_pytorch as td3
import gym
env_fn = gym.make('BipedalWalker-v3')

ac_kwargs = dict(hidden_sizes=[256,256])  

logger_kwargs = dict(output_dir='data/bipedwalkertf2', exp_name='pytorchbipedwalker')
from spinup import td3_pytorch as td3 
td3(env_fn=env_fn,start_steps=10000, max_ep_len=2500,steps_per_epoch=5000,replay_size=int(5e4), epochs=50,random_exploration=0.6, ac_kwargs=ac_kwargs,save_freq=20,act_noise=0.1,logger_kwargs =logger_kwargs) #,logger_kwargs =logger_kwargs, ac_kwargs=ac_kwargs_5000,,logger_kwargs =logger_kwargs
'''
for i in range(0,2):
    fpath='data/bipedwalkertf2/' 
    _,get_action=load_policy_and_env(fpath, itr='last',deterministic=False)
    env_fn = gym.make('BipedalWalker-v3')
    td3(env_fn=env_fn,start_steps=0, max_ep_len=2500,steps_per_epoch=5000,replay_size=int(5e4), epochs=1000,random_exploration=0.6, ac_kwargs=ac_kwargs,save_freq=20,act_noise=0.1,logger_kwargs =logger_kwargs)'''