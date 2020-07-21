#import os 
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import gym
import gym_robo
#/home/siwflhc/anaconda3/envs/arm_gpu/bin/python /home/siwflhc/spinningup/spinup/runningscript33v2.py   
robot_kwargs = {
    'use_gui': True,
    'rtf' : 0, #1 10   7.0 before 5k rendering off.
    'control_mode': "Absolute"
    }#   'state_noise_mu': 0, 'state_noise_sigma': 0.075  'random_init_pos': False,
task_kwargs = {
'max_time_step':      500,               # Maximum time step before stopping the episoderosbiped
#'random_goal_file': None,    # Path to the numpy save file containing the random goals'
'accepted_dist_to_bounds': 0.00200 ,   # Allowable distance to joint limits (radians)  0.002  1400epoch  90%++  --> 0.002
'accepted_error': 0.05000  ,           # Allowable distance from target coordinates (metres)
'reach_target_bonus_reward': 0.0000000 ,# Bonus reward upon reaching target
'reach_bounds_penalty': 10.0000000  ,    # Reward penalty when reaching joint limit  38/18
'fall_penalty': 20.0000000   ,        # Reward penalty for collision   38/18
#'episodes_per_goal':        1  ,         # Number of episodes before generating another random goal
#'goal_buffer_size':       1  ,        # Number goals to store in buffer to be reused later  50
#'goal_from_buffer_prob': 0.0000000  ,    # Probability of selecting a random goal from the goal buffer, value between 0 and 1
#'num_adjacent_goals':        0  ,        # Number of nearby goals to be generated for each randomly generated goal 
#'random_goal_seed': 100    ,         # Seed used to generate the random goals 10            18
#'is_validation':   False    ,    #Test policy then put **True**      # Whether this is a validation run, if true will print which points failed and how many reached
'norm_rew_scaling':    300           # Perform reward normalisation, this happens before reward bonus and penalties
#'continuous_run':    False    ,          # Continuously run the simulation, even after it reaches the destination
#'exp_rew_scaling': 7.0            # Constant for exponential reward scaling (None by default, recommended 5.0, cumulative exp_reward = 29.48)'''

}  #

env_fn = gym.make('HyQ-v0',task_kwargs=task_kwargs, robot_kwargs=robot_kwargs)

ac_kwargs = dict(hidden_sizes=[256,256])    #64/32

logger_kwargs = dict(output_dir='data/HyQv0testjo_b8_corrected', exp_name='TD3')
from spinup import td3_pytorch as td3 
from spinup.utils.test_policy import load_policy_and_env, run_policy
td3(env=env_fn,start_steps=10000, max_ep_len=500,steps_per_epoch=5000, epochs=3000, ac_kwargs=ac_kwargs,save_freq=20,act_noise=0.1,logger_kwargs =logger_kwargs) #,logger_kwargs =logger_kwargs, ac_kwargs=ac_kwargs_5000,,logger_kwargs =logger_kwargs



#fpath='data/HyQv0testjo_b8_corrected/'  #set saved model folder to find tf1_save
#env,get_action=load_policy_and_env(fpath, itr='last',deterministic=False)
#run_policy(env_fn, get_action, max_ep_len=None, num_episodes=500, render=True)
