#import os 
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import gym
import gym_robo
from multiprocessing import Process

robot_kwargs = {
                'use_gui': False,
                'rtf': 0.0,
                'control_mode': "Absolute"
            }
task_kwargs = {
    'max_time_step': 1000,
    'reach_bounds_penalty': 10.0,
    'fall_penalty': 50.0,
     'rew_scaling': 100
}  #

env_fn = lambda : gym.make('HyQ-v3',task_kwargs=task_kwargs, robot_kwargs=robot_kwargs, enable_logging=False)


logger_kwargs = dict(output_dir='data/HyQDockerRun1', exp_name='HyQDockerRun1')
from spinup import td3_pytorch as td3
td3(env_fn=env_fn, logger_kwargs=logger_kwargs, start_steps=5000, epochs=5, num_test_episodes=0, steps_per_epoch=4000,
    replay_size=5000, load_model_path="/home/pohzhiee/Documents/spinningup_poh/run_scripts/model", random_exploration=0.0)