import gym
#import gym_robo
from multiprocessing import Process
#from hyq_task6 import HyQTask6
import time
'''
robot_kwargs = {
                'use_gui': False,
                'rtf': 0.0,
                'control_mode': "Absolute",
                'controller_type': "force"
            }
task_kwargs={
        'max_time_step': 1000,
        'reach_bounds_penalty': 12.0,
        'fall_penalty': 20,  # minimize 20             # plot exploration rate and learning rate plot out,   # graph on all absolute action, absolute sum per time step , 
        'x_direction_value' :1.0,
        'y_direction_value':0.0
}'''
# task_kwargs={
#         'max_time_step': 1000,
#         'reach_bounds_penalty': 10.0,
#         'fall_penalty': 100.0,
#         'rew_scaling': 100
# }
class env_func:
    def __init__(self):
        print("Loading env1")
        self.env1 = gym.make('BipedalWalker-v3')
        print("Loading env2")
        self.env2 = gym.make('BipedalWalker-v3')
        self.env_count = 0

    def get_env(self):
        # Assume that get_env will always be called 2 times at once, first time being training env and 2nd time being test
        if(self.env_count % 2 == 0):
            self.env_count += 1
            return self.env1
        else:
            self.env_count += 1
            return self.env2


class exploration:
    def __init__(self):
         self.threshold = 0.95  # 0.95-->0.9 0.85 0.8 0.75     0.3         0.295 0.29                0.01    

    def get_threshold(self, timestep):
        
        if timestep > 500000 and timestep % 40000 == 0 and timestep < 1750000:
            self.threshold = 0.3
        elif timestep > 10000 and timestep % 40000 == 0 and self.previoustimestep != timestep:
            if self.threshold <= 0.3:
                self.threshold -= 0.005
            else:
                self.threshold -= 0.05
        
        self.previoustimestep = timestep

        if self.threshold < 0.01:
            self.threshold = 0.01
        return self.threshold

def q_learning_rate(timestep):
    base_lr = 0.001
    lr = base_lr - (timestep // 2e6) * 0.0001  #  0.001   0.0009   0.0008
    if lr < 0.000001:  #2 000 000
        lr =0.0000000001
    if timestep > 20000 and round(timestep,-2)%50000 ==0:
        print('lr:',lr)
    return lr

env = env_func()

logger_kwargs = dict(output_dir='data/BipedalWalkerv3', exp_name='BipedalWalkerv3')
from td3 import td3
if __name__ == '__main__':
    run_kwargs_base = {'env_fn': env.get_env, 
                  'logger_kwargs': logger_kwargs,
                  'epochs': 10,
                  'random_exploration': exploration().get_threshold,
                  'num_test_episodes': 1
                  #'load_model_file': '/home/siwflhc/Desktop/forward_task_docker/aws/210507/additiongaits4/data/gaitforward/pyt_save/model.pt',
                  #'save_checkpoint_path': "data"
                }
    run_kwargs_normal = {'env_fn': env.get_env, 
                  'logger_kwargs': logger_kwargs,
                  'epochs': 10,
                  'random_exploration': exploration().get_threshold,
                  'num_test_episodes': 1
                  #'load_checkpoint_path': "data"
                }
    for i in range(20):
        if  i == 0:
            run_kwargs = run_kwargs_base
            #env.env_count = 0
            #env.env1.reset()
            #env.env2.reset()
        else:
            env.env_count = 0
            run_kwargs = run_kwargs_normal
        td3(**run_kwargs)

        print(f"Finished task {i}")

#td3(env_fn=env_fn, logger_kwargs=logger_kwargs, epochs=100, random_exploration=exploration().get_threshold, num_test_episodes=0, save_checkpoint_path="/opt/run")

# # --- Testing Script ---
# from spinup.utils.test_policy import load_policy_and_env, run_policy
# fpath = 'Mdata/HyQP2'
# env, get_action = load_policy_and_env(fpath,itr ='last',deterministic=False)
# run_policy(env_fn, get_action, max_ep_len=0, num_episodes=10, render=True) 

