import gym
import gym_robo
#from hyq_task6 import HyQTask6
from multiprocessing import Process

'''robot_kwargs = {
    'use_gui': False,
    'rtf' : 0, #1 10   7.0 before 5k rendering off.
    'control_mode': "Absolute"
    }#   'state_noise_mu': 0, 'state_noise_sigma': 0.075  'random_init_pos': False,
task_kwargs = {
                'max_time_step': 1000,
                'reach_bounds_penalty': 10.0,
                'fall_penalty': 50.0,
                'boundary_radius': 0.5,
                'out_of_boundary_penalty': 100.0,
                'action_space_range': 0.1

}  #'''
#env_fn = lambda : gym.make('HyQ-v4', task_kwargs=task_kwargs, robot_kwargs=robot_kwargs)
env_fn = lambda : gym.make('HyQ-v6')

class exploration:
    def __init__(self):
         self.threshold = 0.9
    
    def get_threshold(self, timestep):
         if timestep > 20000 and timestep % 25000 == 0:
             self.threshold -= 0.1
         if self.threshold < 0.0:
             self.threshold = 0.0
         return self.threshold

def learning_rate(timestep):
    base_lr = 0.001
    lr = base_lr - (timestep // 1e6) * 0.0001
    if lr < 0.0000001:
        lr =0.0000001
    if timestep > 20000 and round(timestep,-2)%50000 ==0:
        print('lr:',lr)
    return lr


logger_kwargs = dict(output_dir='data/HYQ_task6_forward_minus_speed_error', exp_name='HYQ_task6_forward_minus_speed_error')#1:1
from spinup import td3_pytorch as td3
if __name__ == '__main__':
    run_kwargs_base = {'env_fn': env_fn, 
                  'logger_kwargs': logger_kwargs,
                  'epochs': 200,
                  'random_exploration': exploration().get_threshold,
                  'num_test_episodes': 1,
                  'save_checkpoint_path': "data/HYQ_task6_forward_minus_speed_error"
                }
    run_kwargs_normal = {'env_fn': env_fn, 
                  'logger_kwargs': logger_kwargs,
                  'epochs': 200,
                  'random_exploration': exploration().get_threshold,
                  'num_test_episodes': 1,
                  'load_checkpoint_path': "data/HYQ_task6_forward_minus_speed_error"
                }
    
    for i in range(5):
        if  i == 0:
            run_kwargs = run_kwargs_base
        else:
            run_kwargs = run_kwargs_normal
        p = Process(target=td3, kwargs=run_kwargs)
        p.start()
        p.join()

        print(f"Finished task {i}")

#td3(env_fn=env_fn, logger_kwargs=logger_kwargs, epochs=100, random_exploration=exploration().get_threshold, num_test_episodes=0, save_checkpoint_path="/opt/run")

# --- Testing Script ---
#from spinup.utils.test_policy import load_policy_and_env, run_policy
#fpath = '/home/joann/spinningup/spinup/algos/pytorch/td3/Pdata/HyQP6'
#env, get_action = load_policy_and_env(fpath,itr ='last',deterministic=False)
#run_policy(env_fn(), get_action, max_ep_len=0, num_episodes=10, render=True) 
