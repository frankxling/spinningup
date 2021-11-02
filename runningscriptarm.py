import gym
import gym_robo
#from hyq_task6 import HyQTask6
from multiprocessing import Process
#python /home/siwflhc/spinningupmodified/runningscriptarm.py
robot_kwargs = {'use_gui': False, 'rtf' :0,'controller_type': "force"
    }
task_kwargs = {
'max_time_step':      500,               # Maximm time step before stopping the episoderosbiped
'random_goal_file': None,    # Path to the numpy save file containing the random goals'
'accepted_dist_to_bounds': 0.0010000 ,   # Allowable distance to joint limits (radians)  0.002  1400epoch  90, %++
'accepted_error': 0.0010000  ,           # Allowable distance from target coordinates (metres)
'reach_target_bonus_reward': 0.0000000 ,# Bonus reward upon reaching target
'reach_bounds_penalty': 10.0000000  ,    # Reward penalty when reaching joint limit  38/18
'contact_penalty': 10.0000000   ,        # Reward penalty for collision   38/18
'episodes_per_goal':        1  ,         # Number of episodes before generating another random goal
'goal_buffer_size':       1  ,        # Number goals to store in buffer to be reused later  50
'goal_from_buffer_prob': 0.0000000  ,    # Probability of selecting a random goal from the goal buffer, value between 0 and 1
'num_adjacent_goals':        0  ,        # Number of nearby goals to be generated for each randomly generated goal 
'random_goal_seed': 2     ,         # Seed used to generate the random goals 10            18
'is_validation':   False    ,    #Test policy then put **True**      # Whether this is a validation run, if true will print which points failed and how many reached
'normalise_reward':   True,          # Perform reward normalisation, this happens before reward bonus and penalties
'continuous_run':    False    ,          # Continuously run the simulation, even after it reaches the destination
'reward_noise_mu': None      ,      # Reward noise mean (reward noise follows gaussian distribution)
'reward_noise_sigma': None  ,       # Reward noise standard deviation, recommended 0.5
'reward_noise_decay': None  ,          # Constant for exponential reward noise decay (recommended 0.31073, decays to 0.002 in 20 steps)
'exp_rew_scaling': 4.8  #lower it        # Constant for exponential reward scaling (None by default, recommended 5.0, cumulative exp_reward = 29.48)'''  X

} 
env_fn = lambda : gym.make('LobotArmContinuous-v2', task_kwargs=task_kwargs, robot_kwargs=robot_kwargs)


class exploration:
    def __init__(self):
         self.threshold = 0.9
    
    def get_threshold(self, timestep):
         if timestep > 20000 and timestep % 25000 == 0:
             self.threshold -= 0.1
         if self.threshold < 0.0:
             self.threshold = 0.0
         return self.threshold

def q_learning_rate(timestep):
    base_lr = 0.001
    if timestep >6000000:
        lr = base_lr - (((timestep-6000000)// 1e5) * 2e-5 )
    if lr < 0.0000001:
        lr =0.00050
    if timestep > 20000 and round(timestep,-2)%50000 ==0:
        print('lr:',lr)
    return lr


logger_kwargs = dict(output_dir='data/armbesttest', exp_name='armbesttest')
from spinup import td3_pytorch as td3arm
if __name__ == '__main__':
    run_kwargs_base = {'env_fn': env_fn, 
                  'logger_kwargs': logger_kwargs,
                  'epochs': 100,
                  'random_exploration': exploration().get_threshold,
                  'num_test_episodes': 1,
                  'seed' : 10,
                  'save_checkpoint_path': "data/armbesttest"
                }
    run_kwargs_normal = {'env_fn': env_fn, 
                  'logger_kwargs': logger_kwargs,
                  'epochs': 100,
                  'random_exploration': exploration().get_threshold,
                  'num_test_episodes': 1,
                  'seed' : 10,
                  'load_checkpoint_path': "data/armbesttest"
                }
    
    for i in range(10):
        if  i == 0:
            run_kwargs = run_kwargs_base
        else:
            run_kwargs = run_kwargs_normal
        p = Process(target=td3arm, kwargs=run_kwargs)
        p.start()
        p.join()

        print(f"Finished task {i}")

#td3(env_fn=env_fn, logger_kwargs=logger_kwargs, epochs=100, random_exploration=exploration().get_threshold, num_test_episodes=0, save_checkpoint_path="/opt/run")

# --- Testing Script ---
#from spinup.utils.test_policy import load_policy_and_env, run_policy
#fpath = '/home/joann/spinningup/spinup/algos/pytorch/td3/Pdata/HyQP6'
#env, get_action = load_policy_and_env(fpath,itr ='last',deterministic=False)
#run_policy(env_fn(), get_action, max_ep_len=0, num_episodes=10, render=True) 
