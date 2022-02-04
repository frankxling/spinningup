from copy import deepcopy
import os
import random
import itertools
import numpy as np
import torch
from torch.optim import Adam
from torch.optim import SGD
from typing import Dict, Union, Callable
import time
import spinup.algos.pytorch.td3.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.torch_algo_utils import update_learning_rate, get_schedule_fn
import glob

def scale_action(action_space, action):
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0

def unscale_action(action_space, scaled_action):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param scaled_action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))

def scale_obs(obs_space, obs):
    low, high = obs_space.low, obs_space.high
    return 2.0 * ((obs - low) / (high - low)) - 1.0

def unscale_obs(obs_space, scaled_obs):
    low, high = obs_space.low, obs_space.high
    return low + (0.5 * (scaled_obs + 1.0) * (high - low))

def load_latest_state_dict(path: str):
    files_path = os.path.join(path, '*.pt')
    list_of_files = glob.glob(files_path)
    latest_save_file = max(list_of_files, key=os.path.getctime)
    state_dict = torch.load(latest_save_file)
    return state_dict

def delete_old_files(path: str, max_num_files: int = 10):
    assert max_num_files > 0, "Maximum number of checkpoint files should be a positive number"
    files_path = os.path.join(path, '*.pt')
    list_of_files = glob.glob(files_path)
    if len(list_of_files) > max_num_files:
        num_files_to_remove = len(list_of_files) - max_num_files
        for _ in range(num_files_to_remove):
            oldest_save_file = min(list_of_files, key=os.path.getctime)
            print(f"Deleting {oldest_save_file}")
            os.remove(oldest_save_file)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs]) 
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}#torch.as_tensor([0,0,0], device=torch.device('cuda')



def td3(env_fn: Callable,
        actor_critic: torch.nn.Module = core.MLPActorCritic,
        ac_kwargs: Dict = None,
        seed: int = 0,
        steps_per_epoch: int = 4000,
        epochs: int = 2000,
        replay_size: int = int(1e6),
        gamma: float = 0.99,
        polyak: float = 0.995,
        pi_lr: Union[Callable, float] = 1e-3,
        q_lr: Union[Callable, float] = 1e-3,
        batch_size: int = 100,
        start_steps: int = 10000,
        update_after: int = 1000,
        update_every: int = 100,
        act_noise: Union[Callable, float] = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        num_test_episodes: int = 1,
        max_ep_len: int = 1000,
        logger_kwargs: Dict = None,
        save_freq: int = 1,
        random_exploration: Union[Callable, float] = 0.0,
        save_checkpoint_path: str = None,
        load_checkpoint_path: str = None,
        load_model_file: str = None):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float or callable): Learning rate for policy.

        q_lr (float or callable): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float or callable): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        random_exploration (float or callable): Probability to randomly select
            an action instead of selecting from policy.

        save_checkpoint_path (str): Path to save the model. If not set, no model
            will be saved

        load_checkpoint_path (str): Path to load the model. Cannot be set if
            save_model_path is set.
    """
    if logger_kwargs is None:
        logger_kwargs = dict()
    if ac_kwargs is None:
        ac_kwargs = dict()

    if save_checkpoint_path is not None:
        assert load_checkpoint_path is None, "load_model_path cannot be set when save_model_path is already set"
        if not os.path.exists(save_checkpoint_path):
            print(f"Folder {save_checkpoint_path} does not exist, creating...")
            os.makedirs(save_checkpoint_path)

    if load_checkpoint_path is not None:
        assert load_model_file is None, "load_checkpoint_path cannot be set when load_model_file is already set"
    # ------------ Initialisation begin ------------
    loaded_state_dict = None
    if load_checkpoint_path is not None:
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())
        loaded_state_dict = load_latest_state_dict(load_checkpoint_path)

        logger.epoch_dict = loaded_state_dict['logger_epoch_dict']
        q_learning_rate_fn = loaded_state_dict['q_learning_rate_fn']
        pi_learning_rate_fn = loaded_state_dict['pi_learning_rate_fn']
        epsilon_fn = loaded_state_dict['epsilon_fn']
        act_noise_fn = loaded_state_dict['act_noise_fn']
        replay_buffer = loaded_state_dict['replay_buffer']
        #env, test_env = loaded_state_dict['env'], loaded_state_dict['test_env']
        # Assume that env_fn will return existing env object instead of creating new
        env, test_env = env_fn(), env_fn()
        env.reset() #, test_env.reset()
        ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        ac_targ = deepcopy(ac)  #deepcopy, to confirm whether the tensor and bias is the same result 
        ac.load_state_dict(loaded_state_dict['ac'])
        ac_targ.load_state_dict(loaded_state_dict['ac_targ'])
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        env.action_space.np_random.set_state(loaded_state_dict['action_space_state'])

        # List of parameters for both Q-networks (save this for convenience)
        q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
        t_ori = loaded_state_dict['t']
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_learning_rate_fn(t_ori))
        pi_optimizer.load_state_dict(loaded_state_dict['pi_optimizer'])
        q_optimizer = Adam(q_params, lr=q_learning_rate_fn(t_ori))
        q_optimizer.load_state_dict(loaded_state_dict['q_optimizer'])
        np.random.set_state(loaded_state_dict['np_rng_state'])
        torch.set_rng_state(loaded_state_dict['torch_rng_state'])

    else:
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        q_learning_rate_fn = get_schedule_fn(q_lr)
        pi_learning_rate_fn = get_schedule_fn(pi_lr)
        act_noise_fn = get_schedule_fn(act_noise)
        epsilon_fn = get_schedule_fn(random_exploration)

        env, test_env = env_fn(), env_fn()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        env.action_space.seed(seed)

        # Experience buffer
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)


        # Create actor-critic module and target networks
        if load_model_file is not None:
            assert os.path.exists(load_model_file), f"Model file path does not exist: {load_model_file}"
            ac = torch.load(load_model_file)
        else:
            ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        ac_targ = deepcopy(ac)

        # List of parameters for both Q-networks (save this for convenience)
        q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

        # Set up optimizers for policy and q-function
        pi_optimizer = Adam(ac.pi.parameters(), lr=pi_learning_rate_fn(0))
        q_optimizer = Adam(q_params, lr=q_learning_rate_fn(0))

        #pi_optimizer = SGD(ac.pi.parameters(), lr=pi_learning_rate_fn(0))
        #q_optimizer = SGD(q_params, lr=q_learning_rate_fn(0))
        t_ori = 0

    act_limit = 1.0

    # ------------ Initialisation end ------------

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False



    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    #torch.set_printoptions(profile="default")
    torch.set_printoptions(precision=10,linewidth=100, edgeitems=20,threshold=100)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)
            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)
            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)  #criticize o2 and a2. whether during o2 , do a2 okay or not. 
            q2_pi_targ = ac_targ.q2(o2, a2)   #criticize o2 and a2. whether during o2 , do a2 okay or not. 
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)   #choose the minimal from all the tensor
            backup = r + gamma * (1 - d) * q_pi_targ #reward Q table 
            #torch.set_printoptions(profile='full')#precision=2,linewidth=80, edgeitems=3,threshold=60000)
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q1_pi = ac.q1(o, ac.pi(o))
        return -q1_pi.mean()

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward() 
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += torch.as_tensor(noise_scale * np.random.randn(act_dim), dtype=torch.float32)
        return np.clip(a.tolist(), -act_limit, act_limit)

    def test_agent():
        sum = 0
        for _ in range(num_test_episodes):
            
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            #test_env.render()
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                scaled_action = get_action(o, 0)
                o, r, d, _ = test_env.step(unscale_action(env.action_space, scaled_action),threshold3 )#,q_lr, explo
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            sum += ep_ret
        test_average = sum / num_test_episodes
        return test_average

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    highest_test_reward = 0
    threshold2 = 0.9
    i22=0
    if loaded_state_dict is not None:
        o = loaded_state_dict['o']
        ep_ret = loaded_state_dict['ep_ret']
        ep_len = loaded_state_dict['ep_len']
        threshold2 = loaded_state_dict['threshold2']
        i22 = loaded_state_dict['i22']
        highest_test_reward = loaded_state_dict['highest_test_reward']
    else:
        o, ep_ret, ep_len = env.reset(), 0, 0
    # Main loop: collect experience in env and update/log each epoch
    #env.render()
    
    for t in range(total_steps):
        t += t_ori
        # printMemUsage(f"start of step {t}")
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, s
        # use the larned policy (with some noise, via act_noise).
        #explo= epsilon_fn(t)
        if t > start_steps and np.random.rand() > epsilon_fn(t):
            a = get_action(o, act_noise_fn(t))
            unscaled_action = unscale_action(env.action_space, a)
        else:
            unscaled_action = env.action_space.sample()
            a = scale_action(env.action_space, unscaled_action)
            
        threshold3=epsilon_fn(t)
        
        # Step the env
        o2, r, d, _ = env.step(unscaled_action,threshold3)#,threshold2)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Early stop if cum_reward lower than -500
        elif ep_ret < -300:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0
            
        # Update handling
        if t >= update_after and t % update_every == 0:  # each step, 1000 
            for j in range(update_every): #during timestep, it will update with len(update_every)
                batch = replay_buffer.sample_batch(batch_size)  #100 # 1, 150[]  2, 30[]  3, 50[]            
                update(data=batch, timer=j) # 150 [1111]     50[1001]
            #print('timestep during update',t)    
            #print(batch['obs'].size(),batch['act'].size())
        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            # Perform LR decay
            update_learning_rate(q_optimizer, q_learning_rate_fn(t))
            update_learning_rate(pi_optimizer, pi_learning_rate_fn(t))
            epoch = (t + 1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            average_test = test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            # Save model and checkpoint
            save_checkpoint = False
            checkpoint_path = ""
            if save_checkpoint_path is not None:
                save_checkpoint = True
                checkpoint_path = save_checkpoint_path
            if load_checkpoint_path is not None:
                save_checkpoint = True
                checkpoint_path = load_checkpoint_path
            if (epoch % save_freq == 0) or (epoch == epochs):
                if average_test > highest_test_reward:
                    logger.save_state({}, None)
                    print('Saved highest reward model!')
                    #time.sleep(2.0)
                    highest_test_reward = average_test

                if save_checkpoint:
                    checkpoint_file = os.path.join(checkpoint_path, f'save_{epoch}.pt')
                    torch.save({'ac': ac.state_dict(),
                                'ac_targ': ac_targ.state_dict(),
                                'replay_buffer': replay_buffer,
                                'pi_optimizer': pi_optimizer.state_dict(),
                                'q_optimizer': q_optimizer.state_dict(),
                                'logger_epoch_dict': logger.epoch_dict,
                                'q_learning_rate_fn': q_learning_rate_fn,
                                'pi_learning_rate_fn': pi_learning_rate_fn,
                                'epsilon_fn': epsilon_fn,
                                'act_noise_fn': act_noise_fn,
                                'torch_rng_state': torch.get_rng_state(),
                                'np_rng_state': np.random.get_state(),
                                'action_space_state': env.action_space.np_random.get_state(),
                                #'env': env,
                                #'test_env': test_env,
                                'ep_ret': ep_ret,
                                'ep_len': ep_len,
                                'threshold2' : threshold2,
                                'i22' : i22,
                                'o': o,
                                'highest_test_reward': highest_test_reward,
                                't': t+1}, checkpoint_file)
                    delete_old_files(checkpoint_path, 2)


if __name__ == '__main__':
    raise RuntimeError("This file is not supposed to be ran, please use another script to run this file")