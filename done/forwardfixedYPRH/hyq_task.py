from typing import Dict, Tuple
import numpy
import math
from HyQPy import HyQObservation
from gym_robo.utils import quaternion_to_euler
from gym_robo.robots import HyQSim
from gym_robo.tasks import HyQState
from gym.spaces import Box

#forward
class HyQTask:
    def __init__(self, robot: HyQSim, max_time_step: int = 1000,
                 accepted_dist_to_bounds=0.01, reach_bounds_penalty=0.0, fall_penalty=0.0,
                 rew_scaling=None,yaw_factor_power=1.0,roll_factor_power=1.0,height_factor_power=1.0,pitch_factor_power=1.0,y_penalty_factor=1.5):
        self.robot = robot
        self._max_time_step = max_time_step
        self.accepted_dist_to_bounds = accepted_dist_to_bounds
        self.reach_bounds_penalty = reach_bounds_penalty
        self.fall_penalty = fall_penalty
        self.rew_scaling = rew_scaling
        self.yaw_factor_power = yaw_factor_power
        self.roll_factor_power = roll_factor_power
        self.height_factor_power = height_factor_power
        self.pitch_factor_power = pitch_factor_power
        self.y_penalty_factor = y_penalty_factor
        
        print(f'-------------------------------Setting task parameters-------------------------------')
        print('max_time_step: %8d               # Maximum time step before stopping the episode' % self._max_time_step)
        print('accepted_dist_to_bounds: %8.7f    # Allowable distance to joint limits (radians)' % self.accepted_dist_to_bounds)
        print('reach_bounds_penalty: %8.7f      # Reward penalty when reaching joint limit' % self.reach_bounds_penalty)
        print('fall_penalty: %8.7f           # Reward penalty for falling' % self.fall_penalty)
        print(
            f'rew_scaling: %{self.rew_scaling}            # Constant for scaling the normalised reward, if set this will use normalised reward instead of base reward')
        print('yaw_factor_power: %8.7f          # Yaw power for penalty when HYQ yaw is not within range' % self.yaw_factor_power)
        print('pitch_factor_power: %8.7f          # pitch power for penalty when HYQ yaw is not within range' % self.pitch_factor_power)
        print('roll_factor_power: %8.7f          # Roll power for penalty when HYQ yaw is not within range' % self.roll_factor_power)
        print('height_factor_power: %8.7f          # height power for penalty when HYQ yaw is not within range' % self.height_factor_power)
        print('y_penalty_factor: %8.7f          # y penalty when there is change in y value' % self.y_penalty_factor)
        print(f'-------------------------------------------------------------------------------------')

        assert self.accepted_dist_to_bounds >= 0.0, 'Allowable distance to joint limits should be positive'
        assert self.fall_penalty >= 0.0, 'Contact penalty should be positive'
        assert self.reach_bounds_penalty >= 0.0, 'Reach bounds penalty should be positive'
        self._max_time_step = max_time_step

        obs = self.robot.get_observations()
        self.initial_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        self.previous_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])

        self.__reset_count: int = 0
        self.__reach_count: int = 0
        self.__ep_step_num: int = 0
        self.__total_step_num: int = 0

        self.__robot_obs_space = self.robot.get_observation_space()

    def is_done(self, obs: HyQObservation) -> Tuple[bool, Dict]:
        failed, state = self.__is_failed(obs)
        info_dict = {'state': state}

        if state != HyQState.Undefined:  # Undefined is basically not reaching any of the failure conditions
            return failed, info_dict

        info_dict['state'] = HyQState.InProgress
        return False, info_dict

    def compute_reward(self, obs: HyQObservation, state: HyQState, *args) -> Tuple[float, Dict]:

        current_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        reward_info = {'current_coords': current_coords}
        assert state != HyQState.Undefined, f'State cannot be undefined, please check logic'

        reward = self.__calc_forward_reward(self.previous_coords, current_coords)
        reward_info["x_dist_reward"] = reward

        
        # Scale up normalised reward slightly such that the total reward is between 0 and 10 by default instead of between 0 and 1
        if self.rew_scaling is not None:
            reward *= self.rew_scaling
            reward_info["scaled_reward"] = reward

        self.previous_coords = current_coords

        # Orientation reward
        orientation_reward = self.__calc_orientation_reward(obs.pose, reward_info, self)
        reward += (orientation_reward/7)
        reward_info["orientation_add_reward"] = orientation_reward/7

        # Check if it has approached any joint limits
        if state == HyQState.ApproachJointLimits:
            reward -= self.reach_bounds_penalty

        # Check for collision
        if state == HyQState.Fallen:
            reward -= self.fall_penalty
                
        reward -= (0.5*(abs(obs.angular_velocity.x)+abs(obs.angular_velocity.y)+abs(obs.angular_velocity.z)))
        reward_info["minus_angular_reward"] = (0.5*(abs(obs.angular_velocity.x)+abs(obs.angular_velocity.y)+abs(obs.angular_velocity.z)))

        return reward, reward_info

    def set_action(self, action: numpy.ndarray) -> Dict:
        b = 8

        '''true_action = numpy.zeros((12,))
        true_action[0] = 0+ action[0]
        true_action[1] = (0.45 * math.sin((self.__ep_step_num + 12.57) / b) + 0.45) + action[1]
        true_action[2] = (0.3 * math.sin((self.__ep_step_num + 12.57) / b) - 1.1) + action[2]
        true_action[3] = 0+ action[3]
        true_action[4] = (-0.45 * math.sin((self.__ep_step_num + 12.57) / b) - 0.45) + action[4]
        true_action[5] = (0.3 * math.sin((self.__ep_step_num + 12.57) / b) + 1.1) + action[5]
        true_action[6] = 0+ action[6]
        true_action[7] = (-0.45 * math.sin((self.__ep_step_num + 12.57) / b) + 0.45) + action[7]
        true_action[8] = (-0.3 * math.sin((self.__ep_step_num + 12.57) / b) - 1.1) + action[8]
        true_action[9] = 0+ action[9]
        true_action[10] = (0.45 * math.sin((self.__ep_step_num + 12.57) / b) - 0.45) + action[10]
        true_action[11] = (-0.3 * math.sin((self.__ep_step_num + 12.57) / b) + 1.1) + action[11]'''

        true_action = numpy.zeros((12,))
        true_action[0] = 0+ action[0]
        true_action[1] = (0.45 * math.sin((self.__ep_step_num + 12.57) / b) + 0.45) #+ action[1]
        true_action[2] = (0.3 * math.sin((self.__ep_step_num + 12.57) / b) - 1.1) + action[1]
        true_action[3] = 0+ action[2]
        true_action[4] = (-0.45 * math.sin((self.__ep_step_num + 12.57) / b) - 0.45) #+ action[4]
        true_action[5] = (0.3 * math.sin((self.__ep_step_num + 12.57) / b) + 1.1) + action[3]
        true_action[6] = 0+ action[4]
        true_action[7] = (-0.45 * math.sin((self.__ep_step_num + 12.57) / b) + 0.45) #+ action[7]
        true_action[8] = (-0.3 * math.sin((self.__ep_step_num + 12.57) / b) - 1.1) + action[5]
        true_action[9] = 0+ action[6]
        true_action[10] = (0.45 * math.sin((self.__ep_step_num + 12.57) / b) - 0.45) #+ action[10]
        true_action[11] = (-0.3 * math.sin((self.__ep_step_num + 12.57) / b) + 1.1) + action[7]
        self.__ep_step_num += 1
        self.robot.set_action(true_action)
        action_info = {'agent_action': action}
        return action_info

    def reset(self):
        obs = self.robot.get_observations()
        self.initial_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        self.previous_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        self.__reset_count += 1
        self.__total_step_num += self.__ep_step_num
        self.__ep_step_num = 0

    @staticmethod
    def get_observations(obs_data_struct: HyQObservation, *args):
        np_obs = HyQSim.convert_obs_to_numpy(obs_data_struct)
        np_obs_new = numpy.delete(np_obs, [36,37])
        return np_obs_new, {}

    def get_observation_space(self):
        robot_obs_space: gym.spaces.Box = self.robot.get_observation_space()
        new_low = numpy.delete(robot_obs_space.low, [36,37])
        new_high = numpy.delete(robot_obs_space.high, [36,37])
        return Box(new_low, new_high)

    def get_action_space(self):
        return Box(-0.2, 0.2, (8,))

    def __is_failed(self, obs: HyQObservation) -> Tuple[bool, HyQState]:
        info_dict = {'state': HyQState.Undefined}

        # Check if time step exceeds limits, i.e. timed out
        # Time step starts from 1, that means if we only want to run 2 steps time_step will be 1,2
        if self.__ep_step_num >= self._max_time_step:
            return True, HyQState.Timeout

        # Check that joint values are not approaching limits
        joint_angles = numpy.array(obs.joint_positions)
        upper_bound = self.__robot_obs_space.high[:12]  # First 12 values are the joint angles
        lower_bound = self.__robot_obs_space.low[:12]
        min_dist_to_upper_bound = numpy.amin(abs(joint_angles - upper_bound))
        min_dist_to_lower_bound = numpy.amin(abs(joint_angles - lower_bound))
        # self.accepted_dist_to_bounds is basically how close to the joint limits can the joints go,
        # i.e. limit of 1.57 with accepted dist of 0.1, then the joint can only go until 1.47
        lower_limits_reached = min_dist_to_lower_bound < self.accepted_dist_to_bounds
        upper_limits_reached = min_dist_to_upper_bound < self.accepted_dist_to_bounds
        if lower_limits_reached or upper_limits_reached:
            info_dict['state'] = HyQState.ApproachJointLimits
            if lower_limits_reached:
                min_dist_lower_index = numpy.argmin(abs(joint_angles - lower_bound))
                # print(f"Joint with index {min_dist_lower_index} approached lower joint limits, current value: {joint_angles[min_dist_lower_index]}")
            else:
                min_dist_upper_index = numpy.argmin(abs(joint_angles - upper_bound))
                # print(f"Joint with index {min_dist_upper_index} approached upper joint limits, current value: {joint_angles[min_dist_upper_index]}")

            return False, HyQState.ApproachJointLimits

        if obs.trunk_contact:
            return True, HyQState.Fallen

        # Didn't fail
        return False, HyQState.Undefined

    def __calc_forward_reward(self, coords_init: numpy.ndarray,
                           coords_next: numpy.ndarray) -> float:
        x = coords_next[0] - coords_init[0]
        y = abs(coords_next[1] - coords_init[1]) #check any changes on y

        return x - self.y_penalty_factor*y 

    """
    Calculates the reward penalties based on orientation and height of the robot

    :param pose: The pose of the robot, expects a Pose object of HyQPy.Pose
    :param reward_info: Dictionary of the reward info, since python accepts dictionaries by reference we just directly modify this
    :returns: Total penalty scaling, a multiple of both the orientation and height penalty
    """
    @staticmethod
    def __calc_orientation_reward(pose, reward_info: Dict, self) -> float:
        q = pose.rotation
        [roll, pitch, yaw] = quaternion_to_euler(q.w, q.x, q.y, q.z)
        reward_info['roll_x'] = roll
        reward_info['pitch_y'] = pitch
        reward_info['yaw_z'] = yaw

        #print('roll',roll,'pitch', pitch,'yaw', yaw)
        # ----- Orientation penalty -----
        # For orientation, we allow 2.5 degrees each side for allowance, then start penalising after
        allowable_yaw_deg = 0.0  #rotation on z axis help to left or right
        allowable_yaw_rad = allowable_yaw_deg * math.pi / 180
        # Note: This logic doesn't work well when yaw is beyond 90 degrees, because roll and pitch will flip sign and yaw will still be less than 90
        if abs(yaw) > allowable_yaw_rad:
            yaw_penalty_factor = math.cos(yaw)
        else:
            yaw_penalty_factor = 1.0

        allowable_roll_deg = 0.0    #rotation on x axis #body balancing , need with haa joint to keep body left right balancing 
        allowable_roll_rad = allowable_roll_deg * math.pi / 180
        if abs(roll) > allowable_roll_rad:
            roll_penalty_factor = math.cos(roll)
        else:
            roll_penalty_factor = 1.0

        # Pitch penalty
        allowable_pitch_deg = 0.0#2.5  #rotation on y axis #need leg force to help balancing
        allowable_pitch_rad = allowable_pitch_deg * math.pi / 180
        # Note: This logic doesn't work well when yaw is beyond 90 degrees, because roll and pitch will flip sign and yaw will still be less than 90
        if abs(pitch) > allowable_pitch_rad:
            pitch_penalty_factor = math.cos(pitch)
            # pitch_penalty_factor = 0.7
        else:
            pitch_penalty_factor = 1.0
        # ----- Height Penalty -----
        # For height, we do not penalise for height between 0.42 and 0.47 (spawn height is 0.47 then dropped to 0.445 at steady state during ep start)
        current_height = pose.position.z
        if 0.54 < current_height < 0.69:
            height_penalty_factor = 1.0
        else:
            height_diff = abs(0.6 - current_height)
            height_penalty_factor = math.exp(height_diff * -2)

        base_reward=1

        yaw_penalty_factor = yaw_penalty_factor **  self.yaw_factor_power     #TODO to furthur imporve for walk forward
        yaw_reward = 2*base_reward * yaw_penalty_factor
        reward_info['yaw_reward'] = yaw_reward

        pitch_penalty_factor = pitch_penalty_factor **  self.pitch_factor_power   #TODO to furthur imporve for walk forward
        pitch_reward = base_reward * pitch_penalty_factor
        reward_info['pitch_reward'] = pitch_reward

        roll_penalty_factor = roll_penalty_factor ** self.roll_factor_power
        roll_reward = 2*base_reward * roll_penalty_factor
        reward_info['roll_penalty_factor'] = roll_penalty_factor

        height_penalty_factor = height_penalty_factor ** self.height_factor_power
        height_reward = 2*base_reward * height_penalty_factor
        reward_info['height_reward'] = height_reward
        return yaw_reward + pitch_reward + height_reward + roll_reward

