from typing import Dict, Tuple
import numpy
import math
from HyQPy import HyQObservation
from gym_robo.utils import quaternion_to_euler
from gym_robo.robots import HyQSim
from gym_robo.tasks import HyQState
from gym.spaces import Box
#velocity

class MovingAverageBuffer:
    def __init__(self, size=100):
        self.size = size
        self.index = 0
        self.buffer = numpy.zeros(size)
    def put(self,data):
        self.buffer[self.index % self.size]=data
        self.index += 1
    
    def get_average(self):
        if self.index < self.size:
            return sum(self.buffer) / self.index
        else:
            return sum(self.buffer) / self.size
    
    def clear(self):
        self.index = 0
        self.buffer = numpy.zeros(self.size)


class HyQTask:
    def __init__(self, robot: HyQSim, max_time_step: int = 1000,
                 accepted_dist_to_bounds=0.01, reach_bounds_penalty=0.0, fall_penalty=0.0,
                 rew_scaling=None,action_space_range=0.1,x_direction_value : float =0.0,y_direction_value: float =0.0,
                 yaw_factor_power=1.0,roll_factor_power=1.0,height_factor_power=1.0,pitch_factor_power=1.0):
        self.robot = robot
        self._max_time_step = max_time_step
        self.accepted_dist_to_bounds = accepted_dist_to_bounds
        self.reach_bounds_penalty = reach_bounds_penalty
        self.fall_penalty = fall_penalty
        self.rew_scaling = rew_scaling
        self.action_space_range = action_space_range
        self.x_direction_value=x_direction_value
        self.y_direction_value=y_direction_value
        self.x_sma_buffer = MovingAverageBuffer(100)
        self.y_sma_buffer = MovingAverageBuffer(100)
        self.yaw_factor_power = yaw_factor_power
        self.roll_factor_power = roll_factor_power
        self.height_factor_power = height_factor_power
        self.pitch_factor_power = pitch_factor_power
        #self.x_list=[0.2,0.4,0.6,0.8,1.0]
        #self.y_list=[0.2,0.4,0.6,0.8,1.0]
        print(f'-------------------------------Setting task parameters-------------------------------')
        print('max_time_step: %8d               # Maximum time step before stopping the episode' % self._max_time_step)
        print('accepted_dist_to_bounds: %8.7f    # Allowable distance to joint limits (radians)' % self.accepted_dist_to_bounds)
        print('reach_bounds_penalty: %8.7f      # Reward penalty when reaching joint limit' % self.reach_bounds_penalty)
        print('fall_penalty: %8.7f           # Reward penalty for falling' % self.fall_penalty)
        print(
            f'rew_scaling: %{self.rew_scaling}            # Constant for scaling the normalised reward, if set this will use normalised reward instead of base reward')
        print(f'action_space_range: {self.action_space_range}     # Scale of the action space')
        print(f'x_direction_value: {self.x_direction_value}     # desired x speed value, m/s')
        print(f'y_direction_value: {self.y_direction_value}     # desired y speed value, m/s')
        print('yaw_factor_power: %8.7f          # Yaw power for penalty when HYQ yaw is not within range' % self.yaw_factor_power)
        print('pitch_factor_power: %8.7f          # pitch power for penalty when HYQ yaw is not within range' % self.pitch_factor_power)
        print('roll_factor_power: %8.7f          # Roll power for penalty when HYQ yaw is not within range' % self.roll_factor_power)
        print('height_factor_power: %8.7f          # height power for penalty when HYQ yaw is not within range' % self.height_factor_power)
        print(f'-------------------------------------------------------------------------------------')

        assert self.accepted_dist_to_bounds >= 0.0, 'Allowable distance to joint limits should be positive'
        assert self.fall_penalty >= 0.0, 'Contact penalty should be positive'
        assert self.reach_bounds_penalty >= 0.0, 'Reach bounds penalty should be positive'
        assert -3.0<=self.x_direction_value <= 3.0, 'x_direction, expected to be within -3.0 to 3.0, m/s'
        assert -3.0<=self.y_direction_value <= 3.0, 'y_direction, expected to be within -3.0 to 3.0, m/s'

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
        assert state != HyQState.Undefined, f'State cannot be undefined, please check logic'
        #continous velocity change is secondary

        x_speed = self.__calc_x_per_timestep_changes(self.previous_coords, current_coords) 
        y_speed = self.__calc_y_per_timestep_changes(self.previous_coords, current_coords)
        different_coord= current_coords - self.previous_coords
        
        target_x_velocity = self.x_direction_value * 0.01
        target_y_velocity = self.y_direction_value * 0.01

        self.x_sma_buffer.put(x_speed)
        self.y_sma_buffer.put(y_speed)
        
        self.x_average_velocity, self.y_average_velocity= self.x_sma_buffer.get_average(), self.y_sma_buffer.get_average()
        x_speed_penalty= 0.007*abs((50-50**abs(self.x_average_velocity/target_x_velocity)))
        y_speed_penalty= 3*abs(self.y_average_velocity-target_y_velocity)

        reward =1-x_speed_penalty-y_speed_penalty

        reward_info = {'current_coords': current_coords,
                       'reward_after_minus_penalty' : reward,
                       'different_coord':different_coord,
                       'x_average_velocity' : self.x_average_velocity, 
                       'y_average_velocity' : self.y_average_velocity}  
        reward_info["x_speed_penalty"] = x_speed_penalty
        reward_info["y_speed_penalty"] = y_speed_penalty

        # Scaling reward penalties
        # orientation_penalty_factor = self.__calc_orientation_reward(obs.pose, reward_info)
        #reward -=( 0.7 -(1*orientation_penalty_factor))#orientation_penalty_factor best position is 1
        #reward_info["orientation_penalty_factor"] = orientation_penalty_factor
        orientation_reward = self.__calc_orientation_reward(obs.pose, reward_info, self)
        reward += (orientation_reward/7)
        reward_info["orientation_reward"] = orientation_reward/7

        if self.rew_scaling is not None:
            reward *= self.rew_scaling
            reward_info["scaled_reward_after_orientation"] = reward

        self.previous_coords = current_coords

        # Check if it has approached any joint limits
        if state == HyQState.ApproachJointLimits:
            reward -= self.reach_bounds_penalty

        # Check for fall
        if state == HyQState.Fallen:
            reward -= self.fall_penalty

        reward -= (0.5*(abs(obs.angular_velocity.x)+abs(obs.angular_velocity.y)+abs(obs.angular_velocity.z)))
        reward_info["angular_velocity_penalty"] = (0.5*(abs(obs.angular_velocity.x)+abs(obs.angular_velocity.y)+abs(obs.angular_velocity.z)))

        reward_info["end_reward_timestep"] = reward
        return reward, reward_info
    def set_action(self, action: numpy.ndarray) -> Dict:
        b =0.1831*(self.x_direction_value**4)-1.6117*(self.x_direction_value**3)+5.269*(self.x_direction_value**2)-9.888*self.x_direction_value+16.047
        true_action = numpy.zeros((12,))
        true_action[0] = 0  + action[0]
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
        true_action[11] = (-0.3 * math.sin((self.__ep_step_num + 12.57) / b) + 1.1) + action[11]
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
        self.x_sma_buffer.clear()
        self.y_sma_buffer.clear()

    def get_observations(self, obs_data_struct: HyQObservation, *args):
        np_obs = numpy.delete(HyQSim.convert_obs_to_numpy(obs_data_struct),[36,37])
        return numpy.append(np_obs,[self.x_direction_value,self.y_direction_value ,self.previous_coords[0],self.previous_coords[1]]), {}

    def get_observation_space(self):
        robot_obs_space: gym.spaces.Box = self.robot.get_observation_space()
        new_low = numpy.append(numpy.delete(robot_obs_space.low, [36,37]), [-1.5, -1.5,-50.0 ,-50.0]) #[x  ,y ,previous coordinate/one timestep=velocity (add velocity obs)]
        new_high = numpy.append(numpy.delete(robot_obs_space.high, [36,37]), [1.5, 1.5,50.0 ,50.0])  #[x ,y ,previous coordinate/one timestep =velocity (add velocity obs)]
        return Box(new_low, new_high)

    def get_action_space(self):
        return Box(-self.action_space_range, self.action_space_range, (12,))

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

    def __calc_x_per_timestep_changes(self, coords_init: numpy.ndarray,
                           coords_next: numpy.ndarray) -> float:
        x = coords_next[0] - coords_init[0]
        return x 

    def __calc_y_per_timestep_changes(self, coords_init: numpy.ndarray,
                           coords_next: numpy.ndarray) -> float:
        y = coords_next[1] - coords_init[1]  
        return y  

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
        # ----- Orientation penalty -----
        # For orientation, we allow 2.5 degrees each side for allowance, then start penalising after
        allowable_yaw_deg = 0.0
        allowable_yaw_rad = allowable_yaw_deg * math.pi / 180
        # Note: This logic doesn't work well when yaw is beyond 90 degrees, because roll and pitch will flip sign and yaw will still be less than 90
        if abs(yaw) > allowable_yaw_rad:
            yaw_penalty_factor = math.cos(yaw)
        else:
            yaw_penalty_factor = 1.0

        allowable_roll_deg = 0.0
        allowable_roll_rad = allowable_roll_deg * math.pi / 180
        if abs(roll) > allowable_roll_rad:
            roll_penalty_factor = math.cos(roll)
        else:
            roll_penalty_factor = 1.0
        
        # ----- Pitch penalty -----
        allowable_pitch_deg = 0.0
        allowable_pitch_rad = allowable_pitch_deg * math.pi / 180
        # Note: This logic doesn't work well when yaw is beyond 90 degrees, because roll and pitch will flip sign and yaw will still be less than 90
        if abs(pitch) > allowable_pitch_rad:
            pitch_penalty_factor = math.cos(pitch)
        else:
            pitch_penalty_factor = 1.0

        # ----- Height Penalty -----
        # For height, we do not penalise for height between 0.42 and 0.47 (spawn height is 0.47 then dropped to 0.445 at steady state during ep start)
        penalty_scale_height = 1.0
        current_height = pose.position.z
        if 0.56 < current_height < 0.69:
            height_penalty_factor = 1.0
        else:
            height_diff = abs(0.6 - current_height)
            height_penalty_factor = math.exp(height_diff * -2)

        base_reward=1

        yaw_penalty_factor = yaw_penalty_factor **  self.yaw_factor_power     #TODO to furthur imporve for walk forward
        yaw_reward = 2*base_reward * yaw_penalty_factor

        pitch_penalty_factor = pitch_penalty_factor **  self.pitch_factor_power   #TODO to furthur imporve for walk forward
        pitch_reward = base_reward * pitch_penalty_factor

        roll_penalty_factor = roll_penalty_factor ** self.roll_factor_power
        roll_reward = 2*base_reward * roll_penalty_factor

        height_penalty_factor = height_penalty_factor ** self.height_factor_power
        height_reward = 2*base_reward * height_penalty_factor
        return yaw_reward + pitch_reward + height_reward + roll_reward

