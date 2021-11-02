from typing import Dict, Tuple
import numpy
import math
from HyQPy import HyQObservation
from gym_robo.utils import quaternion_to_euler
from gym_robo.robots import HyQSim
from gym_robo.tasks import HyQState
from gym.spaces import Box

class HyQTask63:
    def __init__(self, robot: HyQSim, max_time_step: int = 1000,
                 accepted_dist_to_bounds=0.01, reach_bounds_penalty=0.0, fall_penalty=0.0,
                 boundary_radius: float = 0.5, out_of_boundary_penalty: float =0.0):
        self.robot = robot
        self._max_time_step = max_time_step
        self.accepted_dist_to_bounds = accepted_dist_to_bounds
        self.reach_bounds_penalty = reach_bounds_penalty
        self.fall_penalty = fall_penalty
        self.boundary_radius = boundary_radius
        self.out_of_boundary_penalty = out_of_boundary_penalty
        print(f'-------------------------------Setting task parameters-------------------------------')
        print('max_time_step: %8d               # Maximum time step before stopping the episode' % self._max_time_step)
        print('accepted_dist_to_bounds: %8.7f    # Allowable distance to joint limits (radians)' % self.accepted_dist_to_bounds)
        print('reach_bounds_penalty: %8.7f      # Reward penalty when reaching joint limit' % self.reach_bounds_penalty)
        print('fall_penalty: %8.7f           # Reward penalty for falling' % self.fall_penalty)
        print(f'boundary_radius: {self.boundary_radius}          # Radius of boundary that the robot is required to stay in')
        print(f'out_of_boundary_penalty: {self.out_of_boundary_penalty}     # Penalty given when the robot centre moved outside of the specified boundary')
        print(f'-------------------------------------------------------------------------------------')

        assert self.accepted_dist_to_bounds >= 0.0, 'Allowable distance to joint limits should be positive'
        assert self.fall_penalty >= 0.0, 'Contact penalty should be positive'
        assert self.reach_bounds_penalty >= 0.0, 'Reach bounds penalty should be positive'
        assert self.out_of_boundary_penalty >= 0.0, 'Out of boundary penalty should be positive'
        assert self.boundary_radius >= 0.01, 'Boundary radius too small, expected to be at least greater than 0.01'

        self._max_time_step = max_time_step

        obs = self.robot.get_observations()
        self.initial_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        self.previous_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        self.previous_action = numpy.zeros((12,))
        self.sum_action_deviation: float = 0.0
        self.__reset_count: int = 0
        self.__reach_count: int = 0
        self.__ep_step_num: int = 0
        self.__cycle_len: int = 0
        self.__gait_step: int = 0
        self.__gait_stepC: int = 0
        self.__total_step_num: int = 0
        self.c: int = 0
        self.a_real=0
        self.a_noise=0
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

        reward = 0.0
        reward_info = {'current_coords': current_coords}

        x = abs(current_coords[0] - self.previous_coords[0])
        y = abs(current_coords[1] - self.previous_coords[1])
        z = abs(current_coords[2] - self.previous_coords[2])
        reward = 1.2 - 300*x - 300*y - 300*z

        self.previous_coords = current_coords

        # Scaling reward penalties
        total_penalty_factor = self.__calc_rew_penalty_scale(obs.pose, reward_info)
        if reward > 0.0:
            reward *= total_penalty_factor
        reward_info['standing_reward'] = reward
        reward_info["a_real"] = self.a_real
        reward_info["a_noise"] = self.a_noise
        # Checking Gait
        self.__gait_step = (self.__ep_step_num - 1) % self.__cycle_len  # gait_step in % but at 100% timing might have a bit issues
        self.__gait_stepC = (self.__ep_step_num - 1) % (self.__cycle_len * 2)
        self.__gait_stepC = (self.__gait_stepC / (self.__cycle_len * 2)) * 100
        num_contact = int(obs.lf_foot_contact) + int(obs.lh_foot_contact) + int(obs.rf_foot_contact) + int(obs.rh_foot_contact)
        reward_for_correct_contact = 0.20
        if 0 <= self.__gait_stepC <= 10 or 50 < self.__gait_stepC <= 60:
            reward += ((reward_for_correct_contact * num_contact) - (0.005*sum(obs.applied_joint_energies)))
        elif 10 < self.__gait_stepC <= 50:
            if obs.lf_foot_contact: reward += (reward_for_correct_contact - 0.02*sum(obs.applied_joint_energies[0:3]))  # else: reward -=1 
            if obs.rh_foot_contact: reward += (reward_for_correct_contact - 0.02*sum(obs.applied_joint_energies[9:])) # else: reward -=1 
            if not obs.lh_foot_contact: reward += reward_for_correct_contact # else: reward -=1
            if not obs.rf_foot_contact: reward += reward_for_correct_contact # else: reward -=1
        elif 60 < self.__gait_stepC <= 100:
            if obs.lh_foot_contact: reward += (reward_for_correct_contact - 0.02*sum(obs.applied_joint_energies[3:6]))  # else: reward -=1
            if obs.rf_foot_contact: reward += (reward_for_correct_contact - 0.02*sum(obs.applied_joint_energies[6:9]))  # else: reward -=1
            if not obs.lf_foot_contact: reward += reward_for_correct_contact  # else: reward -=1
            if not obs.rh_foot_contact: reward += reward_for_correct_contact  # else: reward -=1

        # Check if it has approached any joint limits
        if state == HyQState.ApproachJointLimits:
            reward -= self.reach_bounds_penalty

        # Check if it went out of bounds
        if state == HyQState.OutOfBounds:
            reward -= self.out_of_boundary_penalty

        # Check for fall
        if state == HyQState.Fallen:
            reward -= self.fall_penalty

        act_dev = 0.4 * self.sum_action_deviation
        reward_info["act_dev"] = act_dev
        reward -= act_dev
        ang_vel = 0.5 * (abs(obs.angular_velocity.x) + abs(obs.angular_velocity.y) + abs(obs.angular_velocity.z))
        reward_info["ang_vel"] = ang_vel
        reward -= ang_vel
        reward_info["reward"] = reward
        return reward, reward_info
    
    def get_obs_noise_action(self, a_real, a_noise):
        action_deviation = abs(numpy.subtract(a_real, a_noise))
        self.a_real=a_real
        self.a_noise=a_noise
        self.sum_action_deviation = sum(action_deviation)

    def set_action(self, action: numpy.ndarray) -> None:
        b = 5
        true_action = numpy.zeros((12,))
        Ttrue_action = numpy.zeros((12,))
        true_action[0] = -0.14 + action[0]
        #true_action[0] = (0.10 * math.sin((self.__ep_step_num - (b * 0.5 * math.pi)) / b) - 0.10) + action[0]
        Ttrue_action[1] = 0.81611
        Ttrue_action[2] = -1.43081
        true_action[3] = -0.14 + action[3]
        #true_action[3] = (0.10 * math.sin((self.__ep_step_num - (b * 0.5 * math.pi)) / b) - 0.10) + action[3]
        Ttrue_action[4] = (-0.10 * math.sin((self.__ep_step_num - (b * 0.5 * math.pi)) / b) - 0.90)
        Ttrue_action[5] = (0.4 * math.sin((self.__ep_step_num - (b * 0.5 * math.pi)) / b) + 1.8)
        true_action[6] = -0.14 + action[6]
        #true_action[6] = (0.10 * math.sin((self.__ep_step_num - (b * 0.5 * math.pi)) / b) - 0.10) + action[6]
        Ttrue_action[7] = (0.10 * math.sin((self.__ep_step_num - (b * 0.5 * math.pi)) / b) + 0.90)
        Ttrue_action[8] = (-0.4 * math.sin((self.__ep_step_num - (b * 0.5 * math.pi)) / b) - 1.8)
        true_action[9] = -0.14 + action[9]
        #true_action[9] = (0.10 * math.sin((self.__ep_step_num - (b * 0.5 * math.pi)) / b) - 0.10) + action[9]
        Ttrue_action[10] = -0.82278
        Ttrue_action[11] = 1.453150
        if self.c % 2 == 0:
            true_action[1] = Ttrue_action[1] + action[1]
            true_action[2] = Ttrue_action[2] + action[2]
            true_action[4] = Ttrue_action[4] + action[4]
            true_action[5] = Ttrue_action[5] + action[5]
            true_action[7] = Ttrue_action[7] + action[7]
            true_action[8] = Ttrue_action[8] + action[8]
            true_action[10] = Ttrue_action[10] + action[10]
            true_action[11] = Ttrue_action[11] + action[11]
        else:
            true_action[1] = Ttrue_action[7] + action[1]
            true_action[2] = Ttrue_action[8] + action[2]
            true_action[4] = Ttrue_action[10] + action[4]
            true_action[5] = Ttrue_action[11] + action[5]
            true_action[7] = Ttrue_action[1] + action[7]
            true_action[8] = Ttrue_action[2] + action[8]
            true_action[10] = Ttrue_action[4] + action[10]
            true_action[11] = Ttrue_action[5] + action[11]
        #print (self.__gait_step1)
        if self.__gait_step == 0:
            self.c += 1
        self.__ep_step_num += 1
        self.__cycle_len = int(b * 2 * math.pi)
        self.robot.set_action(true_action)
        return {}

    def reset(self):
        obs = self.robot.get_observations()
        self.initial_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        self.previous_coords = numpy.array([obs.pose.position.x, obs.pose.position.y, obs.pose.position.z])
        self.sum_action_deviation = 0.0
        self.__reset_count += 1
        self.__total_step_num += self.__ep_step_num
        self.__ep_step_num = 0
        self.__gait_step = 0
        self.__gait_stepC = 0
        self.c = 0

    def get_observations(self, obs_data_struct: HyQObservation):
        np_obs = HyQSim.convert_obs_to_numpy(obs_data_struct)
        add_observation = numpy.append(self.previous_action, self.__gait_stepC)
        return numpy.append(np_obs, add_observation), {}

    def get_observation_space(self):
        robot_obs_space: gym.spaces.Box = self.robot.get_observation_space()
        add_low = [-0.2]*12 + [0.0]
        add_high = [0.2]*12 + [100.0]
        new_low = numpy.append(robot_obs_space.low, add_low)
        new_high = numpy.append(robot_obs_space.high, add_high)
        return Box(new_low, new_high)

    def get_action_space(self):
        return Box(-0.2, 0.2, (12,))

    def __is_failed(self, obs: HyQObservation) -> Tuple[bool, HyQState]:
        info_dict = {'state': HyQState.Undefined}

        # Check if time step exceeds limits, i.e. timed out
        # Time step starts from 1, that means if we only want to run 2 steps time_step will be 1,2
        if self.__ep_step_num >= self._max_time_step:
            return True, HyQState.Timeout

        if obs.trunk_contact:
            return True, HyQState.Fallen

        # Check for out of bounds
        current_x = obs.pose.position.x
        current_y = obs.pose.position.y
        current_coords_2d = numpy.array([current_x, current_y])
        #initial_coords_2d = numpy.array([0.0, 0.0])
        initial_coords_2d = numpy.array([self.initial_coords[0], self.initial_coords[1]])
        dist_from_origin = numpy.linalg.norm(current_coords_2d - initial_coords_2d)
        if dist_from_origin > self.boundary_radius:
            return True, HyQState.OutOfBounds

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

        # Didn't fail
        return False, HyQState.Undefined

    """
    Calculates the reward penalties based on orientation and height of the robot

    :param pose: The pose of the robot, expects a Pose object of HyQPy.Pose
    :param reward_info: Dictionary of the reward info, since python accepts dictionaries by reference we just directly modify this
    :returns: Total penalty scaling, a multiple of both the orientation and height penalty
    """

    @staticmethod
    def __calc_rew_penalty_scale(pose, reward_info: Dict) -> float:
        q = pose.rotation
        [roll, pitch, yaw] = quaternion_to_euler(q.w, q.x, q.y, q.z)

        # ----- Orientation penalty -----
        # For orientation, we allow 2.5 degrees each side for allowance, then start penalising after
        allowable_yaw_deg = 0.1
        allowable_yaw_rad = allowable_yaw_deg * math.pi / 180
        # Note: This logic doesn't work well when yaw is beyond 90 degrees
        if abs(yaw) > allowable_yaw_rad:
            yaw_penalty_factor = math.cos(yaw)
        else:
            yaw_penalty_factor = 1.0

        allowable_roll_deg = 4.0
        allowable_roll_rad = allowable_roll_deg * math.pi / 180
        if abs(roll) > allowable_roll_rad:
            roll_penalty_factor = math.cos(roll)
        else:
            roll_penalty_factor = 1.0
        
        # ----- Pitch penalty -----
        allowable_pitch_deg = 2.3
        allowable_pitch_rad = allowable_pitch_deg * math.pi / 180
        # Note: This logic doesn't work well when yaw is beyond 90 degrees, because roll and pitch will flip sign and yaw will still be less than 90
        if abs(pitch) > allowable_pitch_rad:
            pitch_penalty_factor = math.cos(pitch)
        else:
            pitch_penalty_factor = 1.0

        yaw_penalty_factor = yaw_penalty_factor ** 17
        reward_info['yaw_penalty_factor'] = yaw_penalty_factor

        roll_penalty_factor = roll_penalty_factor ** 17
        reward_info['roll_penalty_factor'] = roll_penalty_factor

        pitch_penalty_factor = pitch_penalty_factor ** 11
        reward_info['pitch_penalty_factor'] = pitch_penalty_factor
        
        return pitch_penalty_factor * yaw_penalty_factor * roll_penalty_factor