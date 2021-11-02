import numpy as np
import json
import sqlite3
from enum import Enum, auto
from matplotlib import pyplot as plt
from gym_robo.envs import HyQEnv
from typing import List
import gym
import os
import sys
import time
import pickle
from packaging.version import parse as parse_version

class HyQState(Enum):
    Reached = auto()
    InProgress = auto()
    ApproachJointLimits = auto()
    Fallen = auto()
    Timeout = auto()
    OutOfBounds = auto()
    Undefined = auto()


def adapt_np_array(arr: np.ndarray):
    return json.dumps(arr.tolist())


def convert_np_array(text):
    return np.array(json.loads(text))


def adapt_state(state):
    return str(state.name)


def convert_state(state):
    key = state.decode('utf-8')
    return HyQState[key]


def get_all_tables_ep_num(conn: sqlite3.Connection):
    cur = conn.cursor()

    get_tables_sql = \
        '''
        SELECT name FROM sqlite_master WHERE type='table'
        ORDER BY name DESC LIMIT 3;
        '''
    cur.execute(get_tables_sql)
    table_names = cur.fetchall()
    table_names = [a for a, in table_names]  # Make it a list of string instead of list of tuple
    episode_nums = []
    for table_name in table_names:
        sql_statement = \
            f''' 
            SELECT episode_num
            from {table_name} ORDER BY id DESC LIMIT 1
            '''
        cur.execute(sql_statement)
        data = cur.fetchall()
        table_ep_num, = data[0]
        episode_nums.append(table_ep_num)
    assert len(episode_nums) == len(table_names), "episode_nums and table_names should have same number of elements, something went wrong"
    return table_names, episode_nums


def get_dist_travelled_for_each_episode(conn: sqlite3.Connection, table_name: str, flag=False):
    sqlite3_ver = parse_version(sqlite3.sqlite_version)
    required_ver = parse_version('3.25.0')
    if sqlite3_ver < required_ver:
        raise RuntimeError(f"Python sqlite adapter targets old version of sqlite3, needs to target >= 3.25, current target: {sqlite3_ver}")
    cur = conn.cursor()
    if flag:
        id_cond = "WHERE id<3600000"
    else:
        id_cond = ""
    sqlite_statement = \
        f''' 
        SELECT
            episode_num, current_coords, step_num
        FROM (
            SELECT
                episode_num, current_coords, step_num,
                ROW_NUMBER() OVER (
                    PARTITION BY episode_num
                    ORDER BY id DESC
                ) RowNum
            FROM
                {table_name} {id_cond})
        WHERE
            RowNum = 1'''
    cur.execute(sqlite_statement)
    data = cur.fetchall()
    data2 = [*zip(*data)]

    coords = data2[1]
    x_dist = [x[0] for x in coords]
    data2[1] = x_dist

    return tuple(data2)

def get_highest_rew_for_each_episode(conn: sqlite3.Connection, table_name: str):
    """
    This function is to be used for plotting reward vs episode num graph.
    Should be called after determining the correct table to query,
    using get_all_tables_ep_num and some additional logic
    :param conn: Sqlite database connection object
    :param table_name: Name of table to be queried
    :return: max(cum_reward), episode_num, step_num
    """
    cur = conn.cursor()
    sqlite_statement = \
        f''' 
        SELECT MAX(cum_reward), episode_num, step_num FROM {table_name} GROUP BY episode_num;
        '''
    cur.execute(sqlite_statement)
    data = cur.fetchall()
    data2 = [*zip(*data)]
    return tuple(data2)


def get_last_rew_for_each_episode(conn: sqlite3.Connection, table_name: str, flag=False):
    """
    This function is to be used for plotting reward vs episode num graph.
    Should be called after determining the correct table to query,
    using get_all_tables_ep_num and some additional logic
    :param conn: Sqlite database connection object
    :param table_name: Name of table to be queried
    :return: cum_reward, episode_num, step_num
    """
    sqlite3_ver = parse_version(sqlite3.sqlite_version)
    required_ver = parse_version('3.25.0')
    if sqlite3_ver < required_ver:
        raise RuntimeError(f"Python sqlite adapter targets old version of sqlite3, needs to target >= 3.25, current target: {sqlite3_ver}")
    cur = conn.cursor()
    if flag:
        id_cond = "WHERE id<3690000"
    else:
        id_cond = ""
    sqlite_statement = \
        f'''SELECT
            cum_reward, episode_num, step_num
        FROM (
            SELECT
                cum_reward, episode_num, step_num,
                ROW_NUMBER() OVER (
                    PARTITION BY episode_num
                    ORDER BY id DESC
                ) RowNum
            FROM
                {table_name} {id_cond})
        WHERE
            RowNum = 1;
        '''
    cur.execute(sqlite_statement)
    data = cur.fetchall()
    data2 = [*zip(*data)]
    return tuple(data2)


def get_data_for_episode(conn: sqlite3.Connection, table_name: str, ep_num: int):
    """
    :param conn: Sqlite3 connection object
    :param table_name: Name of table to be queried
    :param ep_num: Episode to get data from
    :return: action, state, reward, step_num, joint_pos, joint_vel, cum_reward
    """

    cur = conn.cursor()
    sqlite_statement = \
        f''' 
            SELECT action, state, reward, step_num, joint_pos, joint_vel, cum_reward
            FROM {table_name} where episode_num == {ep_num}
        '''

    cur.execute(sqlite_statement)
    data = cur.fetchall()
    data2 = [*zip(*data)]
    return tuple(data2)


def get_min_dist_to_goal(conn: sqlite3.Connection, table_name: str):
    """
    :param conn: Sqlite3 connection object
    :param table_name: Name of table to be queried
    :return: min(dist_to_goal),reward,episode_num,cum_reward,target_coords,state
    """
    cur = conn.cursor()
    sqlite_statement = \
        f''' 
        select min(dist_to_goal),reward,episode_num,cum_reward,target_coords,state
        from {table_name}
            GROUP BY episode_num 
        '''
    cur.execute(sqlite_statement)
    data = cur.fetchall()
    data2 = [*zip(*data)]
    return tuple(data2)


def run_env_with_actions(actions, rewards, ori_cum_rews):
    robot_kwargs = {
        'use_gui': True,
        'rtf': 1.0,
        'control_mode': "Absolute"
    }
    task_kwargs = {
        'max_time_step': 1000,
        'reach_bounds_penalty': 10.0,
        'fall_penalty': 50.0,
        'boundary_radius': 0.5,
        'out_of_boundary_penalty': 100.0,
        'action_space_range': 0.1
    }

    env: HyQEnv = gym.make('HyQ-v4', enable_logging=False, task_kwargs=task_kwargs, robot_kwargs=robot_kwargs)
    env.reset()
    done = False
    count = 0
    cum_reward = 0
    while not done and count < len(actions):
        action = actions[count]
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        expected_reward = rewards[count]
        if abs(expected_reward - reward) > 0.0001:
            print(f"High deviation, {count}")
        count += 1
    ori_cum_rew = ori_cum_rews[-1]
    print(f'Ori: {ori_cum_rew}, Current: {cum_reward}')


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window  #
    smas = np.convolve(values, weights, 'valid')
    return smas


def plot_val_with_moving_average(val: np.ndarray, val_name: str, episode_num: np.ndarray, window: int):
    """
    Plot some value against episode number, can be max cumulative reward, final episode reward, etc.
    :param val: Array of values to be plotted as the y values
    :param val_name: Name of the value to be plotted, used as y-axis label
    :param episode_num: Array of episode numbers
    :param window: Window of moving average, 10 means average of 10 values
    :return: None
    """
    a = movingaverage(val, window)
    desired_num_elem = len(a)
    num_discard_elem = window // 2
    mov_avg_ep_num = episode_num[num_discard_elem:num_discard_elem+desired_num_elem]
    plt.plot(episode_num, val, 'r--', lw=0.4)
    plt.plot(mov_avg_ep_num, a, 'c', lw=1)
    plt.xlabel("Episode")
    plt.ylabel(val_name)
    plt.savefig(f'{val_name.lower().replace(" ", "_")}.png')
    plt.show()
    return


def plot_val_with_moving_average_multigraph(vals: List[np.ndarray], val_name: str, episode_nums: List[np.ndarray], labels: List[str], window: int):
    """
    Plot some value against episode number, can be max cumulative reward, final episode reward, etc.
    :param val: Array of values to be plotted as the y values
    :param val_name: Name of the value to be plotted, used as y-axis label
    :param episode_num: Array of episode numbers
    :param window: Window of moving average, 10 means average of 10 values
    :return: None
    """
    assert len(vals) == len(episode_nums) == len(labels), "Number of items of values, labels and episode numbers must match"
    colors = ['r', 'g', 'c']
    colors_dashed = ['r--', 'g--', 'c--']
    for i in range(len(vals)):
        avg = movingaverage(vals[i], window)
        desired_num_elem = len(avg)
        num_discard_elem = window // 2
        mov_avg_ep_num = episode_nums[i][num_discard_elem:num_discard_elem+desired_num_elem]
        # plt.plot(episode_nums[i], vals[i], colors_dashed[i], lw=0.4)
        plt.plot(mov_avg_ep_num, avg, colors[i], lw=1, label=labels[i])
    plt.xlabel("Episode")
    plt.ylabel(val_name)
    plt.legend()
    plt.savefig(f'{val_name.lower().replace(" ", "_")}.png')
    plt.show()

def main(argv):
    assert len(argv) == 1, f"Program requires 1 input argument representing the path of the sqlite db"
    hasDbFile = os.path.isfile(argv[0])
    assert hasDbFile, f"Given file does not exist: {argv[0]}"
    """ create a database connection to a SQLite database """
    conn = None
    conn2 = None
    try:
        conn = sqlite3.connect(argv[0], detect_types=sqlite3.PARSE_DECLTYPES)
        print(f'Using sqlite3, version: {sqlite3.sqlite_version}')
    except sqlite3.Error as e:
        print(e)
        return

    sqlite3.register_adapter(np.ndarray, adapt_np_array)
    sqlite3.register_converter("np_array", convert_np_array)
    sqlite3.register_adapter(HyQState, adapt_state)
    sqlite3.register_converter("state", convert_state)

    start = time.time()
    table_names, episode_nums = get_all_tables_ep_num(conn)

    # table_names3, episode_nums3 = get_all_tables_ep_num(conn3)
    end1 = time.time()
    print(f"Time taken to get tables ep num: {(end1-start)/1000000000}s")

    # Perform some logic here to get the desired table name to proceed with querying later
    # In this example case the logic is simply getting the table with highest episode count
    highest_num_ep_table_index = episode_nums.index(max(episode_nums))
    highest_num_ep_table = table_names[highest_num_ep_table_index]

    #highest_num_ep_table_index2 = episode_nums2.index(max(episode_nums2))
    #highest_num_ep_table2 = table_names2[highest_num_ep_table_index2]

    #highest_num_ep_table_index3 = episode_nums3.index(max(episode_nums3))
    #highest_num_ep_table3 = table_names3[highest_num_ep_table_index3]

    start2 = time.time()
    ep_num, x_dist, step_num = get_dist_travelled_for_each_episode(conn, highest_num_ep_table)
    #ep_num2, x_dist2, step_num2 = get_dist_travelled_for_each_episode(conn2, highest_num_ep_table2)
    #ep_num3, x_dist3, step_num3 = get_dist_travelled_for_each_episode(conn3, highest_num_ep_table3, True)
    # cum_rews, ep_num_list, step_num = get_highest_rew_for_each_episode(conn, highest_num_ep_table)
    end2 = time.time()
    #plot_val_with_moving_average_multigraph([x_dist], "Horizontal Distance Travelled",
    #                                        [ep_num], ["Frank"], 10)

    print(f"Time taken to get dist travelled for each ep: {(end2-start2)/1000000000}s")

    start4 = time.time()
    cum_rews, ep_num_list, step_num = get_last_rew_for_each_episode(conn, highest_num_ep_table)
    #cum_rews2, ep_num_list2, step_num2 = get_last_rew_for_each_episode(conn2, highest_num_ep_table2)
    #cum_rews3, ep_num_list3, step_num3 = get_last_rew_for_each_episode(conn3, highest_num_ep_table3, True)
    end4 = time.time()
    print(f"Time taken to get last rew for each ep: {(end4-start4)/1000000000}s")

    # plot_val_with_moving_average(cum_rews, "Max Cumulative Reward", ep_num_list, 10)
    # plot_val_with_moving_average(cum_rews2, "Cumulative Reward", ep_num_list2, 10)
    plot_val_with_moving_average_multigraph([cum_rews], "Cumulative Reward",
                                            [ep_num_list], ["Frank"], 10)
    # data_ep_num = 5000
    # start3 = time.time()
    # action, state, reward, step_num, joint_pos, joint_vel, cum_reward = get_data_for_episode(conn, highest_num_ep_table, data_ep_num)
    # end3 = time.time()
    #
    # print(f"Time taken to get data for ep {data_ep_num}: {(end3-start3)/1000000000}s")
    # run_env_with_actions(action, reward, cum_reward)
    # print("end")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        db_path = input("Input database path: ")
        main([db_path])
    else:
        main(sys.argv[1:])
