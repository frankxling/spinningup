#matplotlib notebook
import numpy as np
import json
import sqlite3
from enum import Enum, auto
from matplotlib import pyplot as plt
"/home/siwflhc/anaconda3/envs/arm_gpu/bin/python /home/siwflhc/spinningup/plothyq.py"
class ArmState(Enum):
    Reached = auto()
    InProgress = auto()
    ApproachJointLimits = auto()
    Collision = auto()
    Timeout = auto()
    Undefined = auto()


def adapt_np_array(arr: np.ndarray):
    return json.dumps(arr.tolist())


def convert_np_array(text):
    return np.array(json.loads(text))


def adapt_arm_state(state: ArmState):
    return str(state.name)


def convert_arm_state(state):
    key = state.decode('utf-8')
    return ArmState[key]


table_name = 'HyQv0testjo_b8_corrected_test' #modify place
""" create a database connection to a SQLite database """
conn = None
try:
    conn = sqlite3.connect('hyq_log2.db', detect_types=sqlite3.PARSE_DECLTYPES)  #modify place
    print(f'Using sqlite3, version: {sqlite3.sqlite_version}')
except sqlite3.Error as e:
    print(e)

sqlite3.register_adapter(np.ndarray, adapt_np_array)
sqlite3.register_converter("np_array", convert_np_array)
sqlite3.register_adapter(ArmState, adapt_arm_state)
sqlite3.register_converter("armstate", convert_arm_state)

cur = conn.cursor()
sql_statement = \
    ''' 
    select min(distance_to_goal),reward,episode_num,cum_reward,normalised_reward,state
    from HyQv0testjo_b8_corrected_test
        GROUP BY episode_num 
    '''  #condition is put at where not in "inprogree" or stpe is more or what...   from the saved table  
    #WHERE arm_state == 'Collision' or arm_state == 'ApproachJointLimits'

cur.execute(sql_statement)
data = cur.fetchall()
data2 = [*zip(*data)]
# data[0] is min(dist_to_goal)
# data[1] is reward
# data[2] is episode_num
# data[3] is cum_reward
# data[4] is normalised_reward
# data[5] is state


dist_to_goal = np.array(data2[0])
reward = np.array(data2[1])
episode_num = np.array(data2[2])
cum_reward = np.array(data2[3])
normalised_reward = np.array(data2[4])
def movingaverage(values,window):
    weights = np.repeat(1.0,window)/window   #
    smas =np.convolve(values,weights,'valid')
    return smas
a=movingaverage(dist_to_goal,500)

plt.figure(0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
plt.plot(episode_num,dist_to_goal,'r',  lw=1)
plt.title('dist to goal vs episode')
plt.figure(1) 
plt.plot(np.array(list(range(0,len(a)))),a,'r',  lw=2)
plt.title('MovingAverage dist_to_goal vs episode')


cur = conn.cursor()
sql_statement = \
    ''' 
    select min(distance_to_goal),reward,episode_num,cum_reward,normalised_reward,state,max(step_num)
    from HyQv0testjo_b8_corrected_test
        GROUP BY episode_num 
    '''  #condition is put at where not in "inprogree" or stpe is more or what...   from the saved table  
    #WHERE arm_state == 'Collision' or arm_state == 'ApproachJointLimits'

cur.execute(sql_statement)
data = cur.fetchall()
data2 = [*zip(*data)]

# data[0] is min(dist_to_goal)
# data[1] is reward
# data[2] is episode_num
# data[3] is cum_reward
# data[4] is normalised_reward
# data[5] is state
# data[5] is stepnum
episode_num = np.array(data2[2])
cum_reward = np.array(data2[3])

plt.figure(2)
plt.plot(episode_num,cum_reward,'r',  lw=2)
plt.title('cum_reward vs episode') 

# Good reference SQL statement to get first of every distinct
# This is used to get the first/tenth/whatever reward of every episode
# This requires python 3.7 and above, python 3.6 bundled sqlite client targets sqlite 3.22 which doesn't support window functions (sqlite>3.25)
# https://stackoverflow.com/questions/16847574/how-to-use-row-number-in-sqlite
sql_statement_rew_noise = \
'''SELECT
    rew_noise
FROM (
    SELECT
        rew_noise,
        ROW_NUMBER() OVER (
            PARTITION BY episode_num
            ORDER BY id
        ) RowNum
    FROM
        run_19_02_2020__14_36_56 )
WHERE
    RowNum = 10;
'''
'''  #find max dist to goal along with different episode_num
    select min(dist_to_goal),reward,episode_num,cum_normalised_reward,cum_reward,normalised_reward,arm_state, target_coords
    from run_23_04_2020__12_55_04
	WHERE arm_state != 'Reached'
		GROUP BY episode_num           # /home/siwflhc/anaconda3/envs/arm_gpu/bin/python /home/siwflhc/spinningup/spinup/plotSQL.py
'''  
'''  #find episode num timestep between a few episode
    select episode_num,step_num, current_coords
	from ignirtf0point25_set1run1
	WHERE episode_num <= 10
'''

# cur = conn.cursor()
# sql_statement = \
#     ''' 
#     select action
#     from run_09_06_2020__12_29_17
# 	WHERE episode_num=1
#     '''  #condition is put at where not in "inprogree" or stpe is more or what...   from the saved table  
#     #WHERE arm_state == 'Collision' or arm_state == 'ApproachJointLimits'

# cur.execute(sql_statement)
# data = cur.fetchall()
# data2 = [*zip(*data)]
# action = np.array(data2)
# np.save('actionepisode1.npy', action)



plt.show() 