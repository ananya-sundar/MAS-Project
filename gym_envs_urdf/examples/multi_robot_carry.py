import time

import gym
import os
import numpy as np
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from MotionPlanningEnv.urdfObstacle import UrdfObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from urdfenvs.sensors.lidar import Lidar


lidar = Lidar(4, nb_rays=4, raw_data=False)
urdf_obstacle_dict = {
    "type": "urdf",
    "geometry": {"position": [0.2, -0.0, 1.05]},
    "urdf": os.path.join(os.path.dirname(__file__), "block.urdf"),
}
urdf_obstacle = UrdfObstacle(name="carry_object", content_dict=urdf_obstacle_dict)
goalDict = { "type": "static",
            "desired_position": [0.2, -0.0, 1.05],
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": 0,
            "child_link": 3,
            "epsilon": 0.2,
            "type": "staticSubGoal",
             }
goal = StaticSubGoal(name="goal", content_dict=goalDict)

def add_stuff(env):
    env.add_obstacle(urdf_obstacle)
    env.add_goal(goal)
    env.add_sensor(lidar, robot_ids=[0, 1])
    env.add_walls()



def create_obstacle():
    """
    You can place any arbitrary urdf file in here.
    The physics of this box is not perfect, but this is not the field of
    my expertise.
    """

    urdf_obstacle_dict = {
        "type": "urdf",
        "geometry": {"position": [0.2, -0.0, 1.05]},
        "urdf": os.path.join(os.path.dirname(__file__), "block.urdf"),
    }
    urdf_obstacle = UrdfObstacle(name="carry_object", content_dict=urdf_obstacle_dict)
    return urdf_obstacle


def make_env(render=False):
    robots = [
        GenericUrdfReacher(urdf="loadPointRobot.urdf", mode="vel"),
        GenericUrdfReacher(urdf="loadPointRobot.urdf", mode="vel"),
    ]
    env = gym.make("urdf-env-v0", dt=0.1, robots=robots, render=render)
    # Choosing arbitrary actions
    base_pos = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    env.reset(base_pos=base_pos)
    env.add_stuff()
    ob = env.get_observation()
    return env, ob
def kill_env(env):
    env.close()
    del env
def run_multi_robot_carry(n_steps=1000, render=False):
    # Two robots to carry the load

    history = []
    i = 0
    done = False
    env, ob = make_env(render=render)
    action = [-0.1, 0.1, 0.0, -0.1, 0.1, 0.0]

    for _ in range(n_steps):
        i += 1
        print("i ,", i)
        if done==True:
            kill_env(env)
            env, ob = make_env(render=render)
            print(ob)
            done = False
        else:
            # WARNING: The reward function is not defined for you case.
            # You will have to do this yourself.
            ob, reward, done, info = env.step(action)
            print(done)
        # if done:
        #     break
        history.append(ob)
    env.close()

    return history


if __name__ == "__main__":
    run_multi_robot_carry(render=True)
