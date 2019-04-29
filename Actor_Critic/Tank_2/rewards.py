import numpy as np
from params import TANK_PARAMS

ss_position = TANK_PARAMS[0]["init_level"]


def sum_rewards(states, terminated, get_reward):
    rewards = []
    for i in range(len(states)):
        rewards.append(get_reward(states[i], terminated[i]))
    return rewards


def get_reward_1(state, terminated):
    "Calculates the environments reward for the next state"

    if terminated:
        return -10
    if state[0] > 0.25 and state[0] < 0.75:
        return 1
    return 0


def get_reward_2(state, terminated):
    "Calculates the environments reward for the next state"
    if terminated:
        return -10
    if state[0] > 0.4 and state[0] < 0.6:
        return 1
    return 0


def get_reward_3(state, terminated):
    "Calculates the environments reward for the next state"
    if terminated:
        return -10
    if state[0] > 0.4 and state[0] < 0.6:
        return 1
    return 0.5


def get_reward_ABS(state, terminated):
    "Calculates the environments reward for the next state"

    if terminated:
        return -10
    return -np.absolute(ss_position - state[0])


def get_reward_SSE(state, terminated):
    "Calculates the environments reward for the next state"

    if terminated:
        return -10
    return -(ss_position - state[0]) ** 2
