import numpy as np
from params import TANK_PARAMS

ss_position = TANK_PARAMS["init_level"]


def get_reward_1(h, terminated):  # 200
    "Calculates the environments reward for the next state"

    if terminated:
        return -10
    if h > 0.25 and h < 0.75:
        return 1
    return 0


def get_reward_2(h, terminated):  # Max 12
    "Calculates the environments reward for the next state"

    if terminated:
        return -10
    if h > 0.45 and h < 0.55:
        return 1
    return 0


def get_reward_3(h, terminated):  # 500
    "Calculates the environments reward for the next state"
    if terminated:
        return -10
    if h > 0.45 and h < 0.55:
        return 5
    if h > 0.4 and h < 0.6:
        return 3
    if h > 0.3 and h < 0.7:
        return 2
    if h > 0.2 and h < 0.8:
        return 1
    return 0


def get_reward_ABS(h, terminated):
    "Calculates the environments reward for the next state"

    if terminated:
        return -10
    return np.absolute(ss_position - h)


def get_reward_SSE(h, terminated):
    "Calculates the environments reward for the next state"

    if terminated:
        return -10
    return (ss_position - h) ** 2
