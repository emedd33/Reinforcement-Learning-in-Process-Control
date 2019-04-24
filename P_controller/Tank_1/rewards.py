def sum_rewards(states, terminated, get_reward):
    rewards = []
    for i in range(len(states)):
        rewards.append(get_reward([states[i] / 10], terminated[0]))
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
        return -1
    if state[0] > 0.4 and state[0] < 0.6:
        return 1
    return 0


# def get_reward_ABS(state, terminated):
#     "Calculates the environments reward for the next state"

#     if terminated:
#         return -10
#     return np.absolute(ss_position - state[0])


def get_reward_SSE(state, terminated):
    "Calculates the environments reward for the next state"

    if terminated:
        return -10
    return -(0.5 - state[0]) ** 2
