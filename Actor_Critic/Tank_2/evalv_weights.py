from models.Agent import Agent
from evalv_params import AGENT_PARAMS
import matplotlib.pyplot as plt
import os

plt.style.use("ggplot")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    # ============= Initialize variables and objects ===========
    agent = Agent(AGENT_PARAMS)
    # ================= Running episodes =================#
    w = agent.networks[0].input.weight.data.numpy()
    w1 = w[:, 0]  # level
    w2 = w[:, 1]  # gradient
    w3 = w[:, 2]  # is above 0.5
    w4 = w[:, 3]  # previous valve position
    print(w)
    print(f"w1 {sum(w1)}")
    print(f"w2 {sum(w2)}")
    print(f"w3 {sum(w3)}")
    print(f"w4 {sum(w4)}")

    # Create bars
    plt.subplot(2, 2, 1)
    plt.bar(range(len(w1)), w1)
    plt.subplot(2, 2, 2)
    plt.bar(range(len(w2)), w2)
    plt.subplot(2, 2, 3)
    plt.bar(range(len(w3)), w3)
    plt.subplot(2, 2, 4)
    plt.bar(range(len(w4)), w4)

    # Show graphic
    plt.show()


if __name__ == "__main__":
    main()
