import time
import numpy as np
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.datasets.utils import load_data


def evaluate_2armed_bandit(dataset='credit', algo='liblinear_svc', time_limit=1200):
    raw_data = load_data(dataset, datanode_returned=True)
    bandit = SecondLayerBandit(algo, raw_data)

    _start_time = time.time()
    stats = list()

    for iter in range(100000):
        res = bandit.play_once()
        stats.append([iter, time.time() - _start_time, res])

        if time.time() > time_limit + _start_time:
            break

    print(bandit.final_rewards)
    print(bandit.action_sequence)
    print(bandit.evaluation_cost['fe'])
    print(bandit.evaluation_cost['hpo'])
    print(np.mean(bandit.evaluation_cost['fe']))
    print(np.mean(bandit.evaluation_cost['hpo']))


if __name__ == "__main__":
    evaluate_2armed_bandit()
