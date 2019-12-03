from automlToolkit.bandits.first_layer_bandit import FirstLayerBandit
from automlToolkit.datasets.utils import load_data


def evaluate_1stlayer_bandit(dataset='credit'):
    raw_data = load_data(dataset, datanode_returned=True)
    # algorithms = ['lda', 'k_nearest_neighbors', 'libsvm_svc', 'sgd', 'adaboost', 'random_forest', 'extra_trees', 'gradient_boosting']
    algorithms = ['k_nearest_neighbors', 'libsvm_svc']
    trial_num = 80
    bandit = FirstLayerBandit(trial_num, algorithms, raw_data)

    bandit.optimize()

    print(bandit.final_rewards)
    print(bandit.action_sequence)


if __name__ == "__main__":
    evaluate_1stlayer_bandit()
