import numpy as np
from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.datasets.utils import load_data

raw_data = load_data('pc4', datanode_returned=True)
bandit = SecondLayerBandit('liblinear_svc', raw_data)

rewards = list()
for iter in range(10):
    print('\n'*5)
    res = bandit.play_once()
    rewards.append(res)
    print(rewards)
    print('\n' * 5)

print(bandit.final_rewards)
print(bandit.action_sequence)
print(bandit.evaluation_cost['fe'])
print(bandit.evaluation_cost['hpo'])
print(np.mean(bandit.evaluation_cost['fe']))
print(np.mean(bandit.evaluation_cost['hpo']))
