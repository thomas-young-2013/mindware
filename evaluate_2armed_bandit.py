from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from automlToolkit.datasets.utils import load_data

raw_data = load_data('pc4', datanode_returned=True)
bandit = SecondLayerBandit('liblinear_svc', raw_data)

rewards = list()
for iter in range(20):
    print('\n'*5)
    res = bandit.play_once()
    rewards.append(res)
    print(rewards)
    print('\n' * 5)

print(bandit.final_rewards)
print(bandit.action_sequence)
