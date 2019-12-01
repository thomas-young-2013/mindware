from automlToolkit.bandits.second_layer_bandit import SecondLayerBandit
from evaluate_transgraph import engineer_data

raw_data, _ = engineer_data('pc4', 'none')
bandit = SecondLayerBandit('liblinear_svc', raw_data)

rewards = list()
for iter in range(20):
    print('\n'*5)
    res = bandit.play_once()
    rewards.append(res)
    print(rewards)
    print('\n' * 5)
