import time
from ConfigSpace import ConfigurationSpace, Configuration
from autosklearn.pipeline.components.classification import _classifiers


def explore_config_nums(configuration_space: ConfigurationSpace, batch_size: int=1000) -> int:
    result = []
    cnt, sample_cnt = 0, 0
    while cnt < 5000 and sample_cnt < 20000:
        for config in configuration_space.sample_configuration(batch_size):
            sample_cnt += 1
            if config not in result:
                result.append(config)
                cnt += 1
    result = set(result)
    return len(result)


def show_hpspace(classifier_id='k_nearest_neighbors'):
    clf_class = _classifiers[classifier_id]
    cs = clf_class.get_hyperparameter_search_space()
    configs = cs.sample_configuration(10000)
    print(classifier_id)
    print(len(configs), len(set(configs)))
    print(explore_config_nums(cs))
    print('='*30)


if __name__ == "__main__":
    algorithms = ['lda', 'k_nearest_neighbors', 'libsvm_svc', 'sgd',
                  'adaboost', 'random_forest', 'extra_trees', 'gradient_boosting']
    for algo in algorithms:
        _start_time = time.time()
        show_hpspace(classifier_id=algo)
        print(time.time() - _start_time)
