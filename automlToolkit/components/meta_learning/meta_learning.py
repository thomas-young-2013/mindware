import time
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter

from automlToolkit.components.evaluator import Evaluator
from automlToolkit.utils.metalearning import get_trans_from_str


def apply_metalearning_fe(optimizer, configs):
    rescaler_id = None
    rescaler_dict = {}
    preprocessor_id = None
    preprocessor_dict = {}
    for key in configs:
        key_str = key.split(":")
        # TODO: Imputation, Encoding and Balancing is required in fe_pipeline
        if key_str[0] == 'rescaling':
            if key_str[1] == '__choice__':
                rescaler_id = configs[key]
            else:
                rescaler_dict[key_str[2]] = configs[key]
        elif key_str[0] == 'preprocessor':
            if key_str[1] == '__choice__':
                preprocessor_id = configs[key]
            else:
                preprocessor_dict[key_str[2]] = configs[key]

    preprocessor_tran = get_trans_from_str(preprocessor_id)(**preprocessor_dict)
    rescaler_tran = get_trans_from_str(rescaler_id)(**rescaler_dict)
    rescale_node = rescaler_tran.operate(optimizer.root_node)
    depth = 2
    if rescaler_tran.type != 0:
        rescale_node.depth = depth
        depth += 1
        rescale_node.trans_hist.append(rescaler_tran.type)
        rescale_node.score = 0
        optimizer.temporary_nodes.append(rescale_node)
        optimizer.graph.add_node(rescale_node)
        # Avoid self-loop.
        if rescaler_tran.type != 0 and optimizer.root_node.node_id != rescale_node.node_id:
            optimizer.graph.add_trans_in_graph(optimizer.root_node, rescale_node, rescaler_tran)

    preprocess_node = preprocessor_tran.operate(rescale_node)
    if preprocessor_tran.type != 0:
        preprocess_node.depth = depth
        preprocess_node.trans_hist.append(preprocessor_tran.type)
        preprocess_node.score = 0
        optimizer.temporary_nodes.append(preprocess_node)
        optimizer.graph.add_node(preprocess_node)
        # Avoid self-loop.
        if preprocessor_tran.type != 0 and rescale_node.node_id != preprocess_node.node_id:
            optimizer.graph.add_trans_in_graph(rescale_node, preprocess_node, preprocessor_tran)

    return preprocess_node


def evaluate_metalearning_configs(first_bandit):
    score_list = []
    for config in first_bandit.meta_configs:
        try:
            config = config.get_dictionary()
            # print(config)
            arm = None
            cs = ConfigurationSpace()
            for key in config:
                key_str = key.split(":")
                if key_str[0] == 'classifier':
                    if key_str[1] == '__choice__':
                        arm = config[key]
                        cs.add_hyperparameter(UnParametrizedHyperparameter("estimator", config[key]))
                    else:
                        cs.add_hyperparameter(UnParametrizedHyperparameter(key_str[2], config[key]))

            if arm in first_bandit.arms:
                transformed_node = apply_metalearning_fe(first_bandit.sub_bandits[arm].optimizer['fe'], config)
                default_config = cs.sample_configuration(1)
                hpo_evaluator = Evaluator(None,
                                          data_node=transformed_node, name='hpo',
                                          resampling_strategy=first_bandit.eval_type,
                                          seed=first_bandit.seed)

                start_time = time.time()
                score = 1 - hpo_evaluator(default_config)
                time_cost = time.time() - start_time
                score_list.append((arm, score, default_config, transformed_node, time_cost))
                transformed_node.score = score

                # Evaluate the default config
                start_time = time.time()
                score = 1 - hpo_evaluator(first_bandit.sub_bandits[arm].default_config)
                time_cost = time.time() - start_time
                score_list.append(
                    (arm, score, first_bandit.sub_bandits[arm].default_config, transformed_node, time_cost))
                transformed_node.score = score
        except Exception as e:
            print(e)

    # Sort the meta-configs
    score_list.sort(key=lambda x: x[1], reverse=True)
    meta_arms = list()
    for arm_score_config in score_list:
        if arm_score_config[0] in meta_arms:
            continue

        first_bandit.sub_bandits[arm_score_config[0]].default_config = arm_score_config[2]
        first_bandit.sub_bandits[arm_score_config[0]].collect_iter_stats('fe', (
            arm_score_config[1], arm_score_config[4], arm_score_config[3]))
        # first_bandit.sub_bandits[arm_score_config[0]].collect_iter_stats('hpo',
        #                                                                  (arm_score_config[1], arm_score_config[4],
        #                                                                   arm_score_config[2]))
        first_bandit.sub_bandits[arm_score_config[0]].optimizer['fe'].hp_config = arm_score_config[2]
        meta_arms.append(arm_score_config[0])
    for arm in first_bandit.arms:
        if arm not in meta_arms:
            meta_arms.append(arm)

    first_bandit.final_rewards.append(score_list[0][1])
    first_bandit.action_sequence.append(score_list[0][0])
    first_bandit.time_records.append(score_list[0][2])
    first_bandit.arms = meta_arms
    first_bandit.logger.info("Arms after evaluating meta-configs: " + str(first_bandit.arms))
