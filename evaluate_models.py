def test_alphaml():
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, UnParametrizedHyperparameter
    from alphaml.engine.components.models.classification.adaboost import AdaboostClassifier

    cs = AdaboostClassifier.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", 'random_forest')
    cs.add_hyperparameter(model)

    config = cs.get_default_configuration()


def test_ausk():
    from ConfigSpace.hyperparameters import UnParametrizedHyperparameter
    from autosklearn.pipeline.components.classification.adaboost import AdaboostClassifier

    cs = AdaboostClassifier.get_hyperparameter_search_space()
    model = UnParametrizedHyperparameter("estimator", 'adaboost')
    cs.add_hyperparameter(model)

    for config in cs.sample_configuration(4):
        print('='*20)
        name, estimator = get_estimator(config)
        for attr in dir(estimator):
            if type(getattr(estimator, attr)) in [int, str, float]:
                print("obj.%s = %r" % (attr, getattr(estimator, attr)))


def get_estimator(config):
    from autosklearn.pipeline.components.classification import _classifiers
    classifier_type = config['estimator']
    config = config.get_dictionary()
    config.pop('estimator', None)
    print(_classifiers[classifier_type])
    estimator = _classifiers[classifier_type](**config)
    return classifier_type, estimator


if __name__ == '__main__':
    test_ausk()
