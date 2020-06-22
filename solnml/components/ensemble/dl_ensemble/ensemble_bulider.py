from sklearn.metrics.scorer import _BaseScorer
from solnml.components.ensemble.dl_ensemble.bagging import Bagging
from solnml.components.ensemble.dl_ensemble.blending import Blending
from solnml.components.ensemble.dl_ensemble.ensemble_selection import EnsembleSelection
ensemble_list = ['bagging', 'blending', 'stacking', 'ensemble_selection']


class EnsembleBuilder:
    def __init__(self, stats, ensemble_method: str,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None):
        self.model = None
        if ensemble_method == 'bagging':
            self.model = Bagging(stats=stats,
                                 ensemble_size=ensemble_size,
                                 task_type=task_type,
                                 metric=metric,
                                 output_dir=output_dir)
        elif ensemble_method == 'blending':
            self.model = Blending(stats=stats,
                                  ensemble_size=ensemble_size,
                                  task_type=task_type,
                                  metric=metric,
                                  output_dir=output_dir)
        elif ensemble_method == 'ensemble_selection':
            self.model = EnsembleSelection(stats=stats,
                                           ensemble_size=ensemble_size,
                                           task_type=task_type,
                                           metric=metric,
                                           output_dir=output_dir)
        else:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)

    def fit(self, data):
        return self.model.fit(data)

    def predict(self, data, solvers):
        return self.model.predict(data, solvers)

    def refit(self):
        return self.model.refit()

    def get_ens_model_info(self):
        return self.model.get_ens_model_info()
