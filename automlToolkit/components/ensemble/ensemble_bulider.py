from sklearn.metrics.scorer import _BaseScorer
import numpy as np

ensemble_list = ['bagging', 'blending', 'stacking', 'ensemble_selection']

from automlToolkit.components.ensemble.bagging import Bagging
from automlToolkit.components.ensemble.blending import Blending
from automlToolkit.components.ensemble.stacking import Stacking
from automlToolkit.components.ensemble.ensemble_selection import EnsembleSelection


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
        elif ensemble_method == 'stacking':
            self.model = Stacking(stats=stats,
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
