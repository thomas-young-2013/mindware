# encoding=utf8
import numpy as np

from ..models.abstract_model import AbstractModel
from .acquisition import AbstractAcquisitionFunction, EI, PI


class TAQ_EI(AbstractAcquisitionFunction):

    def __init__(self,
                 model: AbstractModel,
                 source_models,
                 aggregate_method='taff',
                 par: float=0.0):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """

        super(TAQ_EI, self).__init__(model)
        self.long_name = 'TAQF Expected Improvement'
        self.par = par
        self.eta = None
        self.ei_acq = EI(model, par=par)
        self.source_models = source_models
        self.source_etas = None
        self.model_weights = None
        self.aggregate_method = aggregate_method
        if self.aggregate_method == 'taff2':
            self.pi_list = list()
            for i in range(len(self.source_models)):
                self.pi_list.append(PI(self.source_models[i]))

    def update_target_model(self, model, eta, num_data, source_etas, model_weights):
        self.ei_acq.update(model=model, eta=eta, num_data=num_data)
        self.source_etas = source_etas
        self.model_weights = model_weights

        if self.aggregate_method == 'taff2':
            for i in range(len(self.source_models)):
                self.pi_list[i].update(eta=self.source_etas[i])

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        if self.model_weights is None:
            raise ValueError('No model weights specified.')

        n_source_tasks = len(self.source_models)
        assert n_source_tasks + 1 == len(self.model_weights)
        acq_values = self.model_weights[-1] * self.ei_acq._compute(X)
        for i in range(n_source_tasks):
            if self.aggregate_method == 'taff':
                m, v = self.source_models[i].predict_marginalized_over_instances(X)
                s = np.sqrt(v)
                eta_ = self.source_etas[i]
                y_ = np.random.normal(m.flatten(), s.flatten())
                par_acq = np.max(eta_ - y_, 0)
                acq_values += self.model_weights[i] * par_acq.reshape(-1, 1)
            elif self.aggregate_method == 'taff2':
                acq_values += self.model_weights[i] * self.pi_list[i]._compute(X)
            else:
                raise ValueError('Invalid method!')
        return acq_values
