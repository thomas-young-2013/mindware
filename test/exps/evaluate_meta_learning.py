import os
import sys

sys.path.append(os.getcwd())
from autosklearn.smbo import AutoMLSMBO
from autosklearn.constants import *
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.util.backend import create
from autosklearn.util import pipeline, StopWatch

from automlToolkit.datasets.utils import load_data

dataset_name = 'diabetes'
X, y, _ = load_data(dataset_name)


def get_meta_learning_configs(X, y, task_type, dataset_name, metric='accuracy', num_cfgs=5):
    backend = create(temporary_directory=None,
                     output_directory=None,
                     delete_tmp_folder_after_terminate=False,
                     delete_output_folder_after_terminate=False,
                     shared_mode=True)
    dm = XYDataManager(X, y, None, None, task_type, None, dataset_name)

    configuration_space = pipeline.get_configuration_space(dm.info,
                                                           include_estimators=None,
                                                           exclude_estimators=None,
                                                           include_preprocessors=None,
                                                           exclude_preprocessors=None)

    watcher = StopWatch()
    name = os.path.basename(dm.name)
    watcher.start_task(name)

    def reset_data_manager(max_mem=None):
        pass

    automlsmbo = AutoMLSMBO(config_space=configuration_space,
                            dataset_name=dataset_name,
                            backend=backend,
                            total_walltime_limit=1e5,
                            func_eval_time_limit=1e5,
                            memory_limit=1e5,
                            metric=metric,
                            watcher=watcher,
                            metadata_directory=None,
                            num_metalearning_cfgs=num_cfgs)
    automlsmbo.reset_data_manager = reset_data_manager
    automlsmbo.task = task_type
    automlsmbo.datamanager = dm
    configs = automlsmbo.get_metalearning_suggestions()
    return configs


print(get_meta_learning_configs(X, y, BINARY_CLASSIFICATION,
                                dataset_name='diabetes',
                                metric='accuracy',
                                num_cfgs=5))
