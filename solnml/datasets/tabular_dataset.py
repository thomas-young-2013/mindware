from .base_dl_dataset import BaseDataset
from solnml.utils.data_manager import DataManager
from solnml.components.utils.constants import REGRESSION, CLASSIFICATION
from solnml.components.feature_engineering.fe_pipeline import FEPipeline


class TabularDataset(BaseDataset):
    def __init__(self, data_path: str,
                 is_regression=False,
                 label_column=-1,
                 header='infer',
                 sep=',',
                 nan_values=("n/a", "na", "--", "-", "?"),
                 train_val_split: bool = False,
                 val_split_size: float = 0.2):
        super().__init__()
        self.is_regression = is_regression
        self.train_val_split = train_val_split
        self.val_split_size = val_split_size
        self.data_path = data_path
        self.label_column = label_column
        self.header = header
        self.sep = sep
        self.nan_values = nan_values
        self.data_manager = None
        self._process_pipeline = None

    def load_tabular_data(self, data_path):
        self.data_manager = DataManager()
        train_data_node = self.data_manager.load_train_csv(data_path,
                                                           label_col=self.label_column,
                                                           header=self.header, sep=self.sep,
                                                           na_values=list(self.nan_values))

        task_type = REGRESSION if self.is_regression else CLASSIFICATION
        self._process_pipeline = FEPipeline(fe_enabled=False, metric='acc', task_type=task_type)
        return self._process_pipeline.fit_transform(train_data_node)

    def load_data(self):
        self.train_dataset = self.load_tabular_data(self.data_path)

    def load_test_data(self, test_data_path):
        test_data_node = self.data_manager.load_test_csv(test_data_path,
                                                         has_label=False,
                                                         keep_default_na=True,
                                                         header=self.header,
                                                         sep=self.sep)
        self.test_dataset = self._process_pipeline.transform(test_data_node)
