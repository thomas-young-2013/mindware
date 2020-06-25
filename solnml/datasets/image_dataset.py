import os
from .base_dataset import BaseDataset
from solnml.components.models.img_classification.nn_utils.dataset import get_folder_dataset
from solnml.components.models.img_classification.nn_utils.nn_aug.aug_hp_space import get_test_transforms


class ImageDataset(BaseDataset):
    def __init__(self, data_path: str,
                 data_transforms: dict = None,
                 grayscale: bool = False,
                 train_val_split: bool = False,
                 val_split_size: float = 0.2):
        super().__init__()
        self.train_val_split = train_val_split
        self.val_split_size = val_split_size
        self.data_path = data_path

        self.udf_transforms = data_transforms
        self.grayscale = grayscale

    def set_udf_transform(self, udf_transform):
        self.udf_transforms = udf_transform

    def load_data(self):
        self.train_dataset = get_folder_dataset(os.path.join(self.data_path, 'train'),
                                                udf_transforms=self.udf_transforms['train'],
                                                grayscale=self.grayscale)
        if not self.train_val_split:
            self.val_dataset = get_folder_dataset(os.path.join(self.data_path, 'val'),
                                                  udf_transforms=self.udf_transforms['val'],
                                                  grayscale=self.grayscale)
        else:
            self.create_train_val_split(self.train_dataset, train_val_split=self.val_split_size, shuffle=True)

    def load_test_data(self, test_data_path: str = None):
        if test_data_path is None:
            test_data_path = self.data_path

        transforms = get_test_transforms()
        self.test_dataset = get_folder_dataset(os.path.join(test_data_path, 'test'),
                                               udf_transforms=transforms,
                                               grayscale=self.grayscale)
