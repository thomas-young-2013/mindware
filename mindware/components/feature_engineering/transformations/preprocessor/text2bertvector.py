from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition
from mindware.components.feature_engineering.transformations.base_transformer import *
from mindware.components.utils.text_util import build_embeddings_index, load_text_embeddings


class Text2BertVectorTransformation(Transformer):
    type = 500

    def __init__(self, padding_size=256, config_dir=None):
        super().__init__("text2bertvector")
        self.input_type = [TEXT]
        self.output_type = [TEXT_EMBEDDING]
        self.compound_mode = 'replace'

        self.padding_size = padding_size
        self.config_dir = config_dir

    def padding(self, sample):
        sample = sample + [0] * (self.padding_size - len(sample))
        return sample

    @ease_trans
    def operate(self, input_datanode, target_fields=None):
        from transformers import BertTokenizer, BertModel
        import torch

        if self.config_dir is None:
            self.config_dir = './bert-base'

        tokenizer = BertTokenizer.from_pretrained(self.config_dir)
        model = BertModel.from_pretrained(self.config_dir)

        X, y = input_datanode.data
        X_new = X[:, target_fields]
        _X = None

        for i in range(X_new.shape[1]):
            tokens = list()
            for str4token in X_new[:, i]:
                tokens.append(self.padding(tokenizer.encode(str4token)))
            tensor = torch.LongTensor(tokens)
            _, emb_output = model.forward(tensor)
            emb_output = emb_output.detach().numpy()
            if _X is None:
                _X = emb_output.copy()
            else:
                _X = np.hstack((_X, emb_output))

        return _X

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='tpe'):
        cs = ConfigurationSpace()
        return cs
