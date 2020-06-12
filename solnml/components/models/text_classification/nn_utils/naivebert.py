from transformers import BertModel, BertPreTrainedModel
from torch import nn


class Base_Model(BertPreTrainedModel):
    def __init__(self, num_classes, config):
        super(Base_Model, self).__init__(config)
        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(768, num_classes)
        self.init_weights()

    def forward(self, x, masks):
        encoder_out, text_cls = self.bert(input_ids=x, attention_mask=masks)
        x = self.dropout(text_cls)
        x = self.fc1(x)
        return x
