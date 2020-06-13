from transformers import BertModel, BertPreTrainedModel
import torch
import torch.nn.functional as F
from torch import nn


class RCNN_Model(BertPreTrainedModel):

    def __init__(self, num_class, config):
        super(RCNN_Model, self).__init__(config)
        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(768, 256, 2,
                            bidirectional=True, batch_first=True, dropout=0.1)
        self.maxpool = nn.MaxPool1d(512)
        self.fc = nn.Linear(512 + 768, num_class)

    def forward(self, x, masks):
        encoder_out, text_cls = self.bert(input_ids=x, attention_mask=masks)
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        # print(out.size())
        out = self.maxpool(out)
        out = out.squeeze()
        out = self.fc(out)
        return out
