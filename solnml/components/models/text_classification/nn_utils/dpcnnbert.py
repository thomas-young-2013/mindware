from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F
from torch import nn


class DPCNNModel(BertPreTrainedModel):

    def __init__(self, config, num_class):
        super(DPCNNModel, self).__init__(config)
        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True
        # self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.conv_region = nn.Conv2d(1, 250, (3, 768), stride=1)
        self.conv = nn.Conv2d(250, 250, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(250, num_class)

    def forward(self, x, masks):
        encoder_out, text_cls = self.bert(input_ids=x, attention_mask=masks)
        x = encoder_out.unsqueeze(1)  # [batch_size, 1, seq_len, embed]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px  # short cut
        return x
