import torch
import torch.nn as nn
from torch.nn import functional as Func
from transformers import BertModel


class FCBERT(nn.Module):
    def __init__(self, embeddings_path, labels_num):
        super(FCBERT, self).__init__()
        self.model = BertModel.from_pretrained(embeddings_path)
        self.num_labels = labels_num
        self.loss = nn.NLLLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_W1 = nn.Linear(self.model.config.hidden_size, labels_num)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask, ground_truth):
        _, pooler = self.model(input_ids=input_ids, attention_mask=attention_mask)
        Y1 = self.linear_W1(pooler)
        lsm = self.logsoftmax(Y1)
        softmax_values = Func.softmax(Y1, dim=1)
        probabilities, predictions = softmax_values.max(1)
        if lsm.shape[0] > input_ids.shape[0]:
            lsm = lsm[:input_ids.shape[0]]
        loss_val = self.loss(lsm, ground_truth.to(self.device))
        return loss_val, predictions, probabilities
