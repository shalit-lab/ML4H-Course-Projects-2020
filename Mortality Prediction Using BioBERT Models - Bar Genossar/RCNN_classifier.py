import torch
import torch.nn as nn
from torch.nn import functional as Func
from transformers import BertModel


class RCNN(nn.Module):
    def __init__(self, embeddings_path, hidden_dim_lstm, loss_function,
                 labels_num, dropout, linear_output_dim):
        super(RCNN, self).__init__()
        self.model = BertModel.from_pretrained(embeddings_path)
        self.embeddings_dim = self.model.config.hidden_size
        self.hidden_dim_lstm = hidden_dim_lstm
        self.W2_output_dim = linear_output_dim
        self.loss = loss_function
        self.labels_dim = labels_num
        self.dropout = dropout
        self.dropout_linear = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size=self.embeddings_dim,
                            hidden_size=self.hidden_dim_lstm,
                            dropout=self.dropout, bidirectional=True)
        self.linear_W2 = nn.Linear(2*self.hidden_dim_lstm+self.embeddings_dim,
                                   self.W2_output_dim)
        nn.init.xavier_uniform_(self.linear_W2.weight)
        nn.init.zeros_(self.linear_W2.bias)
        self.linear_W4 = nn.Linear(self.W2_output_dim, self.labels_dim)
        nn.init.xavier_uniform_(self.linear_W4.weight)
        nn.init.zeros_(self.linear_W4.bias)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask, ground_truth, calc_loss=True):
        words_embeds, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        words_embeds = words_embeds.permute(1, 0, 2)
        lstm_out, _ = self.lstm(words_embeds)
        X_cat = torch.cat([lstm_out, words_embeds], 2).to(self.device)
        Y2 = self.tanh(self.linear_W2(X_cat))
        Y2 = self.dropout_linear(Y2)
        Y2 = Y2.permute(1, 2, 0)
        Y3 = Func.max_pool1d(Y2, Y2.shape[2]).squeeze(2)
        Y4 = self.linear_W4(Y3)
        lsm = self.logsoftmax(Y4)
        softmax_values = Func.softmax(Y4, dim=1)
        probabilities, predictions = softmax_values.max(1)
        if calc_loss:
            loss_val = self.loss(lsm, ground_truth.to(self.device))
            return loss_val, predictions, probabilities
        else:
            return _, predictions, probabilities

