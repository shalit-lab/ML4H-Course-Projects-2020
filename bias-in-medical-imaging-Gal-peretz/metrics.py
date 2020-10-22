import torch.nn.functional as F
import torch


def accuracy(scores, y_true):
    preds = F.softmax(scores, dim=1)
    y_preds = torch.argmax(preds, dim=1).squeeze()
    return torch.div((y_true == y_preds).sum(), float(len(y_true)))