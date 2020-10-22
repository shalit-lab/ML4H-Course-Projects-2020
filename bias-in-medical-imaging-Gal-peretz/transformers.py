import torch
from datasets import ClassesLabels
from torchvision.transforms import transforms

def convert_classes_label_to_int(label):
    if label == ClassesLabels.Glioma:
        return 0
    elif label == ClassesLabels.Meningioma:
        return 1
    return 2

def process(batch):
    images = []
    labels = []
    is_biases = []
    for image, label, is_bias in batch:
        images.append(transforms.ToTensor()(image))
        labels.append(torch.tensor(convert_classes_label_to_int(label)))
        is_biases.append(torch.tensor(1.) if is_bias else torch.tensor(0.))
    return torch.stack(images), torch.stack(labels), torch.stack(is_biases)
