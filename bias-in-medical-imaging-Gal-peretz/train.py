from typing import Any, List
import torch
from result import FitResult


def run_epoch(
        model,
        optimizer,
        loss_functions: List[Any],
        accuracy_fn,
        data_loader,
        is_train
):
    loss_acc = 0
    accuracy_acc = 0
    number_of_batches = 0
    with torch.set_grad_enabled(is_train):
        for images, labels, biases in data_loader:
            scores = model(images.cuda())
            optimizer.zero_grad()
            losses = [loss_fn(scores, labels, biases) for loss_fn in loss_functions]
            single_loss = sum(losses)
            loss_acc += single_loss.item() / len(losses)
            accuracy_acc += accuracy_fn(scores, labels, biases).item()
            number_of_batches += 1
            if is_train:
                single_loss.backward()
                optimizer.step()
    loss_mean = loss_acc / number_of_batches
    accuracy_mean = accuracy_acc / number_of_batches
    return accuracy_mean, loss_mean

def train_and_evaluate(number_of_epoch, model, optimizer, loss_fn, accuracy_fn, dl_train, dl_test, should_log=False) -> FitResult:
    epoch_train_losses = []
    epoch_train_accuracy = []
    epoch_test_losses = []
    epoch_test_accuracy = []
    for epoch in range(number_of_epoch):
        accuracy_mean, loss_mean = run_epoch(model, optimizer, loss_fn, accuracy_fn, dl_train, is_train=True)
        epoch_train_losses.append(loss_mean)
        epoch_train_accuracy.append(accuracy_mean)
        if should_log:
            print(f"Training epoch: {epoch}, accuracy:{accuracy_mean:.6f}, loss: {loss_mean:.6f} ")
        accuracy_mean, loss_mean = run_epoch(model, optimizer, loss_fn, accuracy_fn, dl_test, is_train=False)
        epoch_test_losses.append(loss_mean)
        epoch_test_accuracy.append(accuracy_mean)
        if should_log:
            print(f"Testing epoch: {epoch}, accuracy:{accuracy_mean:.6f}, loss: {loss_mean:.6f} ")
    result = FitResult(len(epoch_train_losses),
                       epoch_train_losses, epoch_train_accuracy,
                       epoch_test_losses, epoch_test_accuracy)
    return result