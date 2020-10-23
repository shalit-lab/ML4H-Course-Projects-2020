import utils
import time
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from FCBERT_classifier import FCBERT
from RCNN_classifier import RCNN


def get_model(model_name, embeddings_path,
              hid_dim_lstm, loss_function,
              labels_num, dropout, lin_output_dim):
    if model_name == "RCNN":
        return RCNN(embeddings_path, hidden_dim_lstm=hid_dim_lstm,
                    loss_function=loss_function, labels_num=labels_num,
                    dropout=dropout, linear_output_dim=lin_output_dim)
    elif model_name == "FCBERT":
        return FCBERT(embeddings_path, labels_num=labels_num)
    else:
        raise TypeError("The model " + model_name + " is not defined")


def train_primary(model_name, df_name, max_len, train_ratio,
                  text_feature, embeddings_version, embeddings_path,
                  config_primary, models_path, models_df):
    device = utils.find_device()

    loss_function, labels_num, dropout, epochs_num, threshold, \
        hid_dim_lstm, lin_output_dim, lr, batch_size, momentum = \
        utils.read_config_primary(config_primary, "train_primary")

    train_df, val_df = utils.split_train_val(df_name, train_ratio)

    train_dataloader, val_dataloader = utils.create_dataloaders_train(train_df, val_df,
                                                                      text_feature,
                                                                      embeddings_version,
                                                                      max_len, batch_size)

    model = get_model(model_name, embeddings_path,
                      hid_dim_lstm, loss_function,
                      labels_num, dropout, lin_output_dim)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5,  eps=1e-8)
    train_len = len(train_dataloader)
    val_len = len(val_dataloader)
    train_accuracy_list, train_loss_list, accuracy_list_val, loss_list_val = [], [], [], []
    best_val_acc = 0.0
    total_steps = len(train_dataloader) * epochs_num
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    for epoch in range(epochs_num):
        utils.print_epochs_progress(epoch, epochs_num)
        start_train = time.clock()
        acc_num_train = 0.0
        loss_scalar = 0.0
        model.train()
        optimizer.zero_grad()
        t0 = time.time()
        predictions_dict_train = dict()
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx % 10 == 0 and not batch_idx == 0:
                utils.print_batches_progress(t0, batch_idx, train_dataloader)
            input_ids = batch[0].to(device, dtype=torch.long)
            masks = batch[1].to(device, dtype=torch.long)
            labels = batch[2].to(device, dtype=torch.long)
            hadm_id = batch[3]
            model.zero_grad()
            loss, predictions, probabilities = model(input_ids, masks, labels)
            loss_scalar += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            acc_num_train += utils.add_accuracy(predictions, labels)
            optimizer.step()
            scheduler.step()
            predictions_dict_train = utils.update_predictions_dict(predictions_dict_train,
                                                                   hadm_id,
                                                                   predictions,
                                                                   probabilities,
                                                                   labels)
        loss_scalar /= train_len
        end_train = time.clock()
        total_time = end_train - start_train
        utils.print_train_epoch(epoch, acc_num_train, train_len, loss_scalar, total_time)
        train_loss_list.append(round(float(loss_scalar), 3))
        train_accuracy_list.append(round(float(acc_num_train / train_len), 3))
        utils.print_train_epoch_end(t0)
        val_acc, val_loss, predictions_dict_val = evaluate_val(model, val_dataloader,
                                                               device, val_len)
        accuracy_list_val.append(val_acc)
        loss_list_val.append(val_loss)
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), models_path + 'primary.pkl')
            best_val_acc = val_acc
            utils.save_predictions_to_df(predictions_dict_train, models_path, 'train', 'primary')
            utils.save_predictions_to_df(predictions_dict_val, models_path, 'validation', 'primary')
    utils.save_model(models_df, 'primary', hid_dim_lstm, labels_num, loss_function, dropout,
                     lin_output_dim, lr, epochs_num, batch_size, momentum, best_val_acc)
    utils.print_summary(models_path, 'primary', accuracy_list_val)
    test_acc, predictions_dict_test = evaluate_test(model, device,
                                                    embeddings_version,
                                                    max_len, text_feature,
                                                    batch_size)
    utils.save_predictions_to_df(predictions_dict_test, models_path, 'test', 'primary')
    utils.print_test_results(models_path, 'primary', test_acc)
    return models_df


def evaluate_val(model, val_dataloader,  device, val_len):
    start_val = time.clock()
    acc_num_val = 0.0
    loss_scalar = 0.0
    model.eval()
    predictions_dict_val = dict()
    for batch_idx, batch in enumerate(val_dataloader):
        input_ids = batch[0].to(device, dtype=torch.long)
        masks = batch[1].to(device, dtype=torch.long)
        labels = batch[2].to(device, dtype=torch.long)
        hadm_id = batch[3]
        with torch.no_grad():
            loss, predictions, probabilities = model(input_ids, masks, labels)
            predictions_dict_val = utils.update_predictions_dict(predictions_dict_val,
                                                                 hadm_id,
                                                                 predictions,
                                                                 probabilities,
                                                                 labels)
            loss_scalar += loss.item()
            acc_num_val += utils.add_accuracy(predictions, labels)
    loss_scalar /= val_len
    end_val = time.clock()
    total_time = end_val - start_val
    utils.print_validation_epoch(acc_num_val, val_len, loss_scalar, total_time)
    return float(acc_num_val / val_len), float(loss_scalar), predictions_dict_val


def evaluate_test(model, device, embeddings_version, max_len, text_feature, batch_size):
    test_df = pd.read_csv('MORTALITY_TEST.csv')
    start_val = time.clock()
    test_dataloader = utils.create_dataloaders_test(test_df, embeddings_version,
                                                    max_len, text_feature, batch_size)
    test_len = len(test_dataloader)
    acc_num_test = 0.0
    model.eval()
    predictions_dict_test = dict()
    for batch_idx, batch in enumerate(test_dataloader):
        input_ids = batch[0].to(device, dtype=torch.long)
        masks = batch[1].to(device, dtype=torch.long)
        labels = batch[2].to(device, dtype=torch.long)
        hadm_id = batch[3]
        with torch.no_grad():
            _, predictions, probabilities = model(input_ids, masks, labels)
            predictions_dict_test = utils.update_predictions_dict(predictions_dict_test,
                                                                  hadm_id, predictions,
                                                                  probabilities, labels)
            acc_num_test += utils.add_accuracy(predictions, labels)
    end_val = time.clock()
    total_time = end_val - start_val
    print(total_time)
    return float(acc_num_test / test_len), predictions_dict_test
