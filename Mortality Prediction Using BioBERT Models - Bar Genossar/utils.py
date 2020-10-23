import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import random
import numpy as np
from process_data import NotesDataReader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
import datetime
import time


def check_input(model_name, task_type):
    valid_values = {
                    "FCBERT": ["train_primary"],
                    "RCNN": ["train_primary"],
                    "ACFCBERT": ["train_sub", "test"],
                    "ACRCNN": ["train_sub", "classify_sub", "test"]
                   }
    if model_name not in valid_values.keys():
        raise TypeError("The model" + model_name + "is not valid")
    if task_type not in valid_values[model_name]:
        raise TypeError("The task" + task_type + "is not defined")
    return


def find_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_config_main(config):
    return config.DATA_FRAME, config.MAX_LEN, config.TRAIN_RATIO, \
           config.TEXT_FEATURE, config.MODEL_NAME, config.MODELS_PATH, \
           config.EMBEDDINGS_VERSION, config.EMBEDDINGS_PATH, config.TASK_TYPE


def read_config_primary(config, task_type):
    loss_function = config.LOSS_FUNCTION
    labels_num = config.LABELS_NUM
    dropout = config.DROPOUT
    epochs_num = config.EPOCHS_NUM
    threshold = config.THRESHOLD
    if "train" in task_type:
        hid_dim_lstm = int(random.choice(np.linspace(config.HIDDEN_DIM_LSTM[0],
                                                     config.HIDDEN_DIM_LSTM[1],
                                                     config.HIDDEN_DIM_LSTM[2])))

        lin_output_dim = int(random.choice(np.linspace(config.LINEAR_OUTPUT_DIM[0],
                                                       config.LINEAR_OUTPUT_DIM[1],
                                                       config.LINEAR_OUTPUT_DIM[2])))

        lr = random.choice(np.linspace(config.LEARNING_RATE[0],
                                       config.LEARNING_RATE[1],
                                       config.LEARNING_RATE[2]))

        # batch_size = int(random.choice(np.linspace(config.BATCH_SIZE[0],
        #                                            config.BATCH_SIZE[1],
        #                                            config.BATCH_SIZE[2])))
        batch_size = config.BATCH_SIZE_VAL

        momentum = random.choice(np.linspace(config.MOMENTUM[0],
                                             config.MOMENTUM[1],
                                             config.MOMENTUM[2]))
    elif "test" in task_type:
        hid_dim_lstm = config.HIDDEN_DIM_LSTM_VAL
        lin_output_dim = config.LINEAR_OUTPUT_DIM_val
        lr = config.LEARNING_RATE_val
        batch_size = config.BATCH_SIZE_val
        momentum = config.MOMENTUM_val
    else:
        raise TypeError("The task " + task_type + " is not defined")
    return loss_function, labels_num, dropout, epochs_num, threshold, \
           hid_dim_lstm, lin_output_dim, lr, batch_size, momentum


def split_train_val(df_file, train_ratio):
    df = pd.read_csv(df_file)
    return train_test_split(df, random_state=29, test_size=1-train_ratio)


def create_dataloaders_train(train_df, val_df, text_feature,
                             embeddings_version, max_len, batch_size):
    train_datareader = NotesDataReader(train_df, embeddings_version, max_len, text_feature)
    val_datareader = NotesDataReader(val_df, embeddings_version, max_len, text_feature)
    train_dataloader = DataLoader(train_datareader, batch_size=batch_size)
    val_dataloader = DataLoader(val_datareader, batch_size=batch_size)
    return train_dataloader, val_dataloader


def create_dataloaders_test(df, embeddings_version, max_len, text_feature, batch_size):
    test_datareader = NotesDataReader(df, embeddings_version, max_len, text_feature)
    return DataLoader(test_datareader, batch_size=batch_size)


def write_to_file(model_name, text):
    print(text)
    fout = open(str(model_name) + ".txt", "a")
    fout.write(text + '\n')
    fout.close()
    return


def write_results(labels, predictions, file_name):
    fout = open(file_name + ".txt", "a")
    for i in range(len(labels)):
        curr_str = str(i) + ":  "
        curr_str += "label: " + str(labels[i])
        curr_str += " , prediction: " + str(predictions[i])
        curr_str += "\n"
        fout.write(curr_str)
    fout.close()


def print_summary(models_path, model_name, accuracy_list_dev):
    write_to_file(models_path + model_name, "Accuracy Dev:")
    write_to_file(models_path + model_name, str(accuracy_list_dev))
    write_to_file(models_path + model_name, "Best Accuracy Dev:")
    write_to_file(models_path + model_name, str(max(accuracy_list_dev)))
    write_to_file(models_path + model_name, "")
    write_to_file(models_path + model_name, "Training complete!")
    write_to_file(models_path + model_name, "-------------------------------------------------------------")
    write_to_file(models_path + model_name, "")


def print_train_epoch_end(t0):
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Calculate val accuracy")
    print("")

def print_test_results(models_path, model_name, acc):
    write_to_file(models_path + model_name, "Accuracy Test:")
    write_to_file(models_path + model_name, str(acc))
    write_to_file(models_path + model_name, "-------------------------------------------------------------")
    write_to_file(models_path + model_name, "")


def print_train_epoch(epoch, correct_num, train_len, loss_scalar, total_time):
    print("train accuracy for epoch " + str(epoch + 1) + " is: %.3f" % float(correct_num / train_len))
    print("loss after epoch", epoch + 1, "is: %.3f" % float(loss_scalar))
    print("total time: %.3f" % total_time)
    print()


def print_validation_epoch(acc_num, val_len, loss_scalar, total_time):
    print("Dev accuracy for this epoch: %.3f" % float(acc_num / val_len))
    print("Loss for this epoch %.3f" % float(loss_scalar))
    print("Total time: %.3f" % total_time)
    print()
    print()
    return


def write_results(labels, predictions, file_name):
    fout = open(file_name + ".txt", "a")
    for i in range(len(labels)):
        curr_str = str(i) + ":  "
        curr_str += "label: " + str(labels[i])
        curr_str += " , prediction: " + str(predictions[i])
        curr_str += "\n"
        fout.write(curr_str)
    fout.close()


def print_epochs_progress(epoch, epochs_num):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs_num))
    print('Training...')


def print_batches_progress(t0, batch_idx, train_dataloader):
    elapsed = format_time(time.time() - t0)
    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(batch_idx,
                                                                len(train_dataloader), elapsed))


def save_model(models_df, model_name, hid_dim_lstm, labels_num, loss_function, dropout, lin_output_dim, lr,
               epochs_num, batch_size, momentum, best_dev_acc):
    models_df.loc[models_df.index == model_name, 'hid_dim_lstm'] = hid_dim_lstm
    models_df.loc[models_df.index == model_name, 'labels_num'] = labels_num
    models_df.loc[models_df.index == model_name, 'loss_function'] = loss_function
    models_df.loc[models_df.index == model_name, 'dropout'] = dropout
    models_df.loc[models_df.index == model_name, 'lin_output_dim'] = lin_output_dim
    models_df.loc[models_df.index == model_name, 'lr'] = lr
    models_df.loc[models_df.index == model_name, 'epochs_num'] = epochs_num
    models_df.loc[models_df.index == model_name, 'batch_size'] = batch_size
    models_df.loc[models_df.index == model_name, 'momentum'] = momentum
    models_df.loc[models_df.index == model_name, 'accuracy'] = round(best_dev_acc, 3)


def convert_to_torch(inputs_ids, masks, labels):
    return torch.tensor(inputs_ids), torch.tensor(masks), torch.tensor(labels)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def add_accuracy(predictions, labels):
    predictions_flat = predictions.to('cpu').numpy()
    labels_flat = labels.to('cpu').numpy()
    predictions_flat = predictions_flat.flatten()
    labels_flat = labels_flat.flatten()
    # print(predictions_flat)
    # print(labels_flat)
    return np.sum(predictions_flat == labels_flat) / len(labels_flat)


def update_predictions_dict(predictions_dict, hadm_id, predictions, probabilities, labels):
    tmp_dict = dict(zip(hadm_id.tolist(),
                        zip(predictions.tolist(),
                            probabilities.tolist(),
                            labels.tolist())))
    return {**predictions_dict, **tmp_dict}


def save_predictions_to_df(predictions_dict, models_path, dataset_type, model_name):
    df = pd.DataFrame.from_dict(predictions_dict, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['HADM_ID', 'prediction', 'probability', 'label']
    df.to_csv(models_path + 'predictions_' + dataset_type + '_' + model_name + '.csv')


def print_new_sub_model(sub_model):
    print("**************************************************************")
    print(sub_model)
    print("**************************************************************")
    print("")
