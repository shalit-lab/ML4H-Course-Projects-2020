import utils
import time
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from FCBERT_classifier import FCBERT
from RCNN_classifier import RCNN


def get_model(model_name, embeddings_path,
              hid_dim_lstm, loss_function,
              labels_num, dropout, lin_output_dim):
    if "RCNN" in model_name:
        return RCNN(embeddings_path, hidden_dim_lstm=hid_dim_lstm,
                    loss_function=loss_function, labels_num=labels_num,
                    dropout=dropout, linear_output_dim=lin_output_dim)
    elif "FCBERT" in model_name:
        return FCBERT(embeddings_path, labels_num=labels_num)
    else:
        raise TypeError("The model " + model_name + " is not defined")


def ov_un_label(row):
    return int(row['prediction'] > row['label'])


def create_over_under(models_path):
    df_train_primary = pd.read_csv(models_path + 'predictions_train.csv')
    df_val_primary = pd.read_csv(models_path + 'predictions_validation.csv')
    df_train_ov_un = df_train_primary[df_train_primary['label'] != df_train_primary['prediction']]
    df_val_ov_un = df_val_primary[df_val_primary['label'] != df_val_primary['prediction']]
    df_train_ov_un['label'] = df_train_ov_un.apply(lambda row: ov_un_label(row), axis=1)
    df_val_ov_un['label'] = df_val_ov_un.apply(lambda row: ov_un_label(row), axis=1)
    df_train_ov_un = df_train_ov_un[['HADM_ID', 'label']]
    df_val_ov_un = df_val_ov_un[['HADM_ID', 'label']]
    df_train_ov_un.to_csv('Mortality_sub_over_under_TRAIN.csv')
    df_val_ov_un.to_csv('Mortality_sub_over_under_VAL.csv')


def fix_values_over_under(train_df, val_df):
    df_train_ov_un = pd.read_csv('Mortality_sub_over_under_TRAIN.csv')
    df_val_ov_un = pd.read_csv('Mortality_sub_over_under_VAL.csv')
    dict_train = dict(zip(df_train_ov_un.HADM_ID.tolist(), df_train_ov_un.label.tolist()))
    dict_val = dict(zip(df_val_ov_un.HADM_ID.tolist(), df_val_ov_un.label.tolist()))
    train_df['label'] = train_df['HADM_ID'].apply(lambda x: dict_train[x])
    val_df['label'] = val_df['HADM_ID'].apply(lambda x: dict_val[x])
    return train_df, val_df


def create_sub_models_dfs(submodels_list, models_path):
    df_train_primary = pd.read_csv(models_path + 'predictions_train.csv')
    df_val_primary = pd.read_csv(models_path + 'predictions_validation.csv')
    df_train_primary.drop([col for col in df_train_primary.columns if "Unnamed" in col],
                          axis=1, inplace=True)
    df_val_primary.drop([col for col in df_val_primary.columns if "Unnamed" in col],
                        axis=1, inplace=True)

    for sub_model in submodels_list:
        if sub_model == 'over_under':
            continue
        rel_label1 = int(sub_model[-2])
        rel_label2 = int(sub_model[-1])
        df_sub_train = df_train_primary[(df_train_primary['label'] == rel_label1) |
                                        (df_train_primary['label'] == rel_label2)]
        df_sub_val = df_val_primary[(df_val_primary['label'] == rel_label1) |
                                    (df_val_primary['label'] == rel_label2)]
        df_sub_train = df_sub_train[['HADM_ID', 'label']]
        df_sub_val = df_sub_val[['HADM_ID', 'label']]
        df_sub_train['label'] = df_sub_train['label'].apply(lambda x: x - rel_label1)
        df_sub_val['label'] = df_sub_val['label'].apply(lambda x: x - rel_label1)
        df_sub_train.to_csv('Mortality_sub_' + sub_model + '_TRAIN.csv')
        df_sub_val.to_csv('Mortality_sub_' + sub_model + '_VAL.csv')
    return


def get_train_validation(sub_model, train_df_main, val_df_main):
    train_sub_tmp = pd.read_csv('Mortality_sub_' + sub_model + '_TRAIN.csv')
    val_sub_tmp = pd.read_csv('Mortality_sub_' + sub_model + '_VAL.csv')
    train_sub_elements = set(train_sub_tmp['HADM_ID'])
    val_sub_elements = set(val_sub_tmp['HADM_ID'])
    train_df = train_df_main[train_df_main.HADM_ID.isin(train_sub_elements)]
    val_df = val_df_main[val_df_main.HADM_ID.isin(val_sub_elements)]
    if sub_model != 'over_under':
        rel_label1 = int(sub_model[-2])
        train_df['label'] = train_df['Label'].apply(lambda x: x - rel_label1)
        val_df['label'] = val_df['Label'].apply(lambda x: x - rel_label1)
    return train_df, val_df


def train_sub_models(model_name, df_name, max_len, train_ratio,
                     text_feature, embeddings_version, embeddings_path,
                     config_subs, models_path, models_df):
    device = utils.find_device()
    submodels_list = ['over_under', 'model_01', 'model_12', 'model_23', 'model_34']
    create_over_under(models_path)
    create_sub_models_dfs(submodels_list, models_path)
    labels_num = 2
    train_df_main, val_df_main = utils.split_train_val(df_name, train_ratio)
    for sub_model in submodels_list:
        loss_function, _, dropout, epochs_num, threshold, \
        hid_dim_lstm, lin_output_dim, lr, batch_size, momentum = \
            utils.read_config_primary(config_subs, "train_sub")

        train_df, val_df = get_train_validation(sub_model, train_df_main, val_df_main)
        if sub_model == 'over_under':
            train_df, val_df = fix_values_over_under(train_df, val_df)
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
        utils.print_new_sub_model(sub_model)

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
                torch.save(model.state_dict(), models_path + sub_model + '//' + sub_model + '.pkl')
                best_val_acc = val_acc
                utils.save_predictions_to_df(predictions_dict_train, models_path, 'train', sub_model)
                utils.save_predictions_to_df(predictions_dict_val, models_path, 'validation', sub_model)
        utils.save_model(models_df, sub_model, hid_dim_lstm, labels_num, loss_function, dropout,
                         lin_output_dim, lr, epochs_num, batch_size, momentum, best_val_acc)
        utils.print_summary(models_path, sub_model, accuracy_list_val)
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


def read_model_parameters(models_df, model_name):
    hid_dim_lstm = int(models_df.loc[model_name].loc['hid_dim_lstm'])
    dropout = models_df.loc[model_name].loc['dropout']
    lin_output_dim = int(models_df.loc[model_name].loc['lin_output_dim'])
    lr = models_df.loc[model_name].loc['lr']
    batch_size = int(models_df.loc[model_name].loc['batch_size'])
    momentum = models_df.loc[model_name].loc['momentum']
    loss_function = nn.NLLLoss()  # conf_df.loc[model_name].loc['loss_function']
    labels_num = int(models_df.loc[model_name].loc['labels_num'])
    return hid_dim_lstm, loss_function, dropout, lin_output_dim, lr, \
           batch_size, momentum, labels_num


def load_model_parameters(model_name, sub_model, models_path, embeddings_path, models_df):
    device = utils.find_device()
    hid_dim_lstm, loss_function, dropout, lin_output_dim, lr, batch_size, momentum, labels_num = \
        read_model_parameters(models_df, sub_model)
    model = get_model(model_name, embeddings_path,
                      hid_dim_lstm, loss_function,
                      labels_num, dropout, lin_output_dim)
    state_dict = torch.load(models_path + '//' + sub_model + '.pkl', map_location=device)
    model.load_state_dict(state_dict)
    return model


def classify_sub_models(model_name, max_len, text_feature, embeddings_version, embeddings_path, models_path):
    batch_size = 32
    device = utils.find_device()
    models = {'over_under': {}, 'model_01': {},
                   'model_12': {}, 'model_23': {}, 'model_34': {}}
    models_df = pd.read_csv(models_path + 'models_df.csv', index_col=0)
    test_df = pd.read_csv('MORTALITY_TEST.csv')
    test_dataloader = utils.create_dataloaders_test(test_df, embeddings_version,
                                                    max_len, text_feature, batch_size=batch_size)
    dummy_labels = torch.tensor([1 for elem in range(batch_size)]).to(device, dtype=torch.long)
    for sub_model in models.keys():
        # current_df = modify_labels_for_sub(sub_model, test_df)
        print(sub_model)
        model_path_local = models_path + sub_model
        model = load_model_parameters(model_name, sub_model, model_path_local, embeddings_path, models_df)
        model.to(device)
        model.eval()
        predictions_dict_test = dict()
        for batch_idx, batch in enumerate(test_dataloader):
            input_ids = batch[0].to(device, dtype=torch.long)
            masks = batch[1].to(device, dtype=torch.long)
            hadm_id = batch[3]
            with torch.no_grad():
                _, predictions, probabilities = model(input_ids, masks, dummy_labels, calc_loss=False)
                predictions_dict_test = utils.update_predictions_dict(predictions_dict_test,
                                                                      hadm_id, predictions,
                                                                      probabilities, dummy_labels)
        utils.save_predictions_to_df(predictions_dict_test, models_path, 'test', sub_model)
    print("done")
    return models_df

