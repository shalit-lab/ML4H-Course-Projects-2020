import torch
import pandas as pd
import primary_networks
import sub_networks
import ACRCNN
from config import ConfigMain
from config import ConfigPrimary
from config import ConfigSubModel
import utils


def run_task(config, config_primary, config_subs, models_df):
    df_name, max_len, train_ratio, text_feature,\
        model_name, models_path, embeddings_version, \
        embeddings_path, task_type = utils.read_config_main(config)
    utils.check_input(model_name, task_type)
    if task_type == 'train_primary':
        models_df = primary_networks.train_primary(model_name, df_name, max_len, train_ratio,
                                                  text_feature, embeddings_version, embeddings_path,
                                                  config_primary, models_path, models_df)
        models_df.to_csv(models_path + 'models_df.csv')
    elif task_type == "train_sub":
        models_df = sub_networks.train_sub_models(model_name, df_name, max_len, train_ratio,
                                                  text_feature, embeddings_version,
                                                  embeddings_path, config_subs,
                                                  models_path, models_df)
        models_df.to_csv(models_path + 'models_df.csv')
    elif task_type == "classify_sub":
        sub_networks.classify_sub_models(model_name, max_len, text_feature,
                                         embeddings_version,
                                         embeddings_path,
                                         models_path)
    elif task_type == "test":
        ACRCNN.run_test(models_path)


def create_models_df():
    models_dict = {'primary': {}, 'over_under': {}, 'model_01': {},
                   'model_12': {}, 'model_23': {}, 'model_34': {}}
    columns_df = ['hid_dim_lstm', 'dropout', 'lin_output_dim', 'lr',
                  'epochs_num', 'batch_size', 'momentum', 'accuracy']
    for model_name in models_dict.keys():
        models_dict[model_name] = ['' for col_num in range(len(columns_df))]
    models_df = pd.DataFrame.from_dict(models_dict, orient='index')
    models_df.columns = columns_df
    return models_df


def main():
    config = ConfigMain()
    config_primary = ConfigPrimary()
    config_subs = ConfigSubModel()
    models_df = create_models_df()
    run_task(config, config_primary, config_subs, models_df)


if __name__ == '__main__':
    main()
