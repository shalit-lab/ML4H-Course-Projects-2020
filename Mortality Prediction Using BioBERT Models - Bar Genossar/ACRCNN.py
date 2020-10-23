import pandas as pd


def create_RCNN_dict(models_path):
    df = pd.read_csv(models_path + 'predictions_test.csv')
    RCNN_dict = dict(zip(df.HADM_ID, zip(df.prediction, df.probability)))
    return RCNN_dict


def create_predictions_dict(submodels_list, models_path):
    precictions_dict = dict()
    for sub_model in submodels_list:
        df = pd.read_csv(models_path + 'predictions_test_' + sub_model + '.csv')
        if sub_model != 'over_under':
            rel_label1 = int(sub_model[-2])
            df['prediction'] = df['prediction'].apply(lambda x: x + rel_label1)
        sub_dict = dict(zip(df.HADM_ID, zip(df.prediction, df.probability)))
        precictions_dict[sub_model] = sub_dict
    return precictions_dict


def find_sub_dict(over_under_call, prediction):
    if prediction == 0 and over_under_call == 1:
        return None
    elif prediction == 4 and over_under_call == 0:
        return None
    elif over_under_call == 0:
        rel_label2 = int(prediction) + 1
        return 'model_' + str(prediction) + str(rel_label2)
    else:
        rel_label1 = int(prediction) - 1
        return 'model_' + str(rel_label1) + str(prediction)


def ACRCNN_alg(hadm_id, RCNN_dict, predictions_dict, threshold, threshold2=0.5):
    prediction, probability = RCNN_dict[hadm_id]
    if probability > threshold:
        return prediction
    else:
        over_under_call = predictions_dict['over_under'][hadm_id][0]
        rel_sub_dict = find_sub_dict(over_under_call, prediction)
        if rel_sub_dict is None:
            return prediction
        else:
            if predictions_dict[rel_sub_dict][hadm_id][1] > threshold2:
                return predictions_dict[rel_sub_dict][hadm_id][0]
            else:
                return prediction


def run_test(models_path):
    submodels_list = ['over_under', 'model_01', 'model_12', 'model_23', 'model_34']
    RCNN_dict = create_RCNN_dict(models_path)
    precictions_dict = create_predictions_dict(submodels_list, models_path)
    test_df = pd.read_csv('MORTALITY_TEST.csv')
    test_df = test_df[['HADM_ID', 'Label']]
    thresholds_list = [0.05*i for i in range(1, 20)]
    best_acc, best_threshold = 0.0, 0.0
    results = []
    for threshold in thresholds_list:
        test_df['final_prediction'] = test_df['HADM_ID'].apply(lambda hadm_id: ACRCNN_alg(hadm_id, RCNN_dict,
                                                                                          precictions_dict,
                                                                                          threshold))
        accuracy = (test_df['final_prediction'] == test_df['Label']).sum() / len(test_df)
        results.append(round(accuracy, 5))
        if accuracy > best_acc:
            test_df.to_csv(models_path + 'final_predictions.csv')
            best_threshold = threshold
            best_acc = accuracy
        print("ACRCNN accuracy for threshold =", round(threshold, 2), "is: ", round(accuracy, 4))
    print(results)
    print("best threshold: ", best_threshold)
    print("best accuracy: ", best_acc)
    print("Done")


