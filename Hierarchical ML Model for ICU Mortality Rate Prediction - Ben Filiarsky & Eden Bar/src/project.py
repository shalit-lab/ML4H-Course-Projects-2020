from load_data import LoadData
import pandas as pd
import hr_analysis as hr
import load_sigs
import numpy as np
import os
from datetime import datetime
import queries
import pickle
import time
from random import sample, seed
from sklearn.model_selection import train_test_split
from prediction import Model
from matplotlib import pyplot as plt

seed(54)


def pad_zeros(patient: str):
    """
    creates a subject_id representation as in the matched-wf dataset
    :param patient: subject_id
    :return: subject_id
    """
    return "p" + (6 - len(patient)) * "0" + patient


def create_matching(admissions_path, wf_path, numerics=False):
    """

    :param admissions_path: path to all admissions
    :param wf_path: path to matching
    :param numerics: whether include a numeric signal matching
    :return: numeric or ECG matching dataframe
    """
    records_wf = pd.read_csv(wf_path, header=None).to_numpy().flatten()
    if numerics:
        records_wf = [x[12:-1] for x in records_wf]
    else:
        records_wf = [x[12:] for x in records_wf]
    patients_wf = pd.DataFrame.from_dict({'SUBJECT_ID': [int(x[1:7]) for x in records_wf],
                                          'STARTTIME': [
                                              datetime(int(x[8:12]), month=int(x[13:15]), day=int(x[-8:-6]),
                                                       hour=int(x[-5:-3]), minute=int(x[-2:])) for x
                                              in records_wf]})
    admissions = pd.read_csv(admissions_path)
    admissions = admissions[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']]
    admissions.sort_values('SUBJECT_ID', inplace=True)
    admissions['ADMITTIME'] = admissions['ADMITTIME'].apply(pd.to_datetime)
    admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'], infer_datetime_format=True)
    matching = pd.merge(patients_wf, admissions, how='inner', left_on=['SUBJECT_ID'],
                        right_on=['SUBJECT_ID'])
    indices = []
    for idx, row in matching.iterrows():
        if (row['STARTTIME'] < row['ADMITTIME']) or (row['STARTTIME'] > row['DISCHTIME']):
            indices.append(idx)
    matching.drop(indices, axis=0, inplace=True)
    matching.reset_index(inplace=True, drop=True)
    matching.sort_values('SUBJECT_ID')
    if numerics:
        matching.to_csv("matching_numerics.csv")
    else:
        matching.to_csv("matching.csv")


def cardiac_patients(query_proccesor: LoadData, admissions: str, gender: str, path: str, cardiac=True):
    """
    Extract patients using a filtering scheme
    :param query_proccesor: sql handler
    :param admissions: relevant admissions
    :param gender: desired gender for the patients
    :param path: path to already created file
    :param cardiac: cardiac or non-cardiac patients
    :return: patient dataframe with demographics
    """
    try:
        patients = pd.read_csv(path)
    except FileNotFoundError:
        patients = query_proccesor.query_db(queries.patient_selection(admissions, gender, cardiac))
        patients.to_csv(path)
    return patients


def waveform_features(patients_adm, ecg_matching=None, num_matching=None):
    """
    return waveform features for given patient admissions
    :param patients_adm: list of admissions
    :param ecg_matching: ECG matching table
    :param num_matching: Numeric matching table
    :return: feature array
    """
    card_features = {}
    for idx, row in patients_adm.iterrows():
        print(f'patient {idx + 1} out of {patients_adm.shape[0]} patients')
        processor = hr.Waveforms()
        hadm_id = row['HADM_ID']
        starttime_wf = [x[:10] + "-" + x[-8:-6] + "-" + x[-5:-3] for x in
                        ecg_matching[ecg_matching['HADM_ID'] == hadm_id]['STARTTIME']]
        starttime_n = [x[:10] + "-" + x[-8:-6] + "-" + x[-5:-3] for x in
                       num_matching[num_matching['HADM_ID'] == hadm_id]['STARTTIME']]
        patient = pad_zeros(str(row['SUBJECT_ID']))
        processor.get_records(patients=[patient], wf_stays=starttime_wf, n_stays=starttime_n, only_hr=True, save=False)
        for patient, stay_sig in processor.hr.items():
            patient_features = []
            # for stay_wf in processor.records[patient]:
            # processor.combine_stay_signals(patient, stay_wf, 4)
            # ecg_features = {stay_wf: processor.get_ecg_feature(processor.records[patient][stay_wf][0])}
            for stay, wave in stay_sig.items():
                try:
                    if 'HR' in wave:
                        wave['HR'] = processor.preprocess_hr_signal(wave['HR'][0], wave['HR'][1]['fs'],
                                                                    wave['HR'][1]['sig_len'])
                        if wave['HR']:
                            hr_features = processor.get_features(wave['HR'])
                        else:
                            hr_features = [np.nan] * 12
                    if 'PULSE' in wave:
                        wave['PULSE'] = processor.preprocess_hr_signal(wave['PULSE'][0], wave['PULSE'][1]['fs'],
                                                                       wave['PULSE'][1]['sig_len'])
                        if wave['PULSE']:
                            pulse_features = processor.get_features(wave['PULSE'])
                        else:
                            pulse_features = [np.nan] * 12
                    else:
                        patient_features.append(np.array([np.nan] * 24))
                        continue
                    patient_features.append(np.array(hr_features + pulse_features))
                except Exception as e:
                    with open(f"hr_features_{time.time()}", "wb") as f:
                        pickle.dump(card_features, f)
                    with open("Exception_log.txt", "w") as f2:
                        f2.write(str(e))
            card_features[patient] = np.nanmean(patient_features, axis=0)
    return card_features


def create_clinical_features(referance_range, clinical_results):
    """
    given lab results and referance ranges create fetures for all patients in the lab results
    :param referance_range: normal ranges of
    :param clinical_results: lab results
    :return: clinical features array
    """
    mean_values = clinical_results.groupby(['label']).mean()['value']
    print(mean_values)
    stdev = clinical_results.groupby(['label']).std()['value']
    normalizer = [x[1] - x[0] if x[1] - x[0] > 0 else 1 for i, x in referance_ranges.items()]
    labs = sorted(list(x for x in clinical_results['label'].unique()))
    label_to_idx = {labs[idx]: idx for idx in range(len(labs))}
    idx_to_label = {idx: labs[idx] for idx in range(len(labs))}
    feature_size = len(referance_range)
    patient_labs = clinical_results.groupby(['subject_id', 'label'])['value'].mean().to_frame(
        name='mean').reset_index().groupby('subject_id')
    patient_features = {}
    for patient, data in patient_labs:
        patient_features[patient] = np.empty(feature_size)
        patient_features[patient][:] = np.nan
        for _, row in data.iterrows():
            patient_features[patient][label_to_idx[row['label']]] = row['mean']
        for i in range(feature_size):
            if np.isnan(patient_features[patient][i]):
                patient_features[patient][i] = np.random.normal(mean_values[i], stdev[i])
        # create features for distances from the normal range
        for i in range(feature_size):
            scale = normalizer[i]
            if patient_features[patient][i] - referance_ranges[idx_to_label[i]][0] < 0:
                patient_features[patient] = np.append(patient_features[patient], (
                        patient_features[patient][i] - referance_ranges[idx_to_label[i]][0]) / scale)
            elif patient_features[patient][i] - referance_ranges[idx_to_label[i]][1] > 0:
                patient_features[patient] = np.append(patient_features[patient], (
                        patient_features[patient][i] - referance_ranges[idx_to_label[i]][1]) / scale)
            else:
                patient_features[patient] = np.append(patient_features[patient], 0)
    return patient_features


def mortality_split(patients_adm, dead_patients):
    """
    split dead and alive patients
    :param patients_adm: relevant admissions
    :param dead_patients: all dead patients in MIMIC III
    :return: alive and dead patient dataframes
    """
    dead = pd.merge(patients_adm, dead_patients, how="inner", left_on=["subject_id"], right_on=["subject_id"])
    dead_patients = set(dead['subject_id'])
    alive_patients = set(patients_adm['subject_id']).difference(dead_patients)
    alive = patients_adm[patients_adm['subject_id'].isin(alive_patients)]
    alive['mortality'] = 0
    dead['mortality'] = 1
    return alive, dead


def upsample(dead_cardiac, dead_non_cardiac, desired_len=None):
    """
    upsamples cardiac and non-cardiac groups of deceased patients to handle class imbalance using a gaussian data
    generating process
    :param dead_cardiac: dead cardiac patients
    :param dead_non_cardiac: dead non-cardiac patients
    :param desired_len: desired number of patients
    :return: upsampled cardiac and non-cardiac deceased patients
    """
    card_up, ncard_up = dead_cardiac, dead_non_cardiac
    mean_card = np.mean(dead_cardiac, axis=0)
    var_card = np.var(dead_cardiac, axis=0)
    mean_ncard = np.mean(dead_non_cardiac, axis=0)
    var_ncard = np.var(dead_non_cardiac, axis=0)
    if not desired_len:
        desired_len = 2 * (len(dead_cardiac) + len(dead_non_cardiac))
    while len(card_up) + len(ncard_up) < desired_len:
        non_card = np.random.binomial(1, 1 - len(ncard_up) / desired_len, 1)
        if non_card:
            ncard_up = np.vstack([ncard_up,
                                  np.random.multivariate_normal(mean_ncard, np.diag(var_ncard), 1)])
        else:
            card_up = np.vstack([card_up,
                                 np.random.multivariate_normal(mean_card, np.diag(var_card), 1)])
    return card_up, ncard_up


def train_test_cardiac(patient_adm, split_ratio=0.7, threshold=500):
    """
    Train-Test split for data of the cardiac prediction task
    :param patient_adm: relevant admissions
    :param split_ratio: train percentage
    :param threshold: maximum number of patients
    :return:
    """
    train = patient_adm.sample(int(threshold * split_ratio))
    test = patient_adm[~patient_adm['subject_id'].isin(train['subject_id'])].sample(int(threshold * (1 - split_ratio)))
    return train.drop(columns=['card_stat']), train[['subject_id', 'card_stat']], test.drop('card_stat', axis=1), test[
        ['subject_id', 'card_stat']]


def train_test_mortality(patient_adm, mortality, split_ratio=0.7):
    """
    Train-Test split for data of the mortality prediction task
    :param patient_adm: relevant admissions
    :param mortality: all dead patients in the MIMIC III
    :param split_ratio: train percentage
    :return:
    """
    alive, dead = mortality_split(patient_adm, mortality)
    dead_alive_ratio = dead.shape[0] / alive.shape[0]
    print(dead_alive_ratio)
    train_dead = dead.sample(int(dead.shape[0] * split_ratio))
    train_alive = dead.sample(int(alive.shape[0] * split_ratio))
    test_dead = dead[~dead['subject_id'].isin(train_dead['subject_id'])]
    test_alive = dead[~dead['subject_id'].isin(train_alive['subject_id'])]
    train = pd.concat(train_alive, train_dead)
    test = pd.concat(test_alive, test_dead)
    return train.drop('mortality'), train[['subject_id', 'mortality']], test.drop('mortality'), test[
        ['subject_id', 'mortality']]


def create_datasets(load_d: LoadData):
    """
    loads already created important files
    :param load_d: query processor in case of a missing file
    :return: loaded objects
    """
    try:
        wf_matching = pd.read_csv('matching.csv')
    except FileNotFoundError:
        create_matching('admissions.csv', "RECORDS-waveforms.csv")
        wf_matching = pd.read_csv('matching.csv')
    try:
        numeric_matching = pd.read_csv('matching_numerics.csv')
    except FileNotFoundError:
        create_matching('admissions.csv', "RECORDS-numerics.csv", numerics=True)
        numeric_matching = pd.read_csv('matching.csv')
    try:
        mortality = pd.read_csv("in_hospital_mortality.csv")
    except FileNotFoundError:
        mortality = load_d.query_db(queries.in_hospital_mortality())
        mortality.to_csv("in_hospital_mortality.csv")
    try:
        clinical_features = pd.read_csv("features_clinical.csv")
    except FileNotFoundError as e:
        print(str(e))
        print("please create clinical features file for all patients")
    return wf_matching, numeric_matching, mortality, clinical_features


if __name__ == '__main__':
    load_d = LoadData('config_template.ini', 'mimic')
    wf_matching, numeric_matching, mortality, clinical_features = create_datasets(load_d)
    cardiac_codes = ['CMED', 'CSURG', 'TSURG', 'VSURG']
    wf_admissions = "(" + ", ".join([str(x) for x in wf_matching['HADM_ID']]) + ")"
    male_card = cardiac_patients(load_d, wf_admissions, 'M', 'male_card_patients.csv', cardiac=True)
    female_card = cardiac_patients(load_d, wf_admissions, 'F', 'female_card_patients.csv', cardiac=True)
    male_no_card = cardiac_patients(load_d, wf_admissions, 'M', 'male_no_card.csv', cardiac=False)
    female_no_card = cardiac_patients(load_d, wf_admissions, 'F', 'female_no_card.csv', cardiac=False)

    lab_codes = (211, 615, 676, 772, 773, 781, 789, 811, 821, 769, 770, 3837, 3835, 1321,
                 227429, 851, 834, 828, 223830, 1531, 198, 227010, 227443, 226760, 229759, 226766, 226765, 189, 2981,
                 226998, 4948, 225312, 225309, 225310)
    referance_ranges = dict(sorted({'Temperature C': (36, 37.5), 'Resp Rate (Total)': (12, 18), 'Platelets': (150, 350),
                                    'Glucose (70-105)': (70, 120), 'Magnesium (1.6-2.6)': (1.6, 2.3),
                                    'BUN (6-20)': (6, 20), 'Cholesterol (<200)': (0, 200),
                                    'Albumin (>3.2)': (3.2, 4), 'ALT': (0, 35),
                                    'AST': (0, 35), 'Alk. Phosphate': (36, 92), 'GCS Total': (15, 15),
                                    'ART BP Systolic': (90, 120), 'ART BP Diastolic': (60, 80),
                                    'ART BP mean': (70, 100), 'SaO2': (96, 100), 'Troponin-T': (0, 0.01),
                                    'PH (Arterial)': (7.35, 7.45), 'HCO3 (serum)': (24, 30), 'Lactic Acid': (0.5, 1),
                                    'Heart Rate': (60, 100), 'FiO2 (Analyzed)': (1, 1), 'Troponin': (0, 0.5)}.items()))

    male_tr_X, male_tr_Y, male_te_X, male_te_Y = train_test_cardiac(male_card, split_ratio=0.8)
    female_tr_X, female_tr_Y, female_te_X, female_te_Y = train_test_cardiac(female_card, split_ratio=0.8)
    male_tr_n_X, male_tr_n_Y, male_te_n_X, male_te_n_Y = train_test_cardiac(male_no_card, split_ratio=0.8)
    female_tr_n_X, female_tr_n_Y, female_te_n_X, female_te_n_Y = train_test_cardiac(female_no_card, split_ratio=0.8)
    train_card_X = pd.concat([male_tr_X, female_tr_X, male_tr_n_X, female_tr_n_X]).sample(400)
    train_card_y = pd.concat([male_tr_Y, female_tr_Y, male_tr_n_Y, female_tr_n_Y])
    test_card_X = pd.concat([male_te_X, female_te_X, male_te_n_X, female_te_n_X]).sample(100)
    test_card_y = pd.concat([male_te_Y, female_te_Y, male_te_n_Y, female_te_n_Y])
    train_hadm_ids = [x for x in train_card_X['hadm_id']]
    test_hadm_ids = [x for x in test_card_X['hadm_id']]

    matched_train_n = numeric_matching[numeric_matching['HADM_ID'].isin(train_hadm_ids)].reset_index()
    matched_train_n['STARTTIME'] = matched_train_n['STARTTIME'].apply(pd.to_datetime)
    matched_test_n = numeric_matching[numeric_matching['HADM_ID'].isin(test_hadm_ids)].reset_index()
    matched_test_n['STARTTIME'] = matched_test_n['STARTTIME'].apply(pd.to_datetime)

    """ Waveform Features Creation"""
    # matched_train_card_wf = wf_matching[wf_matching['HADM_ID'].isin(train_hadm_ids)]
    # matched_test_card_wf = wf_matching[wf_matching['HADM_ID'].isin(test_hadm_ids)]
    # wf_features_train = waveform_features(matched_train_n, wf_matching, numeric_matching)
    # wf_features_test = waveform_features(matched_test_n, wf_matching, numeric_matching)
    # with open("wf_features_train", "wb") as f:
    #     pickle.dump(wf_features_train, f)
    # with open("wf_features_test", "wb") as f2:
    #     pickle.dump(wf_features_test, f2)

    """Waveform dict join"""
    # with open("wf_features_train", "rb") as f:
    #     wf_features_train = pickle.load(f)
    # with open("wf_features_test", "rb") as f2:
    #     wf_features_test = pickle.load(f2)
    # with open("wf_dead", "rb") as f:
    #     dead_wf = pickle.load(f)
    # train_good = {int(x[1:]): wf_features_train[x] for x, y in wf_features_train.items() if ~np.isnan(y).all()}
    # test_good = {int(x[1:]): wf_features_test[x] for x, y in wf_features_test.items() if ~np.isnan(y).all()}

    # with open("numeric_wf_473_patients", "rb") as f:
    #     wf_data_473 = pickle.load(f)
    # dead_good = {int(x[1:]): dead_wf[x] for x, y in dead_wf.items() if ~np.isnan(y).all()}
    # good = {int(x): wf_data_473[x] for x, y in wf_data_473.items() if ~np.isnan(y).all()}
    # wf_data = {**good, **dead_good}

    """Cardiac Prediction"""
    wf_total = pd.read_csv("wf_total.csv")
    male_c, male_n, female_c, female_n = male_card['subject_id'].to_numpy(), male_no_card['subject_id'].to_numpy(), \
                                         female_card['subject_id'].to_numpy(), female_no_card['subject_id'].to_numpy()
    clinical_wf_idx = set(wf_total.columns).intersection(set(clinical_features.columns))
    data_clinical = clinical_features[[str(x) for x in clinical_wf_idx]]
    wf_df = pd.DataFrame.from_dict(wf_total)
    data_wf = wf_df[[x for x in clinical_wf_idx]]
    total_data = pd.concat([data_clinical, data_wf])
    patients_total = np.array([int(x) for x in total_data.columns if x != "Unnamed: 0"])
    data = []
    for _, x in enumerate(patients_total):
        data.append(np.append(np.array(total_data[str(x)]), x in male_c or x in female_c))
    cardiac_labeled = np.array(data)
    cardiac = cardiac_labeled[cardiac_labeled[:, -1] == 1][:, 46:-1]
    non_cardiac = cardiac_labeled[cardiac_labeled[:, -1] == 0][:, 46:-1]
    cardiac_plot = {'cardiac': len(cardiac), 'non-cardiac': len(non_cardiac)}
    plt.bar(list(cardiac_plot.keys()), list(cardiac_plot.values()))
    plt.title("Cardiac vs. Non-Cardiac Patients")
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(cardiac_labeled[:, :-1], cardiac_labeled[:, -1], test_size=0.25)
    model = Model('RandomForest')
    model.fit(X_train, y_train)
    options = ('precision', 'recall', 'F1', 'AUC')
    print(f'Cardiac prediction: {model.score(X_test, y_test, options)}')

    """Mortality prediction - direct"""
    data = []
    for idx, x in enumerate(patients_total):
        data.append(np.append(np.array(total_data[str(x)]),
                              np.array([x in male_c or x in female_c, x in mortality['subject_id']])))
    labeled_mortality = np.c_[np.array(data)[:, :46], np.array(data)[:, -2:]]
    dead_cardiac = labeled_mortality[labeled_mortality[:, -2] == 1]
    dead_cardiac = dead_cardiac[dead_cardiac[:, -1] == 1][:, :-2]
    dead_non_cardiac = labeled_mortality[labeled_mortality[:, -2] == 0]
    dead_non_cardiac = dead_non_cardiac[dead_non_cardiac[:, -1] == 1][:, :-2]
    dead_c_up, dead_nc_up = upsample(dead_cardiac, dead_non_cardiac)
    dead = np.c_[np.vstack([dead_c_up, dead_nc_up]), np.ones((len(dead_c_up) + len(dead_nc_up), 1))]
    alive = labeled_mortality[labeled_mortality[:, -1] == 0]
    alive = np.c_[alive[:, :-2], np.zeros((len(alive), 1))]
    dead_labeled = np.vstack([alive, dead])
    dead_plot = {'Dead': len(dead), 'Alive': len(alive)}
    plt.bar(list(dead_plot.keys()), list(dead_plot.values()))
    plt.title("Dead vs. Alive Patients")
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(dead_labeled[:, :-1], dead_labeled[:, -1], test_size=0.25)
    model = Model('RandomForest')
    model.fit(X_train, y_train)
    options = ('precision', 'recall', 'F1', 'AUC')
    print(f'non-hierarchical : {model.score(X_test, y_test, options)}')

    """Mortality Prediction - Hierarchical"""
    alive_cardiac = labeled_mortality[labeled_mortality[:, -2] == 1]
    alive_cardiac = alive_cardiac[alive_cardiac[:, -1] == 0]
    alive_non_cardiac = labeled_mortality[labeled_mortality[:, -2] == 0]
    alive_non_cardiac = alive_non_cardiac[alive_non_cardiac[:, -1] == 0]
    data_dead_alive_card = np.vstack([(np.c_[alive_cardiac[:, :-2], np.zeros((len(alive_cardiac), 1))]),
                                      np.c_[dead_c_up, np.ones((len(dead_c_up), 1))]])
    data_dead_alive_ncard = np.vstack([(np.c_[alive_non_cardiac[:, :-2], np.zeros((len(alive_non_cardiac), 1))]),
                                       np.c_[dead_nc_up, np.ones((len(dead_nc_up), 1))]])
    for dataset, option in zip((data_dead_alive_card, data_dead_alive_ncard), ('cardiac', 'non_cardiac')):
        X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1], dataset[:, -1], test_size=0.25)
        model = Model('RandomForest')
        model.fit(X_train, y_train)
        options = ('precision', 'recall', 'F1', 'AUC')
        print(f'{option}: {model.score(X_test, y_test, options)}')

    """Both clinical and waveform data"""
    # for x, y in train_good.items():
    #     if str(x) not in clinical_features.columns:
    #         continue
    #     y = np.append(y, clinical_features[str(x)])
    # for x, y in test_good.items():
    #     if str(x) not in clinical_features.columns:
    #         continue
    #     y = np.append(y, clinical_features[str(x)])
    # all = {}
    # for d in (train_good, test_good):
    #     for patient, data in d.items():
    #         all[patient] = data
    # df = pd.DataFrame.from_dict(all)
    # df.to_csv("card_clinical_features.csv")
    # with open("all_features", "wb") as a:
    #     pickle.dump(train_good, a)

    # card_train, no_card_train = {}, {}
    # card_test, no_card_test = {}, {}
    # dead, alive = {}, {}
    # labs_all = pd.read_csv("labs_all.csv")
    # with open("all_features", "rb") as f:
    #     print(pickle.load(f))

    """Load Clinical data for dead"""
    # try:
    #     clinical_dead = pd.read_csv("clinical_dead.csv")
    # except FileNotFoundError:
    #     clinical_dead = load_d.query_db(queries.clinical_features(lab_codes, dead))
    #     clinical_dead.to_csv("clinical_dead.csv")
    # print('dead_features')
    # dead_features = create_clinical_features(referance_ranges, clinical_dead)
    # with open("dead_clinical_features", "wb") as dd:
    #     pickle.dump(dead_features, dd)

    """Create Clinical Features"""
    # clinical_features_train = create_clinical_features(referance_ranges, features_mimic_train)
    # with open("clinical_features_train", "wb") as ct:
    #     pickle.dump(clinical_features_train, ct)
    # clinical_features_test = create_clinical_features(referance_ranges, features_mimic_test)
    # with open("clinical_features_test", "wb") as cte:
    #     pickle.dump(clinical_features_test, cte)
    # dead_features = create_clinical_features(referance_ranges, clinical_dead)
    # with open("dead_clinical_features", "wb") as dd:
    #     pickle.dump(dead_features, dd)

    "merge all clinical features"
    # clinical_features = {}
    # for d in (clinical_dead, clinical_test, clinical_train):
    #     for patient, data in d.items():
    #         clinical_features[patient] = data
    # df = pd.DataFrame.from_dict(clinical_features)
    # df.to_csv("clinical_features_all.csv")

    """Dead Patient Splitting"""
    # dead_wf_records = set(mortality['subject_id']).intersection(set(numeric_matching['SUBJECT_ID'])).difference(
    #     set(wf_data.keys()))
    # male_card_dead = male_card[male_card['subject_id'].isin(dead_wf_records)]
    # male_no_card_dead = male_no_card[male_no_card['subject_id'].isin(dead_wf_records)]
    # f_card_dead = female_card[female_card['subject_id'].isin(dead_wf_records)]
    # f_no_card_dead = female_no_card[female_no_card['subject_id'].isin(dead_wf_records)]
    # dead = sample(list(male_card_dead['hadm_id']), 50) + sample(list(f_card_dead['hadm_id']), 50) + \
    #        sample(list(male_no_card_dead['hadm_id']), 50) + sample(list(f_no_card_dead['hadm_id']), 50)
    # dead_indices = numeric_matching[numeric_matching['HADM_ID'].isin(dead)].reset_index()
    # dead_indices['STARTTIME'] = dead_indices['STARTTIME'].apply(pd.to_datetime)
    # patients = set(cardiac_stat).union(set([x for x in wf_data.columns if x != 'Unnamed: 0'])).union(
    #     dead_indices['SUBJECT_ID'])

    """Create Clinical Features"""
    # db_clinical = load_d.query_db(queries.clinical_features(lab_codes, patients))
    # print(db_clinical)
    # db_clinical.to_csv("db_clinical.csv")

    """Waveforms for dead"""
    # wf_dead_features = waveform_features(dead_indices, wf_matching, numeric_matching)
    # with open("dead_200_wf_features", "wb")as d:
    #     pickle.dump(wf_dead_features, d)

    """Check balance in data"""
    # for patient, data in features.iteritems():
    #     if patient == 'Unnamed: 0':
    #         continue
    #     if int(patient) in male_c or patient in female_c:
    #         card_train[patient] = data
    #     if int(patient) in mortality['subject_id']:
    #         dead[patient] = data
    #     elif int(patient) not in mortality['subject_id']:
    #         alive[patient] = data
    #     elif int(patient) in male_n or patient in female_n:
    #         no_card_train[patient] = data
    #     else:
    #         alive[patient] = data
    # for patient, data in wf_data.items():
    #     if patient in male_c or patient in female_c:
    #         card_test[patient] = data
    #     if patient in mortality['subject_id']:
    #         dead[patient] = data
    #     elif patient not in mortality['subject_id']:
    #         alive[patient] = data
    #     elif patient in male_n or patient in female_n:
    #         no_card_test[patient] = data
    # print(len(dead))
    # print(len(alive))
