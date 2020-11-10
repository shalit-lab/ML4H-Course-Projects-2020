from hrv.rri import RRi
import load_sigs
from time import time
import random
from datetime import timedelta as delta
from datetime import datetime as dt
import numpy as np
import pickle
from tqdm import tqdm
from hrv.filters import moving_average, threshold_filter
from statistics import mode, stdev, variance
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt


def check_equal_elem(iterator):
    """
    :param iterator: iterable
    :return: are all elements in the iterable equal
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def load_elem(path):
    """
    loads pickle file from path
    :param path: path to pickle file
    :return: loaded file
    """
    with open(path, "rb") as f:
        return pickle.load(f)


class Waveforms:
    """
    handles all extraction and pre-processing of health-signals
    """
    def __init__(self):
        self.records = {}
        self.patients = None
        self.channels = [0]
        self.hr = {}

    def get_records(self, path=None, patients=None, wf_stays=None, n_stays=None, only_hr=True, save=False):
        """
        extracts waveforms and saves to path
        :param path: path to save the file
        :param patients: patients for which to extract the data
        :param wf_stays: relevant ECG stay_ids for the patients
        :param n_stays: relevant Numeric(e.g HR, PULSE) stay_ids for the patients
        If wf_stays or n_stays are not available, extracts all data matched for the patient
        :param only_hr: whether to include ECG signals
        :param save: whether to save the loaded records to a pickle file
        """
        if not patients:
            raise IndexError("no patients to get records for")
        self.hr = {p: {} for p in patients}
        for patient in patients:
            if not n_stays:
                numeric_stays = load_sigs.get_icu_visits(patient, numerics=True)
            else:
                numeric_stays = [patient + "-" + s for s in n_stays]
            if not only_hr:
                self.records = {p: {} for p in patients}
                if not wf_stays:
                    wf_stays = load_sigs.get_icu_visits(patient, numerics=False)
                else:
                    wf_stays = [patient + "-" + s for s in wf_stays]
                for stay_id in wf_stays:
                    self.records[patient][stay_id[8:]] = load_sigs.get_stay_wf(stay_id, patient, self.channels)
            for stay_id in numeric_stays:
                self.hr[patient][stay_id[8:]] = load_sigs.get_stay_hr(stay_id, patient)
        if save:
            if not path:
                raise(NameError('Enter Valid Path to save the records'))
            with open(path, "wb") as f:
                pickle.dump(self, f)

    def combine_stay_signals(self, patient_id: str, in_time: str, threshold: int = 4):
        """
        combines signals of different segments in the same stay
        :param patient_id: patient for which combine the signals
        :param in_time: starttime of the recording - unique identifier for the matched wf dataset
        :param threshold: number of hours to include
        :return: combined signal
        """
        year, month, day = [int(x) for x in in_time.split('-')[:3]]
        date = dt(year, month, day)
        waveforms = np.array([])
        fields_dict = {}
        length = 0
        first_itr = True
        for idx, sig_fields in self.records[patient_id][in_time].items():
            sig, fields = sig_fields
            fields_dict[idx] = fields
            date_time = dt.combine(date, fields['base_time'])
            if first_itr:
                last_sample = date_time + delta(seconds=(fields['sig_len'] / fields['fs']))
                first_itr = False
                np.append(waveforms, sig)
                length += (fields['sig_len'] / fields['fs']) / 3600
                continue
            else:
                if (date_time - last_sample).total_seconds() < -1:  # error margin
                    date_time += delta(days=1)
                waveforms = np.append(waveforms, np.array([[0] * int((date_time - last_sample).total_seconds())]))
                length += (date_time - last_sample).total_seconds() / 3600
                if length >= threshold:
                    waveforms = load_sigs.truncate_signal(waveforms, 4, fields['fs'])
                    break
                last_sample = date_time + delta(seconds=(fields['sig_len'] / fields['fs']))
                waveforms = np.append(waveforms, sig)
                length += (fields['sig_len'] / fields['fs']) / 3600
        comb_fields = self.sum_fields(fields_dict, length)
        waveforms[np.isnan(waveforms)] = 0
        self.records[patient_id][in_time] = waveforms, comb_fields

    @staticmethod
    def sum_fields(fields_dict, length):
        """
        updates metadata about the signal
        :param fields_dict: metadata of the signal
        :param length: new length
        :return: updated metadata dict
        """
        fields = {}
        if not check_equal_elem([fields_dict[x]['fs'] for x in fields_dict]):
            raise Exception("different sampling frequencies")
        fields['fs'] = fields_dict['0001']['fs']
        fields['sig_len'] = int(length * 3600 * fields['fs'])
        return fields

    @staticmethod
    def preprocess_hr_signal(raw_signal, fs, sig_len, desired_fs: int = 1, interval: int = 1):
        """
        correct sampling bias by making a frequency shift, moving average filter and threshold LPF
        :param raw_signal: signal to be preprocessed
        :param fs: the raw_signal's sampling frequency
        :param sig_len: raw signal's number of samples
        :param desired_fs: desired sampling frequency
        :param interval: window for moving average filter
        :return: processed signal
        """
        # interval in hours
        signal = Waveforms.frequency_shift(raw_signal, sig_len, fs, desired_fs)
        signal = pos_sig(signal)
        if np.isnan(signal).all():
            return np.nan
        signal = RRi(signal)
        window = desired_fs * interval * 3600
        return threshold_filter(moving_average(signal, window), threshold='strong')

    @staticmethod
    def get_ecg_feature(signal: np.array):
        """
        Extract ECG variability features
        :param signal:
        :return: a feature vector
        """
        return [signal.max(), signal.min(), signal.mean(), np.median(signal), mode(signal), stdev(signal),
                variance(signal), signal.max() - signal.min(), kurtosis(signal), skew(signal),
                np.mean(signal ** 2), np.mean([signal[i] * signal[i + 1] for i, _ in
                                               enumerate(signal[:-1])])]

    @staticmethod
    def get_features(signal: RRi):
        """
        Extract HR/PULSE variability features
        :param signal:
        :return: a feature vector
        """
        return [np.max(signal), np.min(signal), np.mean(signal), np.median(signal), mode(signal), stdev(signal),
                variance(signal), signal.max() - signal.min(), kurtosis(signal.rri), skew(signal.rri),
                np.sum(signal.rri ** 2), np.mean([signal.rri[i] * signal.rri[i + 1] for i, _ in
                                                  enumerate(signal.rri[:-1])])]

    @staticmethod
    def frequency_shift(signal, sig_len, sig_freq, desired_fs=1):
        """
        Using oversample/downsample methods adapt the singal's frequency to the desired
        :param signal: signal to be processed
        :param sig_len: number of samples
        :param sig_freq: original sampling frequency
        :param desired_fs: desired sampling frequency
        :return: processed signal in the desired frequency
        """
        time = sig_len / sig_freq
        samples = time / desired_fs - sig_len  # if positive need to oversample, if negative to downsample
        if samples > 0:
            return Waveforms.oversample(signal, round(time / desired_fs))
        elif samples < 0:
            return Waveforms.downsample(signal, round(time / desired_fs))
        return signal

    @staticmethod
    def oversample(signal, desired_len):
        """
        oversamples a signal to a desired length
        :param signal: signal
        :param desired_len: desired number of samples
        :return: oversampled signal
        """
        if len(signal) == 1:
            return np.concatenate((signal, np.random.normal(signal[0], 3, desired_len - 1)))
        while signal.shape[0] < desired_len:
            ind = np.array(random.sample(range(signal.shape[0] - 1),
                                         min(int(signal.shape[0] / 2), desired_len - signal.shape[0]))) + 1
            signal = np.insert(signal, ind, np.round((signal[ind - 1] + signal[ind]) / 2))
        return signal

    @staticmethod
    def downsample(signal, desired_len):
        """
        downsamples a signal to a desired length
        :param signal: signal
        :param desired_len: desired number of samples
        :return: downsampled signal
        """
        while signal.shape[0] > desired_len:
            ind = np.random.randint(signal.shape[0] - 1)
            signal[ind + 1] = int((signal[ind] + signal[ind + 1]) / 2)
            signal = np.delete(signal, ind)
        return signal


def pos_sig(signal: np.array):
    """
    remove improper values of the signal
    :param signal:
    :return: processed signal
    """
    indices = []
    for idx, val in enumerate(signal):
        if val <= 0 or np.isnan(val) or not val:
            indices.append(idx)
    return np.delete(signal, indices)


# if __name__ == "__main__":
#     data_path = "records.p"
#     features = []
#     try:
#         ecg = load_elem(data_path)
#     except FileNotFoundError:
#         ecg = Waveforms()
#         ecg.get_records("records.p", ['p000020', 'p000079'])
#     for patient, stay_sig in ecg.hr.items():
#         t = time()
#         stay_ = None
#         for stay, wave in stay_sig.items():
#             stay_ = stay
#             ecg.combine_stay_signals(patient, stay, 4)
#             wave['HR'] = ecg.preprocess_hr_signal(wave['HR'][0], wave['HR'][1]['fs'], wave['HR'][1]['sig_len'])
#             wave['PULSE'] = ecg.preprocess_hr_signal(wave['PULSE'][0], wave['PULSE'][1]['fs'],
#                                                      wave['PULSE'][1]['sig_len'])
#             t2 = time()
#             features.append(ecg.get_features(wave['HR']) + ecg.get_features(wave['PULSE']))
#             # features.append(ecg.get_features(wave['HR']) + ecg.get_features(wave['PULSE']) + ecg.get_ecg_feature(
#             #     ecg.records[patient][stay][0]))
#             print(f'features time: {time() - t2}')
#         print(f'stay: {stay_} - time: {time() - t}')
#     print(features)
