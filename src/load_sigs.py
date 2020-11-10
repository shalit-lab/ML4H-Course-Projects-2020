import wfdb
import time
import requests
from bs4 import BeautifulSoup
from typing import List
from urllib3.exceptions import MaxRetryError
import numpy as np
from tqdm import tqdm


def get_icu_visits(patient_id: str, numerics=False):
    """
    :param patient_id: patient for whom stays are being returned
    :param numerics: whether to include PULSE, HR other than ECG's
    :return: stay list
    """
    parent_dir = patient_id[:3]
    url = f"https://physionet.org/content/mimic3wdb-matched/1.0/{parent_dir}/{patient_id}/#files-panel"
    response = None
    sleep_period = 1
    while not response:
        try:
            response = requests.get(url, params={})
        except (ConnectionError, MaxRetryError, requests.exceptions.ConnectionError):
            time.sleep(sleep_period)
            sleep_period *= 2
    response_text = response.text
    soup = BeautifulSoup(response_text, 'html.parser')
    if not numerics:
        stays = [seg[:-5] for seg in [node.get('href').split('/')[-1] for node in soup.find_all('a') if
                                      node.get('href') and node.get('href').endswith('.hea') and
                                      node.get('href').startswith(patient_id)] if seg.startswith(parent_dir)]
    else:
        stays = [seg[:-5] for seg in [node.get('href').split('/')[-1] for node in soup.find_all('a') if
                                      node.get('href') and node.get('href').endswith('n.hea')] if
                 seg.startswith(parent_dir)]
    return stays


def get_wf_header(stay_id: str, pn_dir: str):
    wf = None
    sleep_period = 1
    while not wf:
        try:
            wf = wfdb.rdheader(stay_id, pn_dir=pn_dir)
        except (ConnectionError, MaxRetryError, requests.exceptions.ConnectionError):
            time.sleep(sleep_period)
            sleep_period *= 2
    return wf


def get_wf_sample(sample_id: str, pn_dir: str, channels: List[int] = None):
    sig = np.array([])
    fields = np.array([])
    sleep_period = 1
    while sig.size == 0:
        try:
            sig, fields = wfdb.rdsamp(sample_id, pn_dir=pn_dir, channels=channels)
        except (ConnectionError, MaxRetryError, requests.exceptions.ConnectionError):
            sleep_period *= 2
            time.sleep(sleep_period)
    return sig, fields


def get_stay_wf(stay_id: str, patient_id: str, channels=None, threshold=4):
    """
    get the first hours of ECG data for a patient
    :param stay_id: the patient's stay
    :param patient_id: relevant patient
    :param channels: which channels to include
    :param threshold: number of hours
    :return: dict of stay-> signals
    """
    wf = get_wf_header(stay_id, f"mimic3wdb-matched/1.0/{patient_id[:3]}/{patient_id}")
    records = {}
    channels = [0]
    length = 0
    for rec in tqdm(wf.seg_name, desc=f"Segments of {patient_id}", leave=True):
        stay = rec.split('_')
        if len(stay) == 1 or stay[1] == 'layout':
            continue
        if stay[1] not in records:
            records[stay[1]] = {}
        sig, fields = get_wf_sample(rec, f"mimic3wdb-matched/1.0/{patient_id[:3]}/{patient_id}", channels)
        records[stay[1]] = sig, fields
        length += (fields['sig_len'] / fields['fs']) / 3600  # minute to hour
        if length >= threshold:
            break
    return records


def get_stay_hr(stay_id: str, patient_id: str, threshold: int = 4, vitals: List[str] = ('HR', 'PULSE')):
    """
    get the first hours of HR and PULSE records
    :param stay_id: the patient's stay
    :param patient_id: relevant patient
    :param threshold: number of hours
    :param: vitals: which numeric vitals to include
    :return: dict of stay-> signals
    """
    sig, fields = get_wf_sample(f"{stay_id}n", f"mimic3wdb-matched/1.0/{patient_id[:3]}/{patient_id}")
    records = {}
    for item in tqdm(vitals, desc="vitals", leave=True):
        if item in fields['sig_name']:
            records[item] = truncate_signal(sig[:, fields['sig_name'].index(item)], threshold, fields['fs']), fields
    return records


def truncate_signal(signal: np.array, threshold: int = 4, fs: int = 125):
    """
    truncate a signal to a given number of hours
    :param signal: the signal to be truncated
    :param threshold: length of signal
    :param fs: signal sampling frequency
    :return: truncated signal
    """
    return signal[:-int(len(signal) - threshold * fs * 3600)]

# if __name__ == '__main__':
#     wf = get_wf_sample('p000470-2132-04-01-19-56n')
#     wf = get_wf_sample('p000470-2132-04-01-19-56n', pn_dir=f"mimic3wdb-matched/1.0/{'p000470'[:3]}/{'p000470'}")
#     print(wf)
#     patients = ['p000020', 'p000079']
#     channels = [0]
#     records = {patient: {} for patient in patients}
#     progress = tqdm(patients, desc="Patients", leave=True)
#     for patient in progress:
#         patient_stays = get_icu_visits(patient)
#         for stay_id in tqdm(patient_stays, desc=f"stays of {patient}", leave=True):
#             records[patient][stay_id[8:]] = get_stay_wf(stay_id, patient, channels)
#     with open("records.p", "wb") as f:
#         pickle.dump(records, f)
