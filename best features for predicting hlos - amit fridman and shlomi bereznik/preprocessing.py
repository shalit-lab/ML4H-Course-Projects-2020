import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv('result.csv', delimiter=',')
    data = data[[
        'subject_id', 'hadm_id', 'admission_type', 'admission_location', 'admittime', 'dischtime', 'edregtime',
        'edouttime', 'insurance', 'religion', 'marital_status', 'ethnicity', 'los', 'gender', 'dob', 'first_careunit',
        'last_careunit', 'first_wardid', 'last_wardid','intime']]
    print(data.isnull().sum())
    data['dischtime'] = pd.to_datetime(data['dischtime'])
    data['intime'] = pd.to_datetime(data['intime'])
    data['admittime'] = pd.to_datetime(data['admittime'])
    data['dob'] = pd.to_datetime(data['dob'])
    data['edregtime'] = pd.to_datetime(data['edregtime'])
    data['edouttime'] = pd.to_datetime(data['edouttime'])
    #preprocessing all the date and time variables.
    data['hlos'] = data['dischtime'].sub(data['admittime'], axis=0)/ np.timedelta64(1, 'D')
    data['age'] = data['admittime'].sub(data['dob'], axis=0) / np.timedelta64(1, 'Y')
    data['edwait'] = data['edouttime'].sub(data['edregtime'], axis=0) / np.timedelta64(1, 'h')
    data['admitmonth']=data['admittime'].dt.month
    data['admittimeday']=data['admittime'].dt.hour/6
    data['admittimeday']=data['admittimeday'].astype('int32')
    data['admitseason']=0
    data.loc[(data['admitmonth']>2) & (data['admitmonth']<6),'admitseason']=1
    data.loc[(data['admitmonth'] >= 6) & (data['admitmonth'] <= 8), 'admitseason'] = 2
    data.loc[(data['admitmonth'] >= 9) & (data['admitmonth'] <= 11), 'admitseason'] = 3
    data['age'] = data['age'].astype('int32')
    data=data.dropna(axis=0,subset=['dischtime'])
    data=data.fillna(0)

    data=data[data['age']>=0]

    data=data.sort_values(by='intime')
    data_val=data.values

    #combinnig different visits to the icu within the same hospital admission.
    admission_data={}
    for i, row in enumerate(data_val):
        visit_id=row[1]
        los=row[12]
        first_careunit=row[15]
        last_careunit=row[16]
        first_wardid=row[17]
        last_wardid = row[18]
        if visit_id not in admission_data:
            admission_data[visit_id]=[]
        temp={}
        temp['los'] = los
        temp['first_careunit'] = first_careunit
        temp['last_careunit'] = last_careunit
        temp['first_wardid'] = first_wardid
        temp['last_wardid'] = last_wardid
        admission_data[visit_id].append(temp)

    for visit_id in admission_data:
        los=0
        for part in admission_data[visit_id]:
            los+=part['los']
        first_unit=admission_data[visit_id][0]['first_careunit']
        last_unit=admission_data[visit_id][-1]['last_careunit']
        first_ward = admission_data[visit_id][0]['first_wardid']
        last_ward = admission_data[visit_id][-1]['last_wardid']
        data.loc[data['hadm_id']==visit_id,'los']=los
        data.loc[data['hadm_id'] == visit_id, 'first_careunit'] = first_unit
        data.loc[data['hadm_id'] == visit_id, 'last_careunit'] = last_unit
        data.loc[data['hadm_id'] == visit_id, 'first_wardid'] = first_ward
        data.loc[data['hadm_id'] == visit_id, 'last_wardid'] = last_ward

    #transform the categorical data into numerical values
    data=data.drop(['intime','edouttime','edregtime','dischtime','dob'],axis=1)
    data['insurance'] = data['insurance'].astype('category').cat.codes
    data['gender'] = data['gender'].astype('category').cat.codes
    data['first_careunit'] = data['first_careunit'].astype('category').cat.codes
    data['last_careunit'] = data['last_careunit'].astype('category').cat.codes
    data['admission_type'] = data['admission_type'].astype('category').cat.codes
    data['admission_location'] = data['admission_location'].astype('category').cat.codes
    data['religion'] = data['religion'].astype('category').cat.codes
    data['marital_status'] = data['marital_status'].astype('category').cat.codes
    data['ethnicity'] = data['ethnicity'].astype('category').cat.codes
    data=data.drop_duplicates()

    tests=pd.read_csv('tests.csv')
    tests['charttime'] = pd.to_datetime(tests['charttime'])
    tests=tests.sort_values(by='charttime')

    tests_dict={}
    tests_list=[]
    for i in tests.index: #create dict of tests for each of the admissions
        visit_id=tests.loc[i,'hadm_id']
        if visit_id not in tests_dict:
            tests_dict[visit_id]={}
        label=tests.loc[i,'label']
        if label not in tests_list:
            tests_list.append(label)
        timedif=(tests.loc[i,'charttime']-data.loc[data['hadm_id']==visit_id,'admittime'])/ np.timedelta64(1, 'D')
        print(timedif)
        if timedif.values<=2:
            if label not in tests_dict[visit_id]:
                tests_dict[visit_id][label]=[]
            tests_dict[visit_id][label].append(tests.loc[i,'value'])


    for label in tests_list:
        data[label]=None

    for visit in tests_dict: #add the tests to the dataset
        for t in tests_dict[visit]:
            data.loc[data['hadm_id']==visit,t]=tests_dict[visit][t][-1]
    data.to_csv('data.csv',index=False)
    print(data)
