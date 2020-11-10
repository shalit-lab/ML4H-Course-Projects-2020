def patient_selection(wf_adm, gender, cardiac):
    """
    Patient Filtering process
    :param wf_adm:
    :param gender: the desired gender of the patients
    :param cardiac: the desired cardiac status e.g does the patient have a cardial medical condition
    :return: dataframe with demographics about the chosen patients
    """
    if cardiac:
        return """
            WITH first_admission_time AS
            (
              SELECT
                  p.subject_id, p.dob, p.gender
                  , MIN (a.admittime) AS first_admittime
                  , AVG(ROUND( (cast(admittime as date) - cast(dob as date)) / 365.242,2) )
                      AS first_admit_age
              FROM mimiciii.patients p
              INNER JOIN mimiciii.admissions a
              ON p.subject_id = a.subject_id
              WHERE a.hadm_id in {}
              GROUP BY p.subject_id, p.dob, p.gender
              ORDER BY p.subject_id
            )
            , age AS
            (
              SELECT
                  subject_id, dob, gender
                  , first_admittime, first_admit_age
                  , CASE
                      -- all ages > 89 in the database were replaced with 300
                      -- we check using > 100 as a conservative threshold to ensure we capture all these patients
                      WHEN first_admit_age > 100
                          then '>89'
                      WHEN first_admit_age >= 18
                          THEN 'adult'
                      WHEN first_admit_age <= 1
                          THEN 'neonate'
                      ELSE 'middle'
                      END AS age_group
              FROM first_admission_time
            ),
            demographics AS
            (
              SELECT 
                    subject_id, hadm_id, insurance, language
                    , religion, marital_status
                    , CASE
                        WHEN a.ethnicity = 'ASIAN'
                            THEN 1
                        WHEN a.ethnicity IN ('BLACK', 'CARIBBEAN ISLAND')
                            THEN 2
                        WHEN a.ethnicity IN ('HISPANIC', 'SOUTH AMERICAN')
                            THEN 3
                        WHEN a.ethnicity IN ('WHITE', 'MIDDLE EASTERN', 'PORTUGUESE')
                            THEN 4
                        ELSE 0
                        END AS ethnicity_code
                FROM mimiciii.admissions a
                WHERE marital_status IS NOT NULL 
            ),
            cardiac_stat as (
                SELECT p.subject_id, serv.prev_service, serv.curr_service, serv.hadm_id,
                    CASE
                        WHEN serv.curr_service IN ('CMED', 'CSURG', 'VSURG', 'TSURG') 
                            OR serv.prev_service IN ('CMED', 'CSURG', 'VSURG', 'TSURG')
                                THEN 1
                        ELSE 0
                        END AS CARDIAC_PATIENT
                    FROM mimiciii.services as serv 
                    JOIN mimiciii.patients as p on serv.subject_id = p.subject_id
                    WHERE serv.curr_service IS NOT NULL OR serv.prev_service IS NOT NULL
            ),
            cardiac_p as (
                SELECT ct.subject_id, MAX(ct.hadm_id) as hadm_id, 
                CASE
                    WHEN SUM(ct.CARDIAC_PATIENT) >= 1
                        THEN 1
                    ELSE 0
                    END AS CARDIAC_PATIENT
                FROM cardiac_stat as ct 
                GROUP BY ct.subject_id
            ), 
            yescardiac as (
            SELECT cp.subject_id, cp.hadm_id
            FROM cardiac_p as cp
            WHERE cp.CARDIAC_PATIENT = 1
            )
            SELECT 
                yescardiac.subject_id, gender, first_admit_age, first_admittime, yescardiac.hadm_id, insurance, language
                ,religion, marital_status, ethnicity_code, TRUE as card_stat
            FROM (age inner join demographics as d on age.subject_id = d.subject_id) 
            INNER JOIN yescardiac ON yescardiac.subject_id = age.subject_id and yescardiac.hadm_id = d.hadm_id
            WHERE age_group = 'adult' 
            /*limit to one admission per patient or same patient some admissions, check offset/head to start from different 
            place
            */
            and gender = {} 
            and NOT(ethnicity_code = 0) 
            limit 1000
            """.format(wf_adm, f"'{gender}'")
    else:
        return """
            WITH first_admission_time AS
            (
              SELECT
                  p.subject_id, p.dob, p.gender
                  , MIN (a.admittime) AS first_admittime
                  , AVG(ROUND( (cast(admittime as date) - cast(dob as date)) / 365.242,2) )
                      AS first_admit_age
              FROM mimiciii.patients p
              INNER JOIN mimiciii.admissions a
              ON p.subject_id = a.subject_id
              WHERE a.hadm_id in {}
              GROUP BY p.subject_id, p.dob, p.gender
              ORDER BY p.subject_id
            )
            , age AS
            (
              SELECT
                  subject_id, dob, gender
                  , first_admittime, first_admit_age
                  , CASE
                      -- all ages > 89 in the database were replaced with 300
                      -- we check using > 100 as a conservative threshold to ensure we capture all these patients
                      WHEN first_admit_age > 100
                          then '>89'
                      WHEN first_admit_age >= 18
                          THEN 'adult'
                      WHEN first_admit_age <= 1
                          THEN 'neonate'
                      ELSE 'middle'
                      END AS age_group
              FROM first_admission_time
            ),
            demographics AS
            (
              SELECT 
                    subject_id, hadm_id, insurance, language
                    , religion, marital_status
                    , CASE
                        WHEN a.ethnicity = 'ASIAN'
                            THEN 1
                        WHEN a.ethnicity IN ('BLACK', 'CARIBBEAN ISLAND')
                            THEN 2
                        WHEN a.ethnicity IN ('HISPANIC', 'SOUTH AMERICAN')
                            THEN 3
                        WHEN a.ethnicity IN ('WHITE', 'MIDDLE EASTERN', 'PORTUGUESE')
                            THEN 4
                        ELSE 0
                        END AS ethnicity_code
                FROM mimiciii.admissions a
                WHERE marital_status IS NOT NULL 
            ),
            cardiac_stat as (
                SELECT p.subject_id, serv.prev_service, serv.curr_service, serv.hadm_id,
                    CASE
                        WHEN serv.curr_service IN ('CMED', 'CSURG', 'VSURG', 'TSURG') 
                            OR serv.prev_service IN ('CMED', 'CSURG', 'VSURG', 'TSURG')
                                THEN 1
                        ELSE 0
                        END AS CARDIAC_PATIENT
                    FROM mimiciii.services as serv 
                    JOIN mimiciii.patients as p on serv.subject_id = p.subject_id
                    WHERE serv.curr_service IS NOT NULL OR serv.prev_service IS NOT NULL
            ),
            cardiac_p as (
                SELECT ct.subject_id, MAX(ct.hadm_id) as hadm_id, 
                CASE
                    WHEN SUM(ct.CARDIAC_PATIENT) >= 1
                        THEN 1
                    ELSE 0
                    END AS CARDIAC_PATIENT
                FROM cardiac_stat as ct 
                GROUP BY ct.subject_id
            ),
            notcardiac as (
            SELECT cp.subject_id, cp.hadm_id
            FROM cardiac_p as cp
            WHERE cp.CARDIAC_PATIENT = 0
            )
            SELECT 
                notcardiac.subject_id, gender, first_admit_age, first_admittime, notcardiac.hadm_id, insurance, language, 
                religion, marital_status, ethnicity_code, FALSE as card_stat
            FROM (age inner join demographics as d on age.subject_id = d.subject_id) 
            INNER JOIN notcardiac ON notcardiac.subject_id = age.subject_id and notcardiac.hadm_id = d.hadm_id
            WHERE age_group = 'adult' 
            /*limit to one admission per patient or same patient some admissions, check offset/head to start from different 
            place
            */
            and gender = {} 
            and NOT(ethnicity_code = 0) 
            limit 1000
            """.format(wf_adm, f"'{gender}'")


def clinical_features(lab_codes, subject_ids):
    """
    lab results extraction
    :param lab_codes: the relevant labs codes
    :param subject_ids: patients for which the data is needed
    :return: df of lab results for the patients
    """
    subject_ids = "(" + ", ".join([str(x) for x in subject_ids]) + ")"
    return """
            select c.subject_id, c.hadm_id, c.itemid, c.value, c.charttime, d.label, d.unitname, d.category
            from mimiciii.chartevents as c
            inner join mimiciii.d_items as d on c.itemid = d.itemid 
            where c.itemid in {} and c.subject_id in {} and c.value IS NOT NULL
            """.format(lab_codes, subject_ids)


def in_hospital_mortality():
    """
    Extract all patients that died in hospital
    :return: df of subject_ids and hadm_id of deceased patients
    """
    return """
    SELECT adm.subject_id, adm.hadm_id
           FROM mimiciii.admissions as adm
           WHERE hospital_expire_flag = 1;
    """


