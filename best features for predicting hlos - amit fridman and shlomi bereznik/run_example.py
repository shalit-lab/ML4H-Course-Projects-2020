# /usr/bin/python

from load_data import LoadData


if __name__ == '__main__':
    query_mimic = """
        select *
        from mimiciii.admissions as a
        join mimiciii.patients as p on a.subject_id = p.subject_id 
        join mimiciii.icustays as i on a.hadm_id = i.hadm_id
        """

    load_d = LoadData('config_template.ini', 'mimic')

    print(load_d.query_db(query_mimic))
    load_d.query_and_save(query_mimic)

    query_mimic = """
            select c.subject_id,c.hadm_id,c.itemid,c.value,c.charttime,p.label
            from mimiciii.chartevents as c
            join mimiciii.d_items as p on c.itemid = p.itemid 
            where c.itemid in (211,615,676,772,773,781,789,811,812,821,769,770,3837,3835,1321,227429,851,834,828,223830,1531,198,227010,227443,226760,229759,226766,226765,189,2981,226998,4948,225312,225309,225310) 

            """

    load_d = LoadData('config_template.ini', 'mimic')

    print(load_d.query_db(query_mimic))
    load_d.query_and_save(query_mimic,file_name='tests')



