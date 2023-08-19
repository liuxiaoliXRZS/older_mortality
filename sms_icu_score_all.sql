--------------------------------------------         MIMIC        ----------------------------------------------

----------------------------------------------------------------------------------------------------------------

drop table if exists `db_name.older_study_mimiciii_sms`;
create table `db_name.older_study_mimiciii_sms` as

-- MIMIC-III
with older_study_cohort_mimiciii as (
    select sc.subject_id, sc.hadm_id, sc.icustay_id
    , sc.age, sc.vent, sc.norepinephrine, sc.epinephrine, sc.dopamine, sc.dobutamine
    , icud.intime, icud.outtime, ad.admission_type, sc.death_hosp
    , icud.dod as deathtime
    from `db_name.older_study_mimiciii` sc
    left join `physionet-data.mimiciii_derived.icustay_detail` icud
    on sc.icustay_id = icud.icustay_id
    left join `physionet-data.mimiciii_clinical.admissions` ad
	on icud.subject_id = ad.subject_id
	and icud.hadm_id = ad.hadm_id
)

, surgflag_info as (
  select ie.icustay_id
    , max(case
        when lower(curr_service) like '%surg%' then 1
        when curr_service = 'ORTHO' then 1
    else 0 end) as surgical
  FROM older_study_cohort_mimiciii ie
  left join `physionet-data.mimiciii_clinical.services` se
    on ie.hadm_id = se.hadm_id
    and se.transfertime < DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
  group by ie.icustay_id
)

, sms_icu_0 as (
    select sc.subject_id, sc.hadm_id, sc.icustay_id
    , age, vent, norepinephrine, epinephrine, dopamine, dobutamine
    , death_hosp
    , case when mil.starttime >= sc.intime and round((DATETIME_DIFF(mil.starttime, sc.intime, MINUTE))/1440, 2) < 1 then 1
    when mil.endtime > sc.intime and round((DATETIME_DIFF(mil.endtime, sc.intime, MINUTE))/1440, 2) <= 1 then 1
    when mil.starttime <= sc.intime and round((DATETIME_DIFF(mil.endtime, sc.intime, MINUTE))/1440, 2) >= 1 then 1
    else 0 end as milrinone
    , case when admission_type in ('EMERGENCY', 'URGENT') and sg.surgical = 1 then 1
    else 0 end as acute_surgical_admission
    , case 
    when round((DATETIME_DIFF(deathtime, intime, MINUTE))/1440, 3) > 0 
    and round((DATETIME_DIFF(deathtime, intime, MINUTE))/1440, 3) <= 90
    then 1 else 0 end as death_90day_icu
    , r.rrt, vt.sysbp_min
    , case when cci.malignant_cancer = 1 then 1 else 0 end as malignant_cancer
    from older_study_cohort_mimiciii sc
    left join `physionet-data.mimiciii_derived.rrtfirstday` r
    on sc.icustay_id = r.icustay_id
    left join `physionet-data.mimiciii_derived.vitals_first_day` vt
    on sc.icustay_id = vt.icustay_id
    left join `physionet-data.mimiciii_derived.milrinone_durations` mil
    on sc.icustay_id = mil.icustay_id
    left join surgflag_info sg
    on sc.icustay_id = sg.icustay_id
    left join 
    (
        select hadm_id, max(malignant_cancer) as malignant_cancer
        from `db_name.charlson_mimiciii`
        group by hadm_id
    ) cci
    on sc.hadm_id = cci.hadm_id
)

, sms_icu_1 as (
    select subject_id, hadm_id, icustay_id
    , case when age < 79 then 10
    when age >= 79 then 13 end as age_score
    , case when sysbp_min >= 90 then 0
    when sysbp_min >= 70 then 3
    when sysbp_min >= 50 then 5
    when sysbp_min <= 49 then 6
    else 0 end as sysbp_min_score
    , case when acute_surgical_admission = 1 then 0
    else 3 end as acute_surgical_admission_score
    , case when malignant_cancer = 1 then 7 
    else 0 end as malignant_cancer_score
    , case when (norepinephrine + epinephrine + dopamine + dobutamine + milrinone) >= 1 then 1
    else 0 end as vasopressor_inotropes_score
    , case when vent = 1 then 5 else 0 end as vent_score
    , case when rrt = 1 then 4 else 0 end as rrt_score
    , death_hosp, death_90day_icu
    from sms_icu_0
)

, sms_icu_2 as (
    select subject_id, hadm_id, icustay_id
    , death_hosp, death_90day_icu
    , (age_score + sysbp_min_score + acute_surgical_admission_score + malignant_cancer_score + vasopressor_inotropes_score + vent_score + rrt_score) as sms_score
    from sms_icu_1
)

, sms_icu as (
    select *
    , exp(-3.388 + 0.136*sms_score)/(1+exp(-3.388 + 0.136*sms_score)) as sms_prob
    from sms_icu_2
)

select subject_id, hadm_id, icustay_id, death_hosp, death_90day_icu
, max(sms_score) as sms_score
, max(sms_prob) as sms_prob
from sms_icu
group by subject_id, hadm_id, icustay_id, death_hosp, death_90day_icu
order by icustay_id;


-----------------------------------------------------------------------------------------------
drop table if exists `db_name.older_study_mimiciv_sms`;
create table `db_name.older_study_mimiciv_sms` as

with older_study_cohort_mimiciv as (
    select sc.subject_id, sc.hadm_id, sc.stay_id
    , sc.age, sc.vent, sc.norepinephrine, sc.epinephrine, sc.dopamine, sc.dobutamine
    , sc.admission_type
    , icud.icu_intime as intime, icud.icu_outtime as outtime, sc.death_hosp
    , CAST(pt.dod AS DATETIME) as deathtime
    from `db_name.older_study_mimiciv` sc
    left join `physionet-data.mimic_derived.icustay_detail` icud
    on sc.stay_id = icud.stay_id
    left join `physionet-data.mimic_core.patients` pt
	on sc.subject_id = pt.subject_id
)

, surgflag_info as (
  select ie.stay_id
    , max(case
        when lower(curr_service) like '%surg%' then 1
        when curr_service = 'ORTHO' then 1
    else 0 end) as surgical
  FROM `physionet-data.mimic_icu.icustays` ie
  left join `physionet-data.mimic_hosp.services` se
    on ie.hadm_id = se.hadm_id
    and se.transfertime < DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
  where ie.stay_id in (select stay_id from older_study_cohort_mimiciv)
  group by ie.stay_id
)

, sms_icu_0 as (
    select sc.subject_id, sc.hadm_id, sc.stay_id
    , age, vent, norepinephrine, epinephrine, dopamine, dobutamine
    , death_hosp
    , case 
    when round((DATETIME_DIFF(deathtime, intime, MINUTE))/1440, 3) > 0 
    and round((DATETIME_DIFF(deathtime, intime, MINUTE))/1440, 3) <= 90
    then 1 else 0 end as death_90day_icu
    , case when mil.starttime >= sc.intime and round((DATETIME_DIFF(mil.starttime, sc.intime, MINUTE))/1440, 2) < 1 then 1
    when mil.endtime > sc.intime and round((DATETIME_DIFF(mil.endtime, sc.intime, MINUTE))/1440, 2) <= 1 then 1
    when mil.starttime <= sc.intime and round((DATETIME_DIFF(mil.endtime, sc.intime, MINUTE))/1440, 2) >= 1 then 1
    else 0 end as milrinone
    , case when admission_type in ('EMERGENCY', 'URGENT') and sg.surgical = 1 then 1
    else 0 end as acute_surgical_admission
    , case when sc.stay_id = r.stay_id then 1 else 0 end as rrt
    , vt.sbp_min as sysbp_min
    , case when cci.malignant_cancer = 1 then 1 else 0 end as malignant_cancer

    from older_study_cohort_mimiciv sc
    left join (
        SELECT distinct r.stay_id 
        FROM `physionet-data.mimic_derived.rrt` r
        inner join `physionet-data.mimic_derived.icustay_detail` icud
        on r.stay_id = icud.stay_id
        and r.charttime >= icud.icu_intime and round((DATETIME_DIFF(r.charttime, icud.icu_intime, MINUTE))/1440, 2) < 1
        where dialysis_present = 1
    ) r
    on sc.stay_id = r.stay_id
    left join `physionet-data.mimic_derived.first_day_vitalsign` vt
    on sc.stay_id = vt.stay_id
    left join (select * from `physionet-data.mimic_derived.milrinone` where vaso_rate > 0) mil
    on sc.stay_id = mil.stay_id
    left join surgflag_info sg
    on sc.stay_id = sg.stay_id
    left join 
    (
        select hadm_id, max(malignant_cancer) as malignant_cancer
        from `physionet-data.mimic_derived.charlson`
        group by hadm_id
    ) cci
    on sc.hadm_id = cci.hadm_id
)

, sms_icu_1 as (
    select subject_id, hadm_id, stay_id
    , case when age < 79 then 10
    when age >= 79 then 13 end as age_score
    , case when sysbp_min >= 90 then 0
    when sysbp_min >= 70 then 3
    when sysbp_min >= 50 then 5
    when sysbp_min <= 49 then 6
    else 0 end as sysbp_min_score
    , case when acute_surgical_admission = 1 then 0
    else 3 end as acute_surgical_admission_score
    , case when malignant_cancer = 1 then 7 
    else 0 end as malignant_cancer_score
    , case when (norepinephrine + epinephrine + dopamine + dobutamine + milrinone) >= 1 then 1
    else 0 end as vasopressor_inotropes_score
    , case when vent = 1 then 5 else 0 end as vent_score
    , case when rrt = 1 then 4 else 0 end as rrt_score
    , death_hosp, death_90day_icu
    from sms_icu_0
)

, sms_icu_2 as (
    select subject_id, hadm_id, stay_id
    , death_hosp, death_90day_icu
    , (age_score + sysbp_min_score + acute_surgical_admission_score + malignant_cancer_score + vasopressor_inotropes_score + vent_score + rrt_score) as sms_score
    from sms_icu_1
)

, sms_icu as (
    select *
    , exp(-3.388 + 0.136*sms_score)/(1+exp(-3.388 + 0.136*sms_score)) as sms_prob
    from sms_icu_2
)

select subject_id, hadm_id, stay_id, death_hosp, death_90day_icu
, max(sms_score) as sms_score
, max(sms_prob) as sms_prob
from sms_icu
group by subject_id, hadm_id, stay_id, death_hosp, death_90day_icu
order by stay_id;



drop table if exists `db_name.older_study_mimic_sms`;
create table `db_name.older_study_mimic_sms` as

with cohort_info_0 as (
    select icustay_id as id, subject_id, hadm_id
    , death_hosp, death_90day_icu, sms_score, sms_prob
    from `db_name.older_study_mimiciii_sms`
    union all
    select stay_id as id, subject_id, hadm_id
    , death_hosp, death_90day_icu, sms_score, sms_prob
    from `db_name.older_study_mimiciv_sms`    
)

, cohort_info as (
  select *, ROW_NUMBER() OVER (ORDER BY id) as id_new
  from cohort_info_0
)

select id_new as id, subject_id, hadm_id
, death_hosp, death_90day_icu, sms_score, sms_prob
from cohort_info
order by id_new;

drop table if exists `db_name.older_study_mimiciii_sms`;
drop table if exists `db_name.older_study_mimiciv_sms`;




--------------------------------------------       eICU-CRD       ----------------------------------------------

----------------------------------------------------------------------------------------------------------------

drop table if exists `db_name.older_study_eicu_sms`;
create table `db_name.older_study_eicu_sms` as

with older_study_cohort_eicu as (
    select sc.id, icud.patientunitstayid
    , age, vent, norepinephrine, epinephrine, dopamine, dobutamine
    , admission_type, death_hosp
    from `db_name.older_study_eicu` sc
    left join (
        select patientunitstayid, (id + 50000) as id
        from `db_name.older_study_eicu_initial1` 
    ) icud
    on sc.id = icud.id
)

, sms_icu_0 as (
    select sc.*, sbp.sysbp_min
    , case when sc.patientunitstayid = rrt.patientunitstayid then 1 else 0 end as rrt
    , case when sc.patientunitstayid = oo.patientunitstayid then 1 else 0 end as acute_surgical_admission
    , case when cci.malignant_cancer = 1 then 1 else 0 end as malignant_cancer
    , case when sc.patientunitstayid = mil.patientunitstayid then 1 else 0 end as milrinone
    from older_study_cohort_eicu sc
    left join (
        select patientunitstayid, min(coalesce(ibp_systolic, nibp_systolic)) as sysbp_min
        from `physionet-data.eicu_crd_derived.pivoted_vital`
        where chartoffset >= 0
        and chartoffset < 24*60
        group by patientunitstayid
    ) sbp
    on sc.patientunitstayid = sbp.patientunitstayid
    left join (
        SELECT distinct patientunitstayid 
        FROM `physionet-data.eicu_crd.treatment`
        where treatmentstring in (
            'renal|dialysis|arteriovenous shunt for renal dialysis'
            ,'renal|dialysis|C A V H D'
            ,'renal|dialysis|C V V H'
            ,'renal|dialysis|C V V H D'
            ,'renal|dialysis|hemodialysis'
            ,'renal|dialysis|hemodialysis|emergent'
            ,'renal|dialysis|hemodialysis|for acute renal failure'
            ,'renal|dialysis|hemodialysis|for chronic renal failure'
            ,'renal|dialysis|SLED'
            ,'renal|dialysis|ultrafiltration (fluid removal only)'
            ,'renal|dialysis|ultrafiltration (fluid removal only)|emergent'
            ,'renal|dialysis|ultrafiltration (fluid removal only)|for acute renal failure'
            ,'renal|dialysis|ultrafiltration (fluid removal only)|for chronic renal failure'
        )
        and treatmentoffset > 0
        and treatmentoffset < 24*60        
    ) rrt
    on sc.patientunitstayid = rrt.patientunitstayid
    left join (
        SELECT distinct patientunitstayid 
        FROM `physionet-data.eicu_crd.patient`
        where hospitaladmitsource = 'Operating Room'
    ) oo
    on sc.patientunitstayid = oo.patientunitstayid
    left join (
        SELECT patientunitstayid, max(malignant_cancer) as malignant_cancer
        FROM `db_name.charlson_comorbidity_eicu`
        group by patientunitstayid
    ) cci
    on sc.patientunitstayid = cci.patientunitstayid
    left join (
        select distinct patientunitstayid
        from `physionet-data.eicu_crd.infusiondrug`
        where lower(drugname) like '%milrinone%'
        and drugrate not in (
          'Documentation undone'
          , 'ERROR'
          , 'UD'
          , ''
        )
        and drugrate not like '%OFF%'
        and cast(drugrate as numeric) > 0
        and infusionoffset > 0
        and infusionoffset < 24*60
    ) mil
    on sc.patientunitstayid = mil.patientunitstayid
)

, sms_icu_1 as (
    select id, patientunitstayid
    , case when age < 79 then 10
    when age >= 79 then 13 end as age_score
    , case when sysbp_min >= 90 then 0
    when sysbp_min >= 70 then 3
    when sysbp_min >= 50 then 5
    when sysbp_min <= 49 then 6
    else 0 end as sysbp_min_score
    , case when acute_surgical_admission = 1 then 0
    else 3 end as acute_surgical_admission_score
    , case when malignant_cancer = 1 then 7 
    else 0 end as malignant_cancer_score
    , case when (norepinephrine + epinephrine + dopamine + dobutamine + milrinone) >= 1 then 1
    else 0 end as vasopressor_inotropes_score
    , case when vent = 1 then 5 else 0 end as vent_score
    , case when rrt = 1 then 4 else 0 end as rrt_score
    , death_hosp
    from sms_icu_0
)

, sms_icu_2 as (
    select id, patientunitstayid, death_hosp
    , (age_score + sysbp_min_score + acute_surgical_admission_score + malignant_cancer_score + vasopressor_inotropes_score + vent_score + rrt_score) as sms_score
    from sms_icu_1
)

, sms_icu as (
    select *
    , exp(-3.388 + 0.136*sms_score)/(1+exp(-3.388 + 0.136*sms_score)) as sms_prob
    from sms_icu_2
)

select id, patientunitstayid, death_hosp
, max(sms_score) as sms_score
, max(sms_prob) as sms_prob
from sms_icu
group by id, patientunitstayid, death_hosp
order by id;



--------------------------------------------       AmsterdamUMC       ----------------------------------------------

--------------------------------------------------------------------------------------------------------------------
drop table if exists `db_name.older_study_ams_sms`;
create table `db_name.older_study_ams_sms` as

with older_study_cohort_ams as (
    select sc.id, sc.admissionid
    , sc.agegroup as age, vent, norepinephrine, epinephrine, dopamine, dobutamine
    , death_hosp, adm.admittedat
    , case 
    when adm.urgency = true then 'unplanned'
    when adm.urgency = False then 'planned'
    else null end as admission_type
    from `db_name.older_study_ams` sc
    left join `physionet-data.amsterdamdb.admissions` adm
    on sc.admissionid = adm.admissionid
)

, surgflag_info_0 as (
        select admissionid
        , CASE
            WHEN itemid IN (
                --SURGICAL
                13116, --D_Thoraxchirurgie_CABG en Klepchirurgie
                16671, --DMC_Thoraxchirurgie_CABG en Klepchirurgie
                13117, --D_Thoraxchirurgie_Cardio anders
                16672, --DMC_Thoraxchirurgie_Cardio anders
                13118, --D_Thoraxchirurgie_Aorta chirurgie
                16670, --DMC_Thoraxchirurgie_Aorta chirurgie
                13119, --D_Thoraxchirurgie_Pulmonale chirurgie
                16673, --DMC_Thoraxchirurgie_Pulmonale chirurgie

                --Not surgical: 13141, --D_Algemene chirurgie_Algemeen   
                --Not surgical: 16642, --DMC_Algemene chirurgie_Algemeen
                13121, --D_Algemene chirurgie_Buikchirurgie
                16643, --DMC_Algemene chirurgie_Buikchirurgie
                13123, --D_Algemene chirurgie_Endocrinologische chirurgie
                16644, --DMC_Algemene chirurgie_Endocrinologische chirurgie
                13145, --D_Algemene chirurgie_KNO/Overige
                16645, --DMC_Algemene chirurgie_KNO/Overige
                13125, --D_Algemene chirurgie_Orthopedische chirurgie
                16646, --DMC_Algemene chirurgie_Orthopedische chirurgie
                13122, --D_Algemene chirurgie_Transplantatie chirurgie
                16647, --DMC_Algemene chirurgie_Transplantatie chirurgie
                13124, --D_Algemene chirurgie_Trauma
                16648, --DMC_Algemene chirurgie_Trauma
                13126, --D_Algemene chirurgie_Urogenitaal
                16649, --DMC_Algemene chirurgie_Urogenitaal
                13120, --D_Algemene chirurgie_Vaatchirurgie
                16650, --DMC_Algemene chirurgie_Vaatchirurgie

                13128, --D_Neurochirurgie _Vasculair chirurgisch
                16661, --DMC_Neurochirurgie _Vasculair chirurgisch
                13129, --D_Neurochirurgie _Tumor chirurgie
                16660, --DMC_Neurochirurgie _Tumor chirurgie
                13130, --D_Neurochirurgie_Overige
                16662, --DMC_Neurochirurgie_Overige

                18596, --Apache II Operatief  Gastr-intenstinaal
                18597, --Apache II Operatief Cardiovasculair
                18598, --Apache II Operatief Hematologisch
                18599, --Apache II Operatief Metabolisme
                18600, --Apache II Operatief Neurologisch
                18601, --Apache II Operatief Renaal
                18602, --Apache II Operatief Respiratoir

                17008, --APACHEIV Post-operative cardiovascular
                17009, --APACHEIV Post-operative gastro-intestinal
                17010, --APACHEIV Post-operative genitourinary
                17011, --APACHEIV Post-operative hematology
                17012, --APACHEIV Post-operative metabolic
                17013, --APACHEIV Post-operative musculoskeletal /skin
                17014, --APACHEIV Post-operative neurologic
                17015, --APACHEIV Post-operative respiratory
                17016, --APACHEIV Post-operative transplant
                17017 --APACHEIV Post-operative trauma

            ) THEN 1
            WHEN itemid = 18669 AND valueid BETWEEN 1 AND 26 THEN 1 --NICE APACHEII diagnosen
            WHEN itemid = 18671 AND valueid BETWEEN 222 AND 452 THEN 1 --NICE APACHEIV diagnosen
            ELSE 0
        END AS surgical
    FROM `physionet-data.amsterdamdb.listitems`
    WHERE itemid IN (
        -- Diagnosis - LEVEL 2
        --SURGICAL
        13116, --D_Thoraxchirurgie_CABG en Klepchirurgie
        16671, --DMC_Thoraxchirurgie_CABG en Klepchirurgie
        13117, --D_Thoraxchirurgie_Cardio anders
        16672, --DMC_Thoraxchirurgie_Cardio anders
        13118, --D_Thoraxchirurgie_Aorta chirurgie
        16670, --DMC_Thoraxchirurgie_Aorta chirurgie
        13119, --D_Thoraxchirurgie_Pulmonale chirurgie
        16673, --DMC_Thoraxchirurgie_Pulmonale chirurgie
        
        13141, --D_Algemene chirurgie_Algemeen   
        16642, --DMC_Algemene chirurgie_Algemeen
        13121, --D_Algemene chirurgie_Buikchirurgie
        16643, --DMC_Algemene chirurgie_Buikchirurgie
        13123, --D_Algemene chirurgie_Endocrinologische chirurgie
        16644, --DMC_Algemene chirurgie_Endocrinologische chirurgie
        13145, --D_Algemene chirurgie_KNO/Overige
        16645, --DMC_Algemene chirurgie_KNO/Overige
        13125, --D_Algemene chirurgie_Orthopedische chirurgie
        16646, --DMC_Algemene chirurgie_Orthopedische chirurgie
        13122, --D_Algemene chirurgie_Transplantatie chirurgie
        16647, --DMC_Algemene chirurgie_Transplantatie chirurgie
        13124, --D_Algemene chirurgie_Trauma
        16648, --DMC_Algemene chirurgie_Trauma
        13126, --D_Algemene chirurgie_Urogenitaal
        16649, --DMC_Algemene chirurgie_Urogenitaal
        13120, --D_Algemene chirurgie_Vaatchirurgie
        16650, --DMC_Algemene chirurgie_Vaatchirurgie

        13128, --D_Neurochirurgie _Vasculair chirurgisch
        16661, --DMC_Neurochirurgie _Vasculair chirurgisch
        13129, --D_Neurochirurgie _Tumor chirurgie
        16660, --DMC_Neurochirurgie _Tumor chirurgie
        13130, --D_Neurochirurgie_Overige
        16662, --DMC_Neurochirurgie_Overige
        
        18596, --Apache II Operatief  Gastr-intenstinaal
        18597, --Apache II Operatief Cardiovasculair
        18598, --Apache II Operatief Hematologisch
        18599, --Apache II Operatief Metabolisme
        18600, --Apache II Operatief Neurologisch
        18601, --Apache II Operatief Renaal
        18602, --Apache II Operatief Respiratoir
        
        17008, --APACHEIV Post-operative cardiovascular
        17009, --APACHEIV Post-operative gastro-intestinal
        17010, --APACHEIV Post-operative genitourinary
        17011, --APACHEIV Post-operative hematology
        17012, --APACHEIV Post-operative metabolic
        17013, --APACHEIV Post-operative musculoskeletal /skin
        17014, --APACHEIV Post-operative neurologic
        17015, --APACHEIV Post-operative respiratory
        17016, --APACHEIV Post-operative transplant
        17017, --APACHEIV Post-operative trauma
        
        --NICE: surgical/medical combined in same parameter
        18669, --NICE APACHEII diagnosen
        18671 --NICE APACHEIV diagnosen
    )
  and admissionid in (select admissionid from older_study_cohort_ams)
)

, surgflag_info as (
    select sc.admissionid, max(case when surgical = 1 then 1 else 0 end) as surgical
    from older_study_cohort_ams sc 
    left join surgflag_info_0 si 
    on sc.admissionid = si.admissionid
    group by sc.admissionid
)

, dialysis_info_0 as (
  SELECT admissionid, start as charttime
  FROM `physionet-data.amsterdamdb.processitems`
  where item like '%CVVH%'
  union all
  SELECT admissionid, measuredat as charttime
  FROM `physionet-data.amsterdamdb.listitems` 
  where item like '%CVVH%'
  union all
  SELECT admissionid, measuredat as charttime
  FROM `physionet-data.amsterdamdb.numericitems` 
  where item like '%CVVH%'
  or item like '%HHD%'
  or item like '%Hemodialyse%'
  or item like '%MFT_UF Totaal (ingesteld)%'
  or item like '%MFT_Ultrafiltratie (ingesteld)%'
)

, dialysis_info as (
      select distinct di.admissionid
      from dialysis_info_0 di
      inner join older_study_cohort_ams sc
      on di.admissionid = sc.admissionid
      and di.charttime >= sc.admittedat
      and (di.charttime - sc.admittedat) < 1000*60*60*24
)

, digoxin_info_0 as (
    select dg.admissionid
    , case when (dg.start - ad.admittedat)/(1000*60*60) >= 0 and (dg.start - ad.admittedat)/(1000*60*60) < 24 then 1
    when (dg.stop - ad.admittedat)/(1000*60*60) > 0 and (dg.stop - ad.admittedat)/(1000*60*60) < 24 then 1
    when (dg.start - ad.admittedat)/(1000*60*60) <= 0 and (dg.stop - ad.admittedat)/(1000*60*60) >= 24 then 1
    else 0 end as digoxin
    from `physionet-data.amsterdamdb.drugitems` dg
    inner JOIN older_study_cohort_ams ad
    ON dg.admissionid = ad.admissionid
    WHERE ordercategoryid = 65 -- continuous i.v. perfusor
    AND itemid IN (
        7173, 9087
    )
    AND rate > 0 and dose > 0
)

, digoxin_info as (
    select distinct admissionid
    from digoxin_info_0
    where digoxin = 1
)

, sbp_info as (
	SELECT vt.admissionid
	, min(vt.value) as sysbp_min
	from `physionet-data.amsterdamdb.numericitems` vt 
	inner join older_study_cohort_ams sc
	on sc.admissionid = vt.admissionid
	and vt.measuredat >= sc.admittedat 
	and (vt.measuredat - sc.admittedat) < 1000*60*60*24 --measurements within 24 hours
	where vt.itemid in (
        6641, 6678, 8841
    )
    and vt.value > 0 and vt.value < 400
    group by vt.admissionid
)

, sms_icu_0 as (
    select sc.id, sc.admissionid
    , age, vent, norepinephrine, epinephrine, dopamine, dobutamine
    , case when sc.admissionid = dig.admissionid then 1 else 0 end as digoxin
    , death_hosp
    , case 
    when admission_type = 'unplanned' and si.surgical = 1 then 1 
    else 0 end as acute_surgical_admission
    , case when sc.admissionid = di.admissionid then 1
    else 0 end as rrt
    , sb.sysbp_min
    from older_study_cohort_ams sc
    left join surgflag_info si
    on sc.admissionid = si.admissionid
    left join dialysis_info di
    on sc.admissionid = di.admissionid
    left join sbp_info sb
    on sc.admissionid = sb.admissionid
    left join digoxin_info dig
    on sc.admissionid = dig.admissionid
)

, sms_icu_1 as (
    select id, admissionid
    , case when age = '70-79' then 10
    when age = '80+' then 13 
    else 0 end as age_score
    , case when sysbp_min >= 90 then 0
    when sysbp_min >= 70 then 3
    when sysbp_min >= 50 then 5
    when sysbp_min <= 49 then 6
    else 0 end as sysbp_min_score
    , case when acute_surgical_admission = 1 then 0
    else 3 end as acute_surgical_admission_score
    , 0 as malignant_cancer_score
    , case when (norepinephrine + epinephrine + dopamine + dobutamine + digoxin) >= 1 then 1
    else 0 end as vasopressor_inotropes_score
    , case when vent = 1 then 5 else 0 end as vent_score
    , case when rrt = 1 then 4 else 0 end as rrt_score
    , death_hosp
    from sms_icu_0
)

, sms_icu_2 as (
    select id, admissionid, death_hosp
    , (age_score + sysbp_min_score + acute_surgical_admission_score + malignant_cancer_score + vasopressor_inotropes_score + vent_score + rrt_score) as sms_score
    from sms_icu_1
)

, sms_icu as (
    select *
    , exp(-3.388 + 0.136*sms_score)/(1+exp(-3.388 + 0.136*sms_score)) as sms_prob
    from sms_icu_2
)

select id, admissionid, death_hosp
, max(sms_score) as sms_score
, max(sms_prob) as sms_prob
from sms_icu
group by id, admissionid, death_hosp
order by id;