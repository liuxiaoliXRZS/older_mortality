-- This code is used to prepare the older study data based on eICU-CRD dataset
-- By Xiaoli Liu
-- 2021.10.05
-- 2022.12.12 change (add readmission info)

-- criteria
-- age > 65; 
-- ethnicity: black, asian, hispanic, white
--     no ethnicity will be excluded & native american
-- first icu admission
-- los icu day >= 1d

drop table if exists `db_name.older_study_eicu_initial`;
create table `db_name.older_study_eicu_initial` as

-- first_icu admission - cohort
-- consider part of patients' very short duration in the unitvisitnumber = 1 should be re-correct to the second time as the first ICU stay
with first_icu_info as (
  select distinct uniquepid, patienthealthsystemstayid, patientunitstayid--, flag
  from `physionet-data.eicu_crd.patient` 
  where unitvisitnumber = 1
  and round(unitdischargeoffset/60, 2) >= 2
  union all
  select distinct uniquepid, patienthealthsystemstayid, patientunitstayid--, flag
  from `physionet-data.eicu_crd.patient` 
  where unitvisitnumber = 2
  and patienthealthsystemstayid in (
    select distinct patienthealthsystemstayid
    from `physionet-data.eicu_crd.patient` 
    where round(unitdischargeoffset/60, 2) < 2
    and unitvisitnumber = 1
  )
  and round(unitdischargeoffset/60, 2) >= 2  
)

-- drop children and higher elderly patients
, older_study_cohort_eicu_0 as (
  SELECT icud.patientunitstayid, icud.uniquepid, icud.patienthealthsystemstayid
  , case
  when unittype = 'CSICU' then 'CSRU'
  when unittype = 'SICU' then 'SICU' 
  when unittype = 'Cardiac ICU' then 'CCU'
  when unittype = 'CCU-CTICU' then 'CSRU'
  when unittype = 'CTICU' then 'CSRU'
  when unittype = 'MICU' then 'MICU'
  when unittype = 'Med-Surg ICU' then 'Med-Surg_ICU'
  when unittype = 'Neuro ICU' then 'NICU'
  when unittype is null then 'Other/Unknown'
  else 'Other/Unknown' end as first_careunit
  , case 
  when gender = 0 then 'F' -- female : 0
  when gender = 1 then 'M'
  else null end as gender
  , cast(case when age = '> 89' then '91.4' else age end as numeric) as age
  , hosp_mort as death_hosp
  , case 
  when ethnicity = 'African American' then 'black'
  when ethnicity = 'Asian' then 'asian'
  when ethnicity = 'Caucasian' then 'white'
  when ethnicity = 'Hispanic' then 'hispanic'
  else 'other' end as ethnicity
  , case
  when pt.hospitaladmitsource = 'Acute Care/Floor' then 'EMERGENCY'
  when pt.hospitaladmitsource = 'Chest Pain Center' then 'EMERGENCY'
  when pt.hospitaladmitsource = 'Direct Admit' then 'EMERGENCY'
  when pt.hospitaladmitsource = 'Emergency Department' then 'EMERGENCY'
  when pt.hospitaladmitsource = 'Floor' then 'Intermediate care unit'
  when pt.hospitaladmitsource = 'ICU' then 'URGENT'
  when pt.hospitaladmitsource = 'ICU to SDU' then 'Intermediate care unit'
  when pt.hospitaladmitsource = 'Observation' then 'Intermediate care unit'
  when pt.hospitaladmitsource = 'Operating Room' then 'URGENT'
  when pt.hospitaladmitsource = 'Other' then 'Other/Unknown'
  when pt.hospitaladmitsource = 'Other Hospital' then 'Intermediate care unit'
  when pt.hospitaladmitsource = 'Other ICU' then 'URGENT'
  when pt.hospitaladmitsource = 'PACU' then 'URGENT'
  when pt.hospitaladmitsource = 'Recovery Room' then 'Intermediate care unit'
  when pt.hospitaladmitsource = 'Step-Down Unit (SDU)' then 'Intermediate care unit'
  when pt.hospitaladmitsource is null then 'Other/Unknown'
  else 'Other/Unknown' end as admission_type
  , round(admissionheight,1) as height
  , case when admissionweight > 20 then round(admissionweight,1) 
  else null end as weight
  , case when admissionweight + admissionheight is not null and admissionheight > 0 and admissionweight > 20
    then round(10000*(admissionweight/admissionheight/admissionheight),1) else null end as bmi
  , icu_los_hours
  , hospitaladmitoffset, hospitaldischargeoffset 
  , unitadmitoffset, unitdischargeoffset
  , round((hospitaldischargeoffset - hospitaladmitoffset)/(24*60),2) as los_hospital_day
  , round(unitdischargeoffset/(24*60),2) as los_icu_day
  , case when hosp_mort = 0 then null else round(unitdischargeoffset/(24*60.0),2) end as deathtime_icu_day
  , case when hosp_mort = 0 then null else CAST(ceil(unitdischargeoffset/60) AS INT64) end as deathtime_icu_hour
  , round(- hospitaladmitoffset/(24*60),3) as los_icu_admit_day
  , hp.hospitalid, hp.hospitaldischargeyear, hp.numbedscategory
  , case when hp.teachingstatus is true then 1 else 0 end as teachingstatus, hp.region 
  FROM `physionet-data.eicu_crd_derived.icustay_detail` icud
  left join (select patientunitstayid, hospitaladmitsource from `physionet-data.eicu_crd.patient`) pt 
  on icud.patientunitstayid = pt.patientunitstayid
  left join (
      SELECT p.patientunitstayid, p.hospitalid, p.hospitaldischargeyear, numbedscategory, teachingstatus, region 
      FROM `physionet-data.eicu_crd.patient` p 
      left join `physionet-data.eicu_crd.hospital` h 
      on p.hospitalid = h.hospitalid
    ) hp 
  on icud.patientunitstayid = hp.patientunitstayid  
  where icud.patientunitstayid in (select patientunitstayid from first_icu_info) -- first icu amdission
  and age not like ''
  and gender is not null
)

, older_study_cohort_eicu as (
    select patientunitstayid, uniquepid, patienthealthsystemstayid
    , first_careunit, gender, age, death_hosp, ethnicity
    , admission_type
    , case when height > 120 and height < 230 then height else null end as height
    , case when weight > 20 and weight < 400 then weight else null end as weight
    , case when height > 120 and height < 230 and weight > 20 and weight < 400 then bmi else null end as bmi
    , icu_los_hours, hospitaladmitoffset
    , hospitaldischargeoffset, unitadmitoffset, unitdischargeoffset, los_hospital_day
    , los_icu_day, deathtime_icu_day, deathtime_icu_hour
    , case when los_icu_admit_day < 0 then 0 else los_icu_admit_day end as los_icu_admit_day
    , hospitalid, hospitaldischargeyear, numbedscategory, teachingstatus, region
    from older_study_cohort_eicu_0
    where age >= 65
    and los_icu_day >= 1
    and los_hospital_day >= 1
)

select *
from older_study_cohort_eicu;


drop table if exists `db_name.older_study_eicu_initial1`;
create table `db_name.older_study_eicu_initial1` as

with older_study_cohort_eicu as (
    select *
    from `db_name.older_study_eicu_initial`
)

-- 1. sofa score
--  GCS, MAP, FiO2, Ventilation status (sourced FROM `physionet-data.mimic_icu.chartevents`)
--  Creatinine, Bilirubin, FiO2, PaO2, Platelets (sourced FROM `physionet-data.mimic_icu.labevents`)
--  Dopamine, Dobutamine, Epinephrine, Norepinephrine (sourced FROM `physionet-data.mimic_icu.inputevents_mv` and INPUTEVENTS_CV)
--  Urine output (sourced from OUTPUTEVENTS)
-- 2. SAPS II:
--  Age, GCS
--  VITALS: Heart rate, systolic blood pressure, temperature
--  FLAGS: ventilation/cpap
--  IO: urine output
--  LABS: PaO2/FiO2 ratio, blood urea nitrogen, WBC, potassium, sodium, HCO3
-- 3. OASIS
--  Heart rate, GCS, MAP, Temperature, Respiratory rate, Ventilation status (sourced FROM `physionet-data.mimic_icu.chartevents`)
--  Urine output (sourced from OUTPUTEVENTS)
--  Elective surgery (sourced FROM `physionet-data.mimic_core.admissions` and SERVICES)
--  Pre-ICU in-hospital length of stay (sourced FROM `physionet-data.mimic_core.admissions` and ICUSTAYS)
--  Age (sourced FROM `physionet-data.mimic_core.patients`)
, surgflag_info as (
    SELECT patientunitstayid
    , case when electivesurgery = 1 then 1 else 0 end as surgical 
    FROM `physionet-data.eicu_crd.apachepredvar`
    where patientunitstayid in (select patientunitstayid from older_study_cohort_eicu)
)

-- first day ventilation
, vent_info AS (
    SELECT ie.patientunitstayid
    , MAX(
        CASE WHEN v.patientunitstayid IS NOT NULL THEN 1 ELSE 0 END
    ) AS vent
    FROM older_study_cohort_eicu ie
    LEFT JOIN `db_name.pivoted_vent_eicu` v
        ON ie.patientunitstayid = v.patientunitstayid
        AND (
           (v.starttime >= 0 and v.starttime < 24*60)
        OR (v.endtime > 0 and v.endtime < 24*60)
        OR (v.starttime < 0 AND v.endtime >= 24*60)
        )
    GROUP BY ie.patientunitstayid
)

, pafi_0 as (
  select patientunitstayid, chartoffset
  , round(pao2/fio2,0) as pao2fio2ratio
  from `physionet-data.eicu_crd_derived.pivoted_bg`
  where fio2 is not null
  and pao2 is not null
  and patientunitstayid in (select patientunitstayid from older_study_cohort_eicu)
)

, pafi_1 as (
  -- join blood gas to ventilation durations to determine if patient was vent
  select pf.patientunitstayid
  , pf.chartoffset
  -- because pafi has an interaction between vent/PaO2:FiO2, we need two columns for the score
  -- it can happen that the lowest unventilated PaO2/FiO2 is 68, but the lowest ventilated PaO2/FiO2 is 120
  -- in this case, the SOFA score is 3, *not* 4.
  , case when pv.patientunitstayid is null then pao2fio2ratio else null end pao2fio2ratio_novent
  , case when pv.patientunitstayid is not null then pao2fio2ratio else null end pao2fio2ratio_vent
  from pafi_0 pf
  left join `db_name.pivoted_vent_eicu` pv
    on pf.patientunitstayid = pv.patientunitstayid
    and pf.chartoffset >= pv.starttime
    and pf.chartoffset <= pv.endtime
)

, pafi as (
    select pf.patientunitstayid, min(pao2fio2ratio_novent) as pao2fio2ratio_novent
    , min(pao2fio2ratio_vent) as pao2fio2ratio_vent
    from pafi_1 pf
    inner join older_study_cohort_eicu sc 
    on pf.patientunitstayid = sc.patientunitstayid
    and pf.chartoffset <= 24*60
    and pf.chartoffset >= -6*60
    group by pf.patientunitstayid    
)

, lab_info_extra_0 as (
    select sc.patientunitstayid
    , max(case when fio2 < 0.21 then 0.21 else fio2 end) as fio2_max
    , min(pao2) as pao2_min
    , max(paco2) as paco2_max
    , min(baseexcess) as baseexcess_min
    from older_study_cohort_eicu sc 
    left join `physionet-data.eicu_crd_derived.pivoted_bg` p 
    on sc.patientunitstayid = p.patientunitstayid
    and p.chartoffset <= 24*60
    and p.chartoffset >= -6*60
    group by sc.patientunitstayid
)

, lab_info_extra_1 as (
	select sc.patientunitstayid
	, max(alt) as alt_max
    , max(ast) as ast_max
    , max(alp) as alp_max
	from older_study_cohort_eicu sc
    left join `physionet-data.eicu_crd_derived.pivoted_lab` lab 
	on lab.patientunitstayid = sc.patientunitstayid
	and lab.chartoffset >= -60*6
	and lab.chartoffset <= 24*60 
    group by sc.patientunitstayid
)

, lab_info_extra_20 as (
  select
      patientunitstayid
    , labname
    , labresultoffset
    , labresultrevisedoffset
  from `physionet-data.eicu_crd.lab`
  where labname in
  ('troponin - T'
    , 'magnesium'
    , 'BNP' 
    , 'fibrinogen'
    , '-lymphs'
    , 'chloride'
    , 'PT'
    , '-polys'   -- , neutrophils
  )
  and patientunitstayid in (select patientunitstayid from older_study_cohort_eicu)
  group by patientunitstayid, labname, labresultoffset, labresultrevisedoffset
  having count(distinct labresult)<=1
)

, lab_info_extra_21 as (
  select
      lab.patientunitstayid
    , lab.labname
    , lab.labresultoffset
    , lab.labresultrevisedoffset
    , lab.labresult
    , ROW_NUMBER() OVER
        (
          PARTITION BY lab.patientunitstayid, lab.labname, lab.labresultoffset
          ORDER BY lab.labresultrevisedoffset DESC
        ) as rn
  from `physionet-data.eicu_crd.lab` lab
  inner join lab_info_extra_20 vw0
    ON  lab.patientunitstayid = vw0.patientunitstayid
    AND lab.labname = vw0.labname
    AND lab.labresultoffset = vw0.labresultoffset
    AND lab.labresultrevisedoffset = vw0.labresultrevisedoffset
  -- only valid lab values
  WHERE
    (lab.labname = 'troponin - T' and lab.labresult > 0 and lab.labresult < 10)
    OR (lab.labname = 'magnesium' and lab.labresult > 0 and lab.labresult <= 243)
    OR (lab.labname = 'BNP' and lab.labresult > 0 and lab.labresult < 50000)
    OR (lab.labname = 'fibrinogen' and lab.labresult > 0 and lab.labresult < 5000)
    OR (lab.labname = '-lymphs' and lab.labresult > 0 and lab.labresult < 100)
    OR (lab.labname = 'chloride' and lab.labresult > 0 and lab.labresult < 1000)
    OR (lab.labname = 'PT' and lab.labresult > 0 and lab.labresult < 150)
    OR (lab.labname = '-polys' and lab.labresult > 0 and lab.labresult < 100)
)

, lab_info_extra_22 as (
	select lab.patientunitstayid 
	, lab.labname
    , lab.labresultoffset
    , lab.labresultrevisedoffset
    , lab.labresult
	from lab_info_extra_21 lab 
	inner join older_study_cohort_eicu sc 
	on lab.patientunitstayid = sc.patientunitstayid
	and lab.labresultoffset >= -6*60
	and lab.labresultoffset <= 24*60
	where rn = 1	
)

, lab_info_extra_2 as (
	select patientunitstayid	  
	  , MAX(case when labname = 'troponin - T' then labresult else null end) as troponin_max
	  , MAX(case when labname = 'magnesium' then labresult else null end) as magnesium_max
	  , MAX(case when labname = 'BNP' then labresult else null end) as bnp_max
	  , MIN(case when labname = 'fibrinogen' then labresult else null end) as fibrinogen_min
	  , MAX(case when labname = '-lymphs' then labresult else null end) as lymphocytes_max        
	  , MIN(case when labname = '-lymphs' then labresult else null end) as lymphocytes_min
	  , MIN(case when labname = '-polys' then labresult else null end) as neutrophils_min
	from lab_info_extra_22
	group by patientunitstayid
)

, lab_info as (
    select sc.patientunitstayid
    , creatinine_max, bilirubin_max, platelet_min
    , bun_max, wbc_max, potassium_max, sodium_max
    , bicarbonate_min, bicarbonate_max
    , pao2fio2ratio_novent, pao2fio2ratio_vent
    , albumin_min, hematocrit_max
    , 100*fio2_max as fio2_max, pao2_min, paco2_max, baseexcess_min
    , aniongap_max, chloride_min,lactate_max
    , pt_max, ptt_max, inr_min, glucose_max
    , alt_max, ast_max, alp_max
    , troponin_max, magnesium_max, bnp_max
	, fibrinogen_min, lymphocytes_max, lymphocytes_min, neutrophils_min

    from older_study_cohort_eicu sc 
    left join `physionet-data.eicu_crd_derived.labsfirstday` lab
    on sc.patientunitstayid = lab.patientunitstayid
    left join pafi pf 
    on sc.patientunitstayid = pf.patientunitstayid
    left join lab_info_extra_0 l0 
    on sc.patientunitstayid = l0.patientunitstayid
    left join lab_info_extra_1 l1 
    on sc.patientunitstayid = l1.patientunitstayid
    left join lab_info_extra_2 l2 
    on sc.patientunitstayid = l2.patientunitstayid     
)


, vital_gcs_uo_info as (
    select sc.patientunitstayid, gcs.gcs_min
    , coalesce(cast(gcs.gcsmotor as numeric),6) as gcsmotor
    , coalesce(cast(gcs.gcsverbal as numeric),5) as gcsverbal
    , coalesce(cast(gcs.gcseyes as numeric),4) as gcseyes
    , round(vital.heart_rate_mean, 1) as heart_rate_mean
    , round(bp.mbp_mean, 1) as mbp_mean
    , round(bp.sbp_mean, 1) as sbp_mean
    , round(vital.resp_rate_mean, 1) as resp_rate_mean
    , round(vital.temperature_mean, 1) as temperature_mean
    , uo.urineoutput, vital.spo2_min
    from older_study_cohort_eicu sc
    left join (
        select patientunitstayid
        , min(gcs) as gcs_min, min(gcs_eyes) as gcseyes, min(gcs_motor) as gcsmotor
        , min(gcs_verbal) as gcsverbal 
        from `physionet-data.eicu_crd_derived.pivoted_score` 
        where chartoffset >= -6*60
        and chartoffset <= 24*60
        and gcs is not null
        group by patientunitstayid
        ) gcs
    on sc.patientunitstayid = gcs.patientunitstayid
    left join (
        select patientunitstayid
        , avg(sbp) as sbp_mean
        , avg(map) as mbp_mean
        from `db_name.pivoted_blood_pressure_eicu`
        where observationoffset >= 0
        and observationoffset <= 24*60
        group by patientunitstayid
    ) bp
    on sc.patientunitstayid = bp.patientunitstayid
    left join (
        select patientunitstayid
        , avg(heartrate) as heart_rate_mean
        , avg(respiratoryrate) as resp_rate_mean
        , avg(temperature) as temperature_mean
        , min(spo2) as spo2_min 
        from `physionet-data.eicu_crd_derived.pivoted_vital`
        where chartoffset >= 0
        and chartoffset <= 24*60
        group by patientunitstayid
    ) vital
    on sc.patientunitstayid = vital.patientunitstayid
    left join (
        select patientunitstayid, sum(urineoutput) as urineoutput
        from `physionet-data.eicu_crd_derived.pivoted_uo`
        where chartoffset >= 0
        and chartoffset <= 24*60
        group by patientunitstayid
    ) uo
    on sc.patientunitstayid = uo.patientunitstayid    
)

, vaso_mv_info AS (
    SELECT ie.patientunitstayid, rate_norepinephrine
    , rate_epinephrine, rate_dopamine, rate_dobutamine
    from older_study_cohort_eicu ie
    LEFT JOIN (
        select patientunitstayid, max(rate_norepinephrine) as rate_norepinephrine
        from `db_name.norepinephrine_info_eicu`
        where infusionoffset >= 0
        and infusionoffset <= 24*60
        group by patientunitstayid
    ) nop
    on ie.patientunitstayid = nop.patientunitstayid
    LEFT JOIN (
        select patientunitstayid, max(rate_epinephrine) as rate_epinephrine
        from `db_name.epinephrine_info_eicu`
        where infusionoffset >= 0
        and infusionoffset <= 24*60
        group by patientunitstayid
    ) ep
    on ie.patientunitstayid = ep.patientunitstayid
    LEFT JOIN (
        select patientunitstayid, max(rate_dopamine) as rate_dopamine
        from `db_name.dopamine_info_eicu`
        where infusionoffset >= 0
        and infusionoffset <= 24*60
        group by patientunitstayid
    ) dop
    on ie.patientunitstayid = dop.patientunitstayid
    LEFT JOIN (
        select patientunitstayid, max(rate_dobutamine) as rate_dobutamine
        from `db_name.dobutamine_info_eicu`
        where infusionoffset >= 0
        and infusionoffset <= 24*60
        group by patientunitstayid
    ) dob
    on ie.patientunitstayid = dob.patientunitstayid
)

-- 7. activity
, activity_info as (
    select patientunitstayid
    , max(case when cplitemvalue in ('Ambulate', 'Ambulate with assistance', 
        'Assistive Device', 'Bathroom privileges') then 1 else 0 end) as stand
    , max(case when cplitemvalue in ('Chair') then 1 else 0 end) as sit 
    , max(case when cplitemvalue in ('Bedrest', 'Do not elevate HOB', 
        'HOB 30 degrees', 'Turn/ROM') then 1 else 0 end) as bed
    , max(case when cplitemvalue is not null then 1 else 0 end) as activity_eva_flag
    from `physionet-data.eicu_crd.careplangeneral` 
    where patientunitstayid in (select patientunitstayid from older_study_cohort_eicu)
    and cplgroup = 'Activity'
    and cplitemoffset <= 24*60
    and cplitemoffset >= -24*60
   group by patientunitstayid
)

-- 8. score
, score_info as (
    select sc.patientunitstayid, aiv.apache_iv, aiv.predictediculos_iv, aiv.predictedhospitallos_iv
    , case when aiv.apsiii is null then 0 else aiv.apsiii end as apsiii, aiv.apache_iv_prob
    , aiva.apache_iva, aiva.apache_iva_prob, aiva.predictediculos_iva, aiva.predictedhospitallos_iva
    , oa.oasis, sp.sapsii as saps, sf.sofa
    , cci.charlson_comorbidity_indexs as cci_score
    from older_study_cohort_eicu sc
    left join (
        SELECT patientunitstayid, apachescore as apache_iv
        , predictediculos as predictediculos_iv
        , predictedhospitallos as predictedhospitallos_iv
        , acutephysiologyscore as apsiii
        , predictedhospitalmortality as apache_iv_prob
        FROM `physionet-data.eicu_crd.apachepatientresult`
        where apacheversion = 'IV'
    ) aiv
    on sc.patientunitstayid = aiv.patientunitstayid
    left join (
        SELECT patientunitstayid, apachescore as apache_iva
        , predictediculos as predictediculos_iva
        , predictedhospitallos as predictedhospitallos_iva
        , predictedhospitalmortality as apache_iva_prob        
        FROM `physionet-data.eicu_crd.apachepatientresult`
        where apacheversion = 'IVa'
    ) aiva    
    on sc.patientunitstayid = aiva.patientunitstayid
    left join `db_name.oasis_firstday_eicu` oa 
    on sc.patientunitstayid = oa.patientunitstayid
    left join `db_name.sapsii_firstday_eicu` sp 
    on sc.patientunitstayid = sp.patientunitstayid
    left join (
        SELECT patientunitstayid, SOFA_24hours as sofa
        FROM `db_name.pivoted_sofa_eicu`
        where hr = 23
    ) sf 
    on sc.patientunitstayid = sf.patientunitstayid
    left join (
        select patientunitstayid, max(charlson_comorbidity_indexs) as charlson_comorbidity_indexs
        from `db_name.charlson_comorbidity_eicu`
        group by patientunitstayid
    ) cci 
    on sc.patientunitstayid = cci.patientunitstayid
)

-- 9. code status
, code_status_info_0 as (
	select patientunitstayid
  , cplitemoffset
  , ROW_NUMBER() over (partition by patientunitstayid order by cplitemoffset desc) as rn
  , cplitemvalue
	from `physionet-data.eicu_crd.careplangeneral`
	where patientunitstayid in (select patientunitstayid from older_study_cohort_eicu)
  and cplgroup = 'Care Limitation'
  and cplitemvalue is not null
  and cplitemoffset >= -24*60
  and cplitemoffset <= 24*60
)

, code_status_info as (
  select patientunitstayid
	, case
	when value in ('Advance directives', 'Comfort measures only', 'Do not resuscitate'
                , 'No CPR', 'No augmentation of care', 'No blood draws', 'No blood products'
                , 'No cardioversion', 'No intubation', 'No vasopressors/inotropes') then 1
  else 0 end as code_status
  , 1 as code_status_eva_flag
	from code_status_info_0
	where rn = 1  
)

, exclude_id_vt_info as (
  select patientunitstayid
  from (
      select patientunitstayid
      , case when heart_rate_mean > 0 then 0 else 1 end as hr_flag
      , case when mbp_mean > 0 then 0 else 1 end as mbp_flag
      , case when resp_rate_mean > 0 then 0 else 1 end as rr_flag
      , case when gcs_min > 0 then 0 else 1 end as gcs_flag
      , case when temperature_mean > 0 then 0 else 1 end as t_flag
      , case when spo2_min > 0 then 0 else 1 end as spo2_flag
      , case when sbp_mean > 0 then 0 else 1 end as sbp_flag    
      from vital_gcs_uo_info
  )
  where (hr_flag + mbp_flag + rr_flag + gcs_flag + t_flag + spo2_flag + sbp_flag) > 0
)

select sc.patientunitstayid
, uniquepid, patienthealthsystemstayid
, ROW_NUMBER() OVER (ORDER BY sc.patientunitstayid) as id
, first_careunit, los_icu_day, los_hospital_day
, case when death_hosp = 1 then 1 else 0 end as death_hosp
, case when death_hosp = 1 then deathtime_icu_hour else null end as deathtime_icu_hour
, los_icu_admit_day as pre_icu_los_day
, admission_type, ethnicity, gender, age
, height, weight, bmi, hospitaldischargeyear as anchor_year_group 

, case 
when admission_type not in ('URGENT', 'EMERGENCY') and surgical = 1 then 1
when admission_type is null or surgical is null then null
else 0 end as electivesurgery

, case when vent = 1 then 1 else 0 end as vent

, creatinine_max, bilirubin_max, platelet_min
, bun_max, wbc_max, glucose_max, hematocrit_max
, potassium_max, sodium_max
, bicarbonate_min, bicarbonate_max
, pao2fio2ratio_novent, pao2fio2ratio_vent, albumin_min
, alt_max, ast_max, alp_max, pt_max, ptt_max, inr_min, chloride_min
, pao2_min, paco2_max, lactate_max, baseexcess_min, fio2_max
, troponin_max, lymphocytes_max, lymphocytes_min, neutrophils_min
, magnesium_max, fibrinogen_min, bnp_max, aniongap_max

, gcs_min, gcseyes, gcsmotor, gcsverbal
, urineoutput, spo2_min
, heart_rate_mean, mbp_mean, sbp_mean
, resp_rate_mean, temperature_mean

, case when rate_norepinephrine > 0 then 1 else 0 end as norepinephrine
, case when rate_epinephrine > 0 then 1 else 0 end as epinephrine
, case when rate_dopamine > 0 then 1 else 0 end as dopamine
, case when rate_dobutamine > 0 then 1 else 0 end as dobutamine

, case when stand = 1 then 1 else 0 end as activity_stand
, case when sit = 1 and (stand is null or stand = 0) then 1 else 0 end as activity_sit
, case when bed = 1 and (stand is null or stand = 0) and (sit is null or sit = 0) then 1 else 0 end as activity_bed
, case when activity_eva_flag = 1 then 1 else 0 end as activity_eva_flag

, apache_iv, apache_iv_prob, predictediculos_iv, predictedhospitallos_iv
, apache_iva, apache_iva_prob, predictediculos_iva, predictedhospitallos_iva
, oasis, 1 / (1 + exp(- (-6.1746 + 0.1275*(oasis) ))) as oasis_prob
, saps, 1 / (1 + exp(- (-7.7631 + 0.0737*(saps) + 0.9971*(ln(saps + 1))) )) as saps_prob
, sofa, 1 / (1 + exp(- (-3.3890 + 0.2439*(sofa) ))) as sofa_prob
, apsiii, 1 / (1 + exp(- (-4.4360 + 0.04726*(apsiii) ))) as apsiii_prob
, cci_score

, case when cs.code_status = 1 then 1 else 0 end as code_status
, case when cs.code_status_eva_flag = 1 then 1 else 0 end as code_status_eva_flag

, hospitalid, teachingstatus, region

from older_study_cohort_eicu sc
left join surgflag_info si 
on sc.patientunitstayid = si.patientunitstayid
left join vent_info vi
on sc.patientunitstayid = vi.patientunitstayid
left join lab_info lab
on sc.patientunitstayid = lab.patientunitstayid
left join vital_gcs_uo_info vgu
on sc.patientunitstayid = vgu.patientunitstayid
left join vaso_mv_info vm
on sc.patientunitstayid = vm.patientunitstayid
left join activity_info ai
on sc.patientunitstayid = ai.patientunitstayid
left join score_info sci 
on sc.patientunitstayid = sci.patientunitstayid
left join code_status_info cs 
on sc.patientunitstayid = cs.patientunitstayid
where sc.patientunitstayid not in (select patientunitstayid from exclude_id_vt_info)
order by sc.patientunitstayid;




drop table if exists `db_name.older_study_eicu`;
create table `db_name.older_study_eicu` as
select (id + 50000) as id -- to avoid overlap id with mimimc, mimic counts 41412
, uniquepid, patienthealthsystemstayid
, first_careunit, los_icu_day, los_hospital_day
, death_hosp, deathtime_icu_hour
, pre_icu_los_day
, admission_type, ethnicity, gender, age
, height, weight, bmi, anchor_year_group 
, electivesurgery, vent
, creatinine_max, bilirubin_max, platelet_min
, bun_max, wbc_max, glucose_max, hematocrit_max
, potassium_max, sodium_max
, bicarbonate_min, bicarbonate_max
, pao2fio2ratio_novent, pao2fio2ratio_vent, albumin_min
, alt_max, ast_max, alp_max, pt_max, ptt_max, inr_min
, chloride_min
, pao2_min, paco2_max, lactate_max, baseexcess_min, fio2_max
, troponin_max, lymphocytes_max, lymphocytes_min, neutrophils_min
, magnesium_max, fibrinogen_min, bnp_max, aniongap_max
, gcs_min--, gcseyes, gcsmotor, gcsverbal
, urineoutput, spo2_min
, heart_rate_mean, mbp_mean, sbp_mean
, resp_rate_mean, temperature_mean
, norepinephrine, epinephrine
, dopamine, dobutamine
, activity_stand, activity_sit, activity_bed, activity_eva_flag
, apache_iv, apache_iv_prob, predictediculos_iv, predictedhospitallos_iv
, apache_iva, apache_iva_prob, predictediculos_iva, predictedhospitallos_iva
, oasis, oasis_prob
, saps, saps_prob
, sofa, sofa_prob
, apsiii, apsiii_prob
, cci_score
, code_status, code_status_eva_flag
, hospitalid, teachingstatus, region
from `db_name.older_study_eicu_initial1`
order by id;

drop table if exists `db_name.older_study_eicu_initial`;