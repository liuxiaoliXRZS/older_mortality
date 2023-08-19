-- This code is used to prepare the older study data based on MIMIC-III dataset
-- By Xiaoli Liu
-- 2021.10.05
-- 2021.12.24/27 change
-- 2022.12.12 change (add readmission info)

-- criteria
-- age >= 65;
-- first icu admission
-- los icu day >= 1d
-- existing records of heart rate, respiratory rate, map, sbp, gcs, T, spo2



-- acquire the height information
drop table if exists `db_name.heightinfo_mimiciii`;
create table `db_name.heightinfo_mimiciii` as
with ce0 as
(
    SELECT
      c.icustay_id
      , c.charttime
      , case
        -- convert inches to centimetres
          when itemid in (920, 1394, 4187, 3486)
              then valuenum * 2.54
            else valuenum
        end as Height
    FROM `physionet-data.mimiciii_clinical.chartevents` c
    inner join `physionet-data.mimiciii_clinical.icustays` ie
        on c.hadm_id = ie.hadm_id
        --and c.charttime <= DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
        --and c.charttime > DATETIME_SUB(ie.intime, INTERVAL '1' DAY) -- some fuzziness for admit time
    WHERE c.valuenum IS NOT NULL
    AND c.itemid in (226730,920, 1394, 4187, 3486,3485,4188) -- height
    AND c.valuenum != 0
    -- exclude rows marked as error
    AND (c.error IS NULL OR c.error = 0)
)
, ce as
(
    SELECT
        icustay_id
        , charttime
        -- extract the median height from the chart to add robustness against outliers
        , AVG(height) as Height_chart
    from ce0
    where height > 100
    group by icustay_id, charttime
)
-- requires the echo-data.sql query to run
-- this adds heights from the free-text echo notes
, echo as
(
    select
        ec.subject_id
        -- all echo heights are in inches
        , 2.54*AVG(height) as Height_Echo
    from `physionet-data.mimiciii_derived.echo_data` ec
    inner join `physionet-data.mimiciii_clinical.icustays` ie
        on ec.subject_id = ie.subject_id
        and ec.hadm_id = ie.hadm_id
        --and ec.charttime < DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
    where height is not null
    and height*2.54 > 100
    group by ec.subject_id
)

select
    ie.icustay_id
    , ce.charttime
    , coalesce(ce.Height_chart, ec.Height_Echo) as height

    -- components
    , ce.height_chart
    , ec.height_echo
FROM `physionet-data.mimiciii_clinical.icustays` ie

-- filter to only adults
inner join `physionet-data.mimiciii_clinical.patients` pat
    on ie.subject_id = pat.subject_id
    and ie.intime > DATETIME_ADD(pat.dob, INTERVAL '1' YEAR)

left join ce
    on ie.icustay_id = ce.icustay_id

left join echo ec
    on ie.subject_id = ec.subject_id;



-- acquire the study cohort basic information
drop table if exists `db_name.older_study_mimiciii`;
create table `db_name.older_study_mimiciii` as

-- MIMIC-III
with older_study_cohort_mimiciii_0 as (
    SELECT icud.subject_id, icud.hadm_id, icud.icustay_id
    , icud.gender, icud.dod as deathtime, icud.admittime, icud.dischtime
    , icud.first_hosp_stay
    , case when icud.admission_age > 90 then 91.4 else icud.admission_age end as age
    , case when icud.ethnicity_grouped in ('unknown', 'other', 'native') then 'other'
    else icud.ethnicity_grouped end as ethnicity
    , icud.intime, icud.outtime
    , round((DATETIME_DIFF(icud.outtime, icud.intime, MINUTE))/1440, 2) as los_icu_day
    , round((DATETIME_DIFF(icud.dischtime, icud.admittime, MINUTE))/1440, 2) as los_hospital_day
    , round((DATETIME_DIFF(icud.intime, icud.admittime, MINUTE))/1440, 2) as los_icu_admit_day
    , ad.admission_type, ie.first_careunit
    FROM `physionet-data.mimiciii_derived.icustay_detail` icud
    left join `physionet-data.mimiciii_clinical.admissions` ad
	on icud.subject_id = ad.subject_id
	and icud.hadm_id = ad.hadm_id
    inner join `physionet-data.mimiciii_clinical.icustays` ie
    on icud.icustay_id = ie.icustay_id
    and ie.dbsource = 'carevue'
    where first_icu_stay is true
    --and first_hosp_stay is true
    and admission_age >= 65
    and round((DATETIME_DIFF(icud.outtime, icud.intime, MINUTE))/1440, 2) >= 1
    and round((DATETIME_DIFF(icud.dischtime, icud.admittime, MINUTE))/1440, 2) >= 1
    -- and icud.ethnicity_grouped not in ('unknown', 'other', 'native')    
)

, older_study_cohort_mimiciii as (
	select icud.subject_id, icud.hadm_id, icud.icustay_id, icud.first_careunit
  	, intime, outtime, los_icu_day
  	, admittime, dischtime, deathtime, los_hospital_day
    , case when deathtime > intime and deathtime <= DATETIME_ADD(dischtime, INTERVAL '1' DAY) then 1
	else 0 end as death_hosp
	, CAST(ceil(DATETIME_DIFF(deathtime, intime, MINUTE)/60) AS INT64) as deathtime_icu_hour
    , los_icu_admit_day, icud.admission_type, icud.ethnicity
  	, gender, age, first_hosp_stay
  	, height, weight
	, case when wt.weight > 0 and ht.height > 0 then round(((10000 * wt.weight)/(ht.height * ht.height)),2) -- weight(kg)/height^2(m)
	else null end as bmi
	from older_study_cohort_mimiciii_0 icud
	left join (
		select icustay_id, round(coalesce(weight, weight_admit),1) as weight 
		from `physionet-data.mimiciii_derived.weight_first_day`
		where coalesce(weight, weight_admit) > 20 -- we think adult weight can not less than 20kg
		and coalesce(weight, weight_admit) < 400
	) wt
	on wt.icustay_id = icud.icustay_id
	left join (
		SELECT icustay_id, height
    , ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY charttime) as rn
    FROM `db_name.heightinfo_mimiciii`
    WHERE height > 120
		AND height < 230
	) ht	 
	on icud.icustay_id = ht.icustay_id
  and rn = 1
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

-- first day ventilation
, vent_info AS (
    SELECT ie.icustay_id
    , MAX(
        CASE WHEN v.icustay_id IS NOT NULL THEN 1 ELSE 0 END
    ) AS vent
    FROM older_study_cohort_mimiciii ie
    LEFT JOIN `physionet-data.mimiciii_derived.ventilation_durations` v
        ON ie.icustay_id = v.icustay_id
        AND (
            v.starttime BETWEEN ie.intime AND DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
        OR v.endtime BETWEEN ie.intime AND DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
        OR v.starttime <= ie.intime AND v.endtime >= DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
        )
    GROUP BY ie.icustay_id
)

, pafi_0 as (
  -- join blood gas to ventilation durations to determine if patient was vent
  select bg.icustay_id, bg.charttime
  , pao2fio2, po2 as pao2, pco2 as paco2, lactate, baseexcess, coalesce(fio2,fio2_chartevents) as fio2
  , case when vd.icustay_id is not null then 1 else 0 end as isvent
  from `physionet-data.mimiciii_derived.blood_gas_first_day_arterial` bg
  left join `physionet-data.mimiciii_derived.ventilation_durations` vd
    on bg.icustay_id = vd.icustay_id
    and bg.charttime >= vd.starttime
    and bg.charttime <= vd.endtime
  where bg.icustay_id in (select icustay_id from older_study_cohort_mimiciii)
  and specimen_pred = 'ART'
  order by bg.icustay_id, bg.charttime
)

, pafi as (
  -- because pafi has an interaction between vent/PaO2:FiO2, we need two columns for the score
  -- it can happen that the lowest unventilated PaO2/FiO2 is 68, but the lowest ventilated PaO2/FiO2 is 120
  -- in this case, the SOFA score is 3, *not* 4.
  select icustay_id
  , min(case when isvent = 0 then pao2fio2 else null end) as pao2fio2ratio_novent
  , min(case when isvent = 1 then pao2fio2 else null end) as pao2fio2ratio_vent
  , min(pao2) as pao2_min
  , max(paco2) as paco2_max
  , max(lactate) as lactate_max
  , min(baseexcess) as baseexcess_min
  , max(fio2) as fio2_max
  from pafi_0
  group by icustay_id
)

, lab_info_extra_0 as (
  SELECT sc.icustay_id
  , CASE
        when itemid in (51002, 51003) then 'troponin'
        when itemid = 50960 then 'magnesium'
        when itemid = 50963 then 'bnp'
        when itemid = 50861 then 'alt'
        when itemid = 50878 then 'ast'       
        when itemid = 50863 then 'alkaline_phosphatase'
        when itemid = 51214 then 'fibrinogen'
        when itemid = 51244 then 'lymphocytes'
        when itemid = 51256 then 'neutrophils'
        when itemid = 3801 then 'ast'
        when itemid = 3802 then 'alt'
      ELSE null
    END AS label
  , -- add in some sanity checks on the values
  -- the where clause below requires all valuenum to be > 0, so these are only upper limit checks
    le.valuenum

  FROM `physionet-data.mimiciii_clinical.labevents` le 
  inner JOIN older_study_cohort_mimiciii sc
  on sc.subject_id = le.subject_id
  and sc.hadm_id = le.hadm_id
  -- AND DATETIME_DIFF(le.charttime, sc.intime, hour) >= -6
  AND le.itemid in (
       51002, 51003, 50960, 50963, 50861, 50878
       , 50863, 51214, 51244, 51256, 3801, 3802
  )
  AND le.valuenum > 0 -- lab values cannot be 0 and cannot be negative
  and DATETIME_DIFF(le.charttime, sc.intime, hour) >= -6
  and DATETIME_DIFF(le.charttime, sc.intime, hour) <= 24
)
        
, lab_info_extra as (
  select icustay_id
  , max(case when label = 'alt' then valuenum else null end) as alt_max
  , max(case when label = 'ast' then valuenum else null end) as ast_max
  , max(case when label = 'alkaline_phosphatase' then valuenum else null end) as alp_max
  , max(case when label = 'troponin' then valuenum else null end) as troponin_max
  , max(case when label = 'lymphocytes' then valuenum else null end) as lymphocytes_max
  , min(case when label = 'lymphocytes' then valuenum else null end) as lymphocytes_min
  , min(case when label = 'neutrophils' then valuenum else null end) as neutrophils_min
  , max(case when label = 'magnesium' then valuenum else null end) as magnesium_max
  , min(case when label = 'fibrinogen' then valuenum else null end) as fibrinogen_min
  , max(case when label = 'bnp' then valuenum else null end) as bnp_max
  from lab_info_extra_0
  group by icustay_id
)

, lab_info as (
    select sc.icustay_id
    , creatinine_max, bilirubin_max, platelet_min
    , bun_max, wbc_max
    , potassium_max
    , sodium_max
    , bicarbonate_min, bicarbonate_max
    , pao2fio2ratio_novent, pao2fio2ratio_vent, albumin_min
    , pao2_min, paco2_max, pf.lactate_max, baseexcess_min, fio2_max
    , glucose_max, hematocrit_max
    , alt_max, ast_max, alp_max, pt_max, ptt_max, inr_min, chloride_min
    , troponin_max, lymphocytes_max, lymphocytes_min, neutrophils_min
    , magnesium_max, fibrinogen_min, bnp_max, aniongap_max

    from older_study_cohort_mimiciii sc 
    left join `physionet-data.mimiciii_derived.labs_first_day` lab
    on sc.icustay_id = lab.icustay_id
    left join pafi pf 
    on sc.icustay_id = pf.icustay_id
    left join lab_info_extra labe
    on sc.icustay_id = labe.icustay_id
)

, vital_gcs_uo_info as (
    select sc.icustay_id, mingcs as gcs_min
    , coalesce(gcsmotor,6) as gcsmotor
    , coalesce(gcsverbal,5) as gcsverbal
    , coalesce(gcseyes,4) as gcseyes
    , round(vital.heartrate_mean,1) as heart_rate_mean
    , round(vital.meanbp_mean,1) as mbp_mean
    , round(vital.sysbp_mean,1) as sbp_mean
    , round(vital.resprate_mean,1) as resp_rate_mean
    , round(vital.tempc_mean,1) as temperature_mean
    , vital.spo2_min
    , case when uo.urineoutput < 0 then 0 else uo.urineoutput end as urineoutput
    from older_study_cohort_mimiciii sc
    left join `physionet-data.mimiciii_derived.gcs_first_day` gcs
    on sc.icustay_id = gcs.icustay_id
    left join `physionet-data.mimiciii_derived.vitals_first_day` vital
    on sc.icustay_id = vital.icustay_id
    left join `physionet-data.mimiciii_derived.urine_output_first_day` uo
    on sc.icustay_id = uo.icustay_id    
)

, wt AS (
  SELECT ie.icustay_id
    -- ensure weight is measured in kg
    , avg(CASE
        WHEN itemid IN (762, 763, 3723, 3580, 226512)
          THEN valuenum
        -- convert lbs to kgs
        WHEN itemid IN (3581)
          THEN valuenum * 0.45359237
        WHEN itemid IN (3582)
          THEN valuenum * 0.0283495231
        ELSE null
      END) AS weight

  FROM older_study_cohort_mimiciii ie
  left join `physionet-data.mimiciii_clinical.chartevents` c
    on ie.icustay_id = c.icustay_id
  WHERE valuenum IS NOT NULL
  AND itemid IN
  (
    762, 763, 3723, 3580,                     -- Weight Kg
    3581,                                     -- Weight lb
    3582,                                     -- Weight oz
    226512 -- Metavision: Admission Weight (Kg)
  )
  AND valuenum != 0
  and charttime between DATETIME_SUB(ie.intime, INTERVAL '1' DAY) and DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
  -- exclude rows marked as error
  AND (c.error IS NULL OR c.error = 0)
  group by ie.icustay_id
)

-- 5% of patients are missing a weight, but we can impute weight using their echo notes
, echo2 as (
  select ie.icustay_id, avg(echo.weight * 0.45359237) as weight
  FROM older_study_cohort_mimiciii ie
  left join `physionet-data.mimiciii_derived.echo_data` echo
    on ie.hadm_id = echo.hadm_id
    and echo.charttime > DATETIME_SUB(ie.intime, INTERVAL '7' DAY)
    and echo.charttime < DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
  group by ie.icustay_id
)

, vaso_cv_info as (
  select ie.icustay_id
    -- case statement determining whether the ITEMID is an instance of vasopressor usage
    , max(case
            when itemid = 30047 then rate / coalesce(wt.weight,ec.weight) -- measured in mcgmin
            when itemid = 30120 then rate -- measured in mcgkgmin ** there are clear errors, perhaps actually mcgmin
            else null
          end) as rate_norepinephrine

    , max(case
            when itemid =  30044 then rate / coalesce(wt.weight,ec.weight) -- measured in mcgmin
            when itemid in (30119,30309) then rate -- measured in mcgkgmin
            else null
          end) as rate_epinephrine

    , max(case when itemid in (30043,30307) then rate end) as rate_dopamine
    , max(case when itemid in (30042,30306) then rate end) as rate_dobutamine

  FROM older_study_cohort_mimiciii ie
  inner join `physionet-data.mimiciii_clinical.inputevents_cv` cv
    on ie.icustay_id = cv.icustay_id and cv.charttime between ie.intime and DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
  left join wt
    on ie.icustay_id = wt.icustay_id
  left join echo2 ec
    on ie.icustay_id = ec.icustay_id
  where itemid in (30047,30120,30044,30119,30309,30043,30307,30042,30306)
  and rate is not null
  group by ie.icustay_id
)

-- 7. activity
, activity_info as (
    select ce.icustay_id
    , max(case 
    when itemid = 31 and value in ('Ambulate', 'Dangle') then 1
    when itemid = 224084 and value in ('Ambulate', 'Dangle') then 1
    else 0 end) as stand
    , max(case 
    when itemid = 31 and value in ('Bed as Chair', 'Chair', 'Commode') then 1
    when itemid = 224084 and value in ('Bed as Chair', 'Chair', 'Commode') then 1
    else 0 end) as sit
    , max(case 
    when itemid = 31 and value in ('Bedrest', 'Lethargic') then 1
    when itemid = 224084 and value in ('Bedrest') then 1 
    else 0 end) as bed
    , max(case 
    when value is not null then 1
    else 0 end) as activity_eva_flag
    from `physionet-data.mimiciii_clinical.chartevents` ce 
    inner join older_study_cohort_mimiciii sc 
    on ce.icustay_id = sc.icustay_id
    where itemid in (31, 224084)
    and ce.charttime >= DATETIME_SUB(sc.intime, INTERVAL '1' DAY)
    and ce.charttime <= DATETIME_ADD(sc.intime, INTERVAL '1' DAY)
    group by ce.icustay_id
)

-- 8. score
-- apsiii, sofa, saps, oasis, cci_score
, score_info as (
  select sc.icustay_id, ap.apsiii, ap.apsiii_prob
  , oa.oasis, oa.oasis_PROB as oasis_prob
  , sp.sapsii as saps,  sp.sapsii_prob as saps_prob
  , sf.sofa, 1 / (1 + exp(- (-3.3890 + 0.2439*(sf.sofa) ))) as sofa_prob
  , cci.charlson_comorbidity_index as cci_score
  from older_study_cohort_mimiciii sc 
  left join `physionet-data.mimiciii_derived.apsiii` ap 
  on sc.icustay_id = ap.icustay_id
  left join `physionet-data.mimiciii_derived.oasis` oa 
  on sc.icustay_id = oa.icustay_id
  left join `physionet-data.mimiciii_derived.sapsii` sp 
  on sc.icustay_id = sp.icustay_id
  left join `physionet-data.mimiciii_derived.sofa` sf 
  on sc.icustay_id = sf.icustay_id
  left join (
    select hadm_id, max(charlson_comorbidity_index) as charlson_comorbidity_index
    from `db_name.charlson_mimiciii`
    group by hadm_id
  ) cci 
  on sc.hadm_id = cci.hadm_id
)

-- 9. code status
, code_status_info_0 as (
	select ce.icustay_id, ce.charttime, ce.value
	, ROW_NUMBER() over (partition by ce.icustay_id order by ce.charttime desc) as rn
	from `physionet-data.mimiciii_clinical.chartevents` ce
	inner join older_study_cohort_mimiciii sc 
	on ce.icustay_id = sc.icustay_id
	where itemid in (128, 223758)
  and ce.value is not null
  AND ce.charttime >= DATETIME_SUB(sc.intime, INTERVAL '1' DAY)
  AND ce.charttime <= DATETIME_ADD(sc.intime, INTERVAL '1' DAY)
)

, code_status_info as (
	select icustay_id
	, case
	when value in ('CPR Not Indicate', 'DNI (do not intubate)'
	, 'DNR (do not resuscitate)', 'DNR / DNI', 'Do Not Intubate'
	, 'Do Not Resuscita'
  , 'Comfort Measures', 'Comfort measures only') then 1
  else 0 end as code_status
  , 1 as code_status_eva_flag
	from code_status_info_0
	where rn = 1
)

, exclude_id_vt_info as (
  select icustay_id
  from (
      select icustay_id
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

, older_study_mimiciii_0 as (
  select sc.subject_id, sc.hadm_id, sc.icustay_id -- 
  , first_careunit, los_icu_day -- , intime, outtime
  , los_hospital_day -- admittime, dischtime, deathtime, 
  , death_hosp
  , case when death_hosp = 1 then deathtime_icu_hour else null end as deathtime_icu_hour
  , case when los_icu_admit_day < 0 then 0 else los_icu_admit_day end as pre_icu_los_day
  , admission_type, ethnicity, '2001 - 2008' as anchor_year_group
  , gender, age --, first_hosp_stay
  , height, weight, bmi

  , case 
  when admission_type = 'ELECTIVE' and surgical = 1 then 1
  when admission_type is null or surgical is null then null
  else 0 end as electivesurgery

  , case when vent = 1 then 1 else 0 end as vent

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

  , apsiii, apsiii_prob, oasis, oasis_prob, saps, saps_prob, sofa, sofa_prob, cci_score

  , case when cs.code_status = 1 then 1 else 0 end as code_status
  , case when cs.code_status_eva_flag = 1 then 1 else 0 end as code_status_eva_flag

  from older_study_cohort_mimiciii sc
  left join surgflag_info si 
  on sc.icustay_id = si.icustay_id
  left join vent_info vi
  on sc.icustay_id = vi.icustay_id
  left join lab_info lab
  on sc.icustay_id = lab.icustay_id
  left join vital_gcs_uo_info vgu
  on sc.icustay_id = vgu.icustay_id
  left join vaso_cv_info vm
  on sc.icustay_id = vm.icustay_id
  left join activity_info ai
  on sc.icustay_id = ai.icustay_id
  left join score_info sci 
  on sc.icustay_id = sci.icustay_id
  left join code_status_info cs 
  on sc.icustay_id = cs.icustay_id
  where sc.icustay_id not in (select icustay_id from exclude_id_vt_info)
  order by sc.icustay_id
)

select distinct * from older_study_mimiciii_0
order by icustay_id;




-- join mimiciii and mimiciv cohorts (there is no overlapping in id)
-- 299984, 200014 (mimiciii) | 39999552, 30000213 (mimiciv)
drop table if exists `db_name.older_study_mimic`;
create table `db_name.older_study_mimic` as

with cohort_info_0 as (
select icustay_id as id, subject_id, hadm_id
, first_careunit, los_icu_day
, los_hospital_day, death_hosp, deathtime_icu_hour, pre_icu_los_day
, admission_type, ethnicity, anchor_year_group
, gender, age, height, weight, bmi
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
, urineoutput, spo2_min, heart_rate_mean
, mbp_mean, sbp_mean, resp_rate_mean, temperature_mean
, norepinephrine, epinephrine, dopamine, dobutamine
, activity_stand, activity_sit, activity_bed, activity_eva_flag
, apsiii, apsiii_prob, oasis, oasis_prob, saps, saps_prob, sofa, sofa_prob, cci_score
, code_status, code_status_eva_flag
from `db_name.older_study_mimiciii`
union all
select stay_id as id, subject_id, hadm_id
, first_careunit, los_icu_day
, los_hospital_day, death_hosp, deathtime_icu_hour, pre_icu_los_day
, admission_type, ethnicity, anchor_year_group
, gender, age, height, weight, bmi
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
, urineoutput, spo2_min, heart_rate_mean
, mbp_mean, sbp_mean, resp_rate_mean, temperature_mean
, norepinephrine, epinephrine, dopamine, dobutamine
, activity_stand, activity_sit, activity_bed, activity_eva_flag
, apsiii, apsiii_prob, oasis, oasis_prob, saps, saps_prob, sofa, sofa_prob, cci_score
, code_status, code_status_eva_flag
from `db_name.older_study_mimiciv`
)

, cohort_info as (
  select *, ROW_NUMBER() OVER (ORDER BY id) as id_new
  from cohort_info_0
)

select id_new as id, subject_id, hadm_id
, first_careunit, los_icu_day
, los_hospital_day, death_hosp, deathtime_icu_hour, pre_icu_los_day
, admission_type, ethnicity, anchor_year_group
, gender, age, height, weight, bmi
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
, urineoutput, spo2_min, heart_rate_mean
, mbp_mean, sbp_mean, resp_rate_mean, temperature_mean
, norepinephrine, epinephrine, dopamine, dobutamine
, activity_stand, activity_sit, activity_bed, activity_eva_flag
, apsiii, apsiii_prob, oasis, oasis_prob, saps, saps_prob, sofa, sofa_prob, cci_score
, code_status, code_status_eva_flag
from cohort_info
order by id_new;