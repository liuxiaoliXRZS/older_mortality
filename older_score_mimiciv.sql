-- This code is used to prepare the older study data based on MIMIC-IV dataset
-- By Xiaoli Liu
-- 2021.10.04 & 2021.12.27
-- 2022.12.12 change (add readmission info)

-- criteria
-- age > 65; 
-- first icu admission
-- los icu day >= 1d
drop table if exists `db_name.older_study_mimiciv`;
create table `db_name.older_study_mimiciv` as

-- MIMIC-IV
with older_study_cohort_mimiciv_0 as (
    SELECT icud.subject_id, icud.hadm_id, icud.stay_id
    , icud.gender, icud.dod as deathtime, icud.admittime, icud.dischtime
    , icud.first_hosp_stay, icud.admission_age as age, icud.ethnicity
    , icud.icu_intime as intime, icud.icu_outtime as outtime, icud.los_icu as los_icu_day
    , round((DATETIME_DIFF(icud.dischtime, icud.admittime, MINUTE))/1440, 2) as los_hospital_day
    , round((DATETIME_DIFF(icud.icu_intime, icud.admittime, MINUTE))/1440, 2) as los_icu_admit_day
    , ad.admission_type, pt.anchor_year_group, ie.first_careunit
    FROM `physionet-data.mimic_derived.icustay_detail` icud
    left join `physionet-data.mimic_core.admissions` ad
	on icud.subject_id = ad.subject_id
	and icud.hadm_id = ad.hadm_id
	left join `physionet-data.mimic_core.patients` pt
	on icud.subject_id = pt.subject_id
    left join `physionet-data.mimic_icu.icustays` ie
    on icud.stay_id = ie.stay_id
    where first_icu_stay is true
    and admission_age >= 65
    and los_icu >= 1
    and los_hospital >= 1
)

, older_study_cohort_mimiciv as (
	select icud.subject_id, icud.hadm_id, icud.stay_id
	, case 
  	when icud.first_careunit in ('Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)') then 'CCU'
  	when icud.first_careunit in ('Neuro Intermediate', 'Neuro Stepdown', 'Neuro Surgical Intensive Care Unit (Neuro SICU)') then 'NICU'
  	when icud.first_careunit in ('Medical Intensive Care Unit (MICU)') then 'MICU'
  	when icud.first_careunit = 'Medical/Surgical Intensive Care Unit (MICU/SICU)' then 'Med-Surg_ICU'
  	when icud.first_careunit = 'Surgical Intensive Care Unit (SICU)' then 'SICU'
  	when icud.first_careunit = 'Trauma SICU (TSICU)' then 'TSICU'
  	else null end as first_careunit
  	, intime, outtime, los_icu_day
  	, admittime, dischtime, deathtime, los_hospital_day
    , case when deathtime > intime and deathtime <= DATETIME_ADD(dischtime, INTERVAL '1' DAY) then 1
	else 0 end as death_hosp
	, CAST(ceil(DATETIME_DIFF(deathtime, intime, MINUTE)/60) AS INT64) as deathtime_icu_hour
    , los_icu_admit_day
  	, case 
    when icud.admission_type in ('AMBULATORY OBSERVATION', 'DIRECT OBSERVATION', 'EU OBSERVATION', 'OBSERVATION ADMIT') then 'OBSERVATION'
    when icud.admission_type in ('ELECTIVE', 'SURGICAL SAME DAY ADMISSION') then 'ELECTIVE'
    when icud.admission_type in ('DIRECT EMER.', 'EW EMER.') then 'EMERGENCY'
    when icud.admission_type in ('URGENT')  then 'URGENT'
    else null end as admission_type
    , case 
    when icud.ethnicity = 'ASIAN' then 'asian'
    when icud.ethnicity = 'BLACK/AFRICAN AMERICAN' then 'black'
    when icud.ethnicity = 'HISPANIC/LATINO' then 'hispanic'
    when icud.ethnicity = 'WHITE' then 'white'
    else 'other' end as ethnicity
    , icud.anchor_year_group
  	, gender, age, first_hosp_stay
  	, height, weight
	, case when wt.weight > 0 and ht.height > 0 then round(((10000 * wt.weight)/(ht.height * ht.height)),2) -- weight(kg)/height^2(m)
	else null end as bmi
	from older_study_cohort_mimiciv_0 icud
	left join (
		select stay_id, round(coalesce(weight, weight_admit),1) as weight 
		from `db_name.first_day_weight_mimiciv`
		where coalesce(weight, weight_admit) > 20 -- we think adult weight can not less than 20kg
		and coalesce(weight, weight_admit) < 400
	) wt
	on wt.stay_id = icud.stay_id
	left join (
		SELECT c.stay_id, round(valuenum,1) as height
		, ROW_NUMBER() OVER (PARTITION BY c.stay_id ORDER BY c.charttime) as rn
    	FROM `physionet-data.mimic_icu.chartevents` c
    	INNER JOIN `physionet-data.mimic_icu.icustays` ie 
    	ON c.stay_id = ie.stay_id
    	WHERE c.valuenum IS NOT NULL
    	AND c.itemid in (226730) -- height
    	AND c.valuenum != 0
		AND c.valuenum > 120 and c.valuenum < 230
	) ht	 
	on icud.stay_id = ht.stay_id
	and ht.rn = 1
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

-- first day ventilation
, vent_info AS
(
    SELECT ie.stay_id
    , MAX(
        CASE WHEN v.stay_id IS NOT NULL THEN 1 ELSE 0 END
    ) AS vent
    FROM `physionet-data.mimic_icu.icustays` ie
    LEFT JOIN `physionet-data.mimic_derived.ventilation` v
        ON ie.stay_id = v.stay_id
        AND (
            v.starttime BETWEEN ie.intime AND DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
        OR v.endtime BETWEEN ie.intime AND DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
        OR v.starttime <= ie.intime AND v.endtime >= DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
        )
        AND v.ventilation_status in ('InvasiveVent', 'Tracheostomy', 'NonInvasiveVent')
    where ie.stay_id in (select stay_id from older_study_cohort_mimiciv)
    GROUP BY ie.stay_id
)

, pafi_0 as
(
  -- join blood gas to ventilation durations to determine if patient was vent
  select ie.stay_id
  , bg.charttime
  -- because pafi has an interaction between vent/PaO2:FiO2, we need two columns for the score
  -- it can happen that the lowest unventilated PaO2/FiO2 is 68, but the lowest ventilated PaO2/FiO2 is 120
  -- in this case, the SOFA score is 3, *not* 4.
  , case when vd.stay_id is null then pao2fio2ratio else null end pao2fio2ratio_novent
  , case when vd.stay_id is not null then pao2fio2ratio else null end pao2fio2ratio_vent
  , po2 as pao2, pco2 as paco2, baseexcess, lactate, coalesce(fio2,fio2_chartevents) as fio2
  FROM `physionet-data.mimic_icu.icustays` ie
  inner join `physionet-data.mimic_derived.bg` bg
    on ie.subject_id = bg.subject_id
  left join `physionet-data.mimic_derived.ventilation` vd
    on ie.stay_id = vd.stay_id
    and bg.charttime >= vd.starttime
    and bg.charttime <= vd.endtime
    and vd.ventilation_status in ('InvasiveVent', 'Tracheostomy', 'Tracheostomy')
  WHERE specimen = 'ART.'
  and ie.stay_id in (select stay_id from older_study_cohort_mimiciv)
)

, pafi as (
    select pf.stay_id, min(pao2fio2ratio_novent) as pao2fio2ratio_novent
    , min(pao2fio2ratio_vent) as pao2fio2ratio_vent
    , min(pao2) as pao2_min
    , max(paco2) as paco2_max
    , max(lactate) as lactate_max
    , min(baseexcess) as baseexcess_min
    , max(fio2) as fio2_max
    from pafi_0 pf 
    inner join older_study_cohort_mimiciv sc 
    on pf.stay_id = sc.stay_id
    and pf.charttime <= DATETIME_ADD(sc.intime, INTERVAL '1' DAY)
    and pf.charttime >= DATETIME_SUB(sc.intime, INTERVAL '6' HOUR)
    group by pf.stay_id
)

, lab_info_extra_0 as (
	select sc.stay_id
    , CASE
        when itemid in (51002, 51003, 52637) then 'troponin'
        when itemid = 50960 then 'magnesium'
        when itemid = 50963 then 'bnp'
        when itemid in (51244, 51688, 51245) then 'lymphocytes'
        when itemid in (51256, 51695) then 'neutrophils'
      ELSE null
    END AS label
    , le.valuenum		
	from `physionet-data.mimic_hosp.labevents` le
	inner join older_study_cohort_mimiciv sc 
	on sc.subject_id = le.subject_id
	and sc.hadm_id = le.hadm_id
    --AND DATETIME_DIFF(le.charttime, sc.intime, hour) >= -6
    AND le.itemid in (
		51002, 51003, 52637, 50960, 50963, 51244
		, 51688, 51245, 51256, 51695
    )
    AND le.valuenum > 0 -- lab values cannot be 0 and cannot be negative
    and  le.charttime <= DATETIME_ADD(sc.intime, INTERVAL '1' DAY)
    and le.charttime >= DATETIME_SUB(sc.intime, INTERVAL '6' HOUR)
)

, lab_info_extra as (
    select stay_id
    , max(case when label = 'troponin' then valuenum else null end) as troponin_max
    , max(case when label = 'lymphocytes' then valuenum else null end) as lymphocytes_max
    , min(case when label = 'lymphocytes' then valuenum else null end) as lymphocytes_min
    , min(case when label = 'neutrophils' then valuenum else null end) as neutrophils_min
    , max(case when label = 'magnesium' then valuenum else null end) as magnesium_max
    , max(case when label = 'bnp' then valuenum else null end) as bnp_max
    from lab_info_extra_0
    group by stay_id
)

, lab_info as (
    select sc.stay_id
    , creatinine_max, bilirubin_total_max as bilirubin_max, platelets_min as platelet_min
    , bun_max, wbc_max, glucose_max, hematocrit_max
    , potassium_max
    , sodium_max
    , bicarbonate_min, bicarbonate_max
    , pao2fio2ratio_novent, pao2fio2ratio_vent, albumin_min
    , alt_max, ast_max, alp_max, pt_max, ptt_max, inr_min, chloride_min
    , pao2_min, paco2_max, lactate_max, baseexcess_min, fio2_max
    , troponin_max, lymphocytes_max, lymphocytes_min, neutrophils_min
    , magnesium_max, fibrinogen_min, bnp_max, aniongap_max
    from older_study_cohort_mimiciv sc 
    left join `physionet-data.mimic_derived.first_day_lab` lab
    on sc.stay_id = lab.stay_id
    left join pafi pf 
    on sc.stay_id = pf.stay_id
    left join lab_info_extra labe
    on sc.stay_id = labe.stay_id
)

, vital_gcs_uo_info as (
    select sc.stay_id, gcs_min
    , coalesce(gcs_motor,6) as gcsmotor
    , coalesce(gcs_verbal,5) as gcsverbal
    , coalesce(gcs_eyes,4) as gcseyes
    , round(vital.heart_rate_mean, 1) as heart_rate_mean
    , round(vital.mbp_mean, 1) as mbp_mean
    , round(vital.sbp_mean, 1) as sbp_mean
    , round(vital.resp_rate_mean, 1) as resp_rate_mean
    , round(vital.temperature_mean, 1) as temperature_mean
    , case when uo.urineoutput < 0 then 0 else uo.urineoutput end as urineoutput
    , vital.spo2_min
    from older_study_cohort_mimiciv sc
    left join `physionet-data.mimic_derived.first_day_gcs` gcs
    on sc.stay_id = gcs.stay_id
    left join `physionet-data.mimic_derived.first_day_vitalsign` vital
    on sc.stay_id = vital.stay_id
    left join `physionet-data.mimic_derived.first_day_urine_output` uo
    on sc.stay_id = uo.stay_id    
)

, vaso_stg as (
  select ie.stay_id, 'norepinephrine' AS treatment, vaso_rate as rate
  FROM older_study_cohort_mimiciv ie
  INNER JOIN `physionet-data.mimic_derived.norepinephrine` mv
    ON ie.stay_id = mv.stay_id
    -- AND mv.starttime >= DATETIME_SUB(ie.intime, INTERVAL '6' HOUR)
    AND mv.starttime <= DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
    and mv.endtime > ie.intime
  UNION ALL
  select ie.stay_id, 'epinephrine' AS treatment, vaso_rate as rate
  FROM older_study_cohort_mimiciv ie
  INNER JOIN `physionet-data.mimic_derived.epinephrine` mv
    ON ie.stay_id = mv.stay_id
    -- AND mv.starttime >= DATETIME_SUB(ie.intime, INTERVAL '6' HOUR)
    AND mv.starttime <= DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
    and mv.endtime > ie.intime
  UNION ALL
  select ie.stay_id, 'dobutamine' AS treatment, vaso_rate as rate
  FROM older_study_cohort_mimiciv ie
  INNER JOIN `physionet-data.mimic_derived.dobutamine` mv
    ON ie.stay_id = mv.stay_id
    -- AND mv.starttime >= DATETIME_SUB(ie.intime, INTERVAL '6' HOUR)
    AND mv.starttime <= DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
    and mv.endtime > ie.intime
  UNION ALL
  select ie.stay_id, 'dopamine' AS treatment, vaso_rate as rate
  FROM older_study_cohort_mimiciv ie
  INNER JOIN `physionet-data.mimic_derived.dopamine` mv
    ON ie.stay_id = mv.stay_id
    -- AND mv.starttime >= DATETIME_SUB(ie.intime, INTERVAL '6' HOUR)
    AND mv.starttime <= DATETIME_ADD(ie.intime, INTERVAL '1' DAY)
    and mv.endtime > ie.intime
)

, vaso_mv_info AS (
    SELECT
    ie.stay_id
    , max(CASE WHEN treatment = 'norepinephrine' THEN rate ELSE NULL END) as rate_norepinephrine
    , max(CASE WHEN treatment = 'epinephrine' THEN rate ELSE NULL END) as rate_epinephrine
    , max(CASE WHEN treatment = 'dopamine' THEN rate ELSE NULL END) as rate_dopamine
    , max(CASE WHEN treatment = 'dobutamine' THEN rate ELSE NULL END) as rate_dobutamine
  from older_study_cohort_mimiciv ie
  LEFT JOIN vaso_stg v
      ON ie.stay_id = v.stay_id
  GROUP BY ie.stay_id
)

-- 7. activity
, activity_info as (
    select ce.stay_id
    , max(case 
    when itemid = 224084 and value in ('Ambulate', 'Dangle') then 1
    when itemid = 229319 and value in ('5 - Stand - >/= One minute', '6 - Walk - 10+ Steps', 
        '7 - Walk - 25+ Feet', '8 - Walk - 250+ Feet') then 1
    when itemid = 229321 and value in ('5 - Stand - >/= One minute', '6 - Walk - 10+ Steps', 
        '7 - Walk - 25+ Feet', '8 - Walk - 250+ Feet') then 1
    when itemid = 229742 and value in ('5 - Stand - >/= One minute', '6 - Walk - 10+ Steps', 
        '7 - Walk - 25+ Feet', '8 - Walk - 250+ Feet') then 1
    else 0 end) as stand
    , max(case 
    when itemid = 224084 and value in ('Bed as Chair', 'Chair','Commode') then 1
    when itemid = 229319 and value in ('3 - Bed - Sit at edge of bed', '4 - Chair - Transfer to chair') then 1
    when  itemid = 229321 and value in ('2c - Lift to chair/bed', '3 - Bed - Sit at edge of bed', 
        '4 - Chair - Transfer to chair/bed') then 1
    when itemid = 229742 and value in ('2c - Lift to chair/bed', '3 - Bed - Sit at edge of bed', 
        '4 - Chair - Transfer to chair/bed') then 1
    else 0 end) as sit
    , max(case 
    when itemid = 224084 and value in ('Bedrest') then 1
    when itemid = 229319 and value in ('1 - Bedrest - Only lying', '2 - Bed - Turn self / Bed activity') then 1
    when itemid = 229321 and value in ('1 - Bedrest - Only lying', '2a - Passive or Active ROM', 
        '2b - Turning in bed') then 1
    when itemid = 229742 and value in ('1 - Bedrest - Only lying', '2a - Passive or Active ROM', 
        '2b - Turning in bed') then 1
    else 0 end) as bed
    , max(case 
    when value is not null then 1
    else 0 end) as activity_eva_flag
    from `physionet-data.mimic_icu.chartevents` ce 
    inner join older_study_cohort_mimiciv sc 
    on ce.stay_id = sc.stay_id
    where itemid in (224084, 229319, 229321, 229742)
    and ce.charttime >= DATETIME_SUB(sc.intime, INTERVAL '1' DAY)
    and ce.charttime <= DATETIME_ADD(sc.intime, INTERVAL '1' DAY)
    group by ce.stay_id
)

-- 8. score
-- cci, sofa, saps, oasis, apsiii
, score_info as (
    select sc.stay_id
    , ap.apsiii, ap.apsiii_prob
    , sf.sofa, 1 / (1 + exp(- (-3.3890 + 0.2439*(sf.sofa) ))) as sofa_prob
    , oa.oasis, oa.oasis_prob
    , sp.sapsii as saps, sp.sapsii_prob as saps_prob
    , cci.charlson_comorbidity_index as cci_score
    from older_study_cohort_mimiciv sc 
    left join (
        select hadm_id, max(charlson_comorbidity_index) as charlson_comorbidity_index
        from `physionet-data.mimic_derived.charlson`
        group by hadm_id
    ) cci 
    on sc.hadm_id = cci.hadm_id
    left join `physionet-data.mimic_derived.apsiii` ap 
    on sc.stay_id = ap.stay_id
    left join `physionet-data.mimic_derived.first_day_sofa` sf 
    on sc.stay_id = sf.stay_id
    left join `physionet-data.mimic_derived.oasis` oa 
    on sc.stay_id = oa.stay_id
    left join `physionet-data.mimic_derived.sapsii` sp 
    on sc.stay_id = sp.stay_id
)

-- 9. code status
, code_status_info_0 as (
	select ce.stay_id, CAST(ceil(DATETIME_DIFF(ce.charttime, sc.intime, MINUTE)/60) AS INT64) as hr, ce.value
	, ROW_NUMBER() over (partition by ce.stay_id order by ce.charttime desc) as rn
	from `physionet-data.mimic_icu.chartevents` ce
	inner join older_study_cohort_mimiciv sc 
	on ce.stay_id = sc.stay_id
	where itemid in (223758, 228687)
  and ce.value is not null
	and CAST(ceil(DATETIME_DIFF(ce.charttime, sc.intime, MINUTE)/60) AS INT64) >= -24
	and CAST(ceil(DATETIME_DIFF(ce.charttime, sc.intime, MINUTE)/60) AS INT64) <= 24
)

, code_status_info as (
	select stay_id
	, case
	when value in ('DNAR (Do Not Attempt Resuscitation)  [DNR]'
		, 'DNAR (Do Not Attempt Resuscitation) [DNR] / DNI'
		, 'DNI (do not intubate)'
		, 'DNR (do not resuscitate)'
		, 'DNR / DNI'
        , 'Comfort measures only') then 1
	else 0 end as code_status
    , 1 as code_status_eva_flag
	from code_status_info_0
	where rn = 1
)

, exclude_id_vt_info as (
  select stay_id
  from (
      select stay_id
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

, older_study_mimiciv_0 as (
    select sc.subject_id, sc.hadm_id, sc.stay_id 
    , first_careunit, los_icu_day -- , intime, outtime
    , los_hospital_day -- admittime, dischtime, deathtime, 
    , death_hosp
    , case when death_hosp = 1 then deathtime_icu_hour else null end as deathtime_icu_hour
    , case when los_icu_admit_day < 0 then 0 else los_icu_admit_day end as pre_icu_los_day
    , admission_type, ethnicity, anchor_year_group
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
    , alt_max, ast_max, alp_max, pt_max, ptt_max, inr_min, chloride_min
    , pao2_min, paco2_max, lactate_max, baseexcess_min, fio2_max
    , troponin_max, lymphocytes_max, lymphocytes_min, neutrophils_min
    , magnesium_max, fibrinogen_min, bnp_max, aniongap_max

    , gcs_min, gcseyes, gcsmotor, gcsverbal
    , urineoutput, spo2_min, heart_rate_mean
    , mbp_mean, sbp_mean, resp_rate_mean, temperature_mean

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

    from older_study_cohort_mimiciv sc
    left join surgflag_info si 
    on sc.stay_id = si.stay_id
    left join vent_info vi
    on sc.stay_id = vi.stay_id
    left join lab_info lab
    on sc.stay_id = lab.stay_id
    left join vital_gcs_uo_info vgu
    on sc.stay_id = vgu.stay_id
    left join vaso_mv_info vm
    on sc.stay_id = vm.stay_id
    left join activity_info ai
    on sc.stay_id = ai.stay_id
    left join score_info sci 
    on sc.stay_id = sci.stay_id
    left join code_status_info cs 
    on sc.stay_id = cs.stay_id
    where sc.stay_id not in (select stay_id from exclude_id_vt_info)
    order by sc.stay_id
)

select distinct * from older_study_mimiciv_0 
order by stay_id;