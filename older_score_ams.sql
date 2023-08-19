-- This code is used to prepare the older study data based on MIMIC-IV dataset
-- By Xiaoli Liu
-- 2021.12.29
-- 2022.12.12 change (add readmission info)

-- criteria
-- age > 65; 
-- first icu admission
-- los icu day >= 1d

drop table if exists `db_name.older_study_ams_initial`;
create table `db_name.older_study_ams_initial` as

with older_study_cohort_ams_0 as (
  SELECT patientid, admissionid, admissioncount, admissionyeargroup
  , case 
  when location = 'MC' then 'medium care unit'
  when location in ('IC', 'IC&MC', 'MC&IC') then 'intensive care unit'
  else null end as first_careunit
  , case 
  when urgency = true then 'unplanned'
  when urgency = False then 'planned'
  else null end as admission_type
  , admittedat
  , dischargedat
  , dateofdeath
  , round((dischargedat - admittedat)/(24*3600000),3) as los_icu_day
  , case when dateofdeath > admittedat and abs(round((dischargedat - dateofdeath)/(24*3600000),3)) <= 2 
  then ceil(dateofdeath/3600000) else null end as deathtime_icu_hour
  , case when dateofdeath > admittedat and abs(round((dischargedat - dateofdeath)/(24*3600000),3)) <= 2 then 1 else 0 end as death_hosp
  , case 
  when gender = 'Man' then 'M'
  when gender = 'Vrouw' then 'F'
  else null end as gender
  , agegroup, weightgroup, lengthgroup as heightgroup
  , lead(admittedat) over (partition by patientid order by admissioncount) as admittedat_second
  FROM `physionet-data.amsterdamdb.admissions`
  where agegroup in ('70-79', '80+') 
)

-- next icu admission diff over 7days will be thought as readmission
, older_study_cohort_ams_1 as (
  select *
  from (
    select *
    , case when admittedat_second is null then 1
    when admittedat_second is not null and round((admittedat_second - admittedat)/(24*3600000),3) > 7 then 1
    else 0 end as flag
    from older_study_cohort_ams_0
  )
  where flag = 1
)

, older_study_cohort_ams as (
  select sc.patientid, sc.admissionid, first_careunit
  , gender, agegroup, death_hosp
  , admittedat, admissionyeargroup, dischargedat
  -- , ethnicity, icu_admit_day -- unknown
  -- , bmi -- need to calculate separately
  , admission_type, heightgroup, weightgroup
  , los_icu_day, deathtime_icu_hour --, icu_los_hours
  from older_study_cohort_ams_1 sc 
  where los_icu_day >= 1
  order by sc.admissionid
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

select sc.patientid, sc.admissionid, first_careunit
, gender, agegroup, death_hosp
, admittedat, admissionyeargroup, dischargedat
, admission_type, heightgroup, weightgroup
, los_icu_day, deathtime_icu_hour
, case when admission_type = 'planned' and surgical = 1 then 1
when admission_type is null or surgical is null then null
else 0 end as electivesurgery
from older_study_cohort_ams sc
left join surgflag_info sg 
on sc.admissionid = sg.admissionid 
order by sc.admissionid;



drop table if exists `db_name.older_study_ams_lab`;
create table `db_name.older_study_ams_lab` as
-- first day ventilation
with older_study_cohort_ams as (
    select *
    from `db_name.older_study_ams_initial`
)

, vent_info_10 as (
  SELECT 
      l.admissionid
      , CAST(ceil((l.measuredat - sc.admittedat)/(1000*60*60)) AS INT64) as hr
  FROM `physionet-data.amsterdamdb.listitems` l
  inner join older_study_cohort_ams sc 
  on l.admissionid = sc.admissionid 
  WHERE 
      (
          itemid = 9534  --Type beademing Evita 1
          AND valueid IN (
              1, --IPPV Intermittent positive-pressure ventilation
              2, --IPPV_Assist
              --3, --CPPV Continuous positive pressure ventilation
              --4, --CPPV_Assist
              5, --SIMV Synchronized intermittent mandatory ventilation
              6, --SIMV_ASB
              --7, --ASB assisted spontaneous breathing
              --8, --CPAP continuous positive airway pressure
              --9, --CPAP_ASB
              10, --MMV Mandatory minute ventilation
              11, --MMV_ASB
              --12, --BIPAP Biphasic positive airway pressure
              13 --Pressure Controled
          )
      )
      OR (
          itemid = 6685 --Type Beademing Evita 4
          AND valueid IN (
              --1, --CPPV
              --3, --ASB
              --5, --CPPV/ASSIST
              6, --SIMV/ASB
              8, --IPPV
              9, --IPPV/ASSIST
              --10, --CPAP
              --11, --CPAP/ASB
              12, --MMV
              13 --, MMV/ASB
              --14, --BIPAP
              --20, --BIPAP-SIMV/ASB
              --22 --BIPAP/ASB
          )
      )
      OR (
          itemid IN (
              12290, --Ventilatie Mode (Set) - Servo-I and Servo-U ventilators
              12347 --Ventilatie Mode (Set) (2) Servo-I and Servo-U ventilators
          )
          AND valueid IN (
              --IGNORE: 1, --Stand By
               2, --PC
               3, --VC
               4, --PRVC
               5, --VS
               6, --SIMV(VC)+PS
               7, --SIMV(PC)+PS
               --8, --PS/CPAP
               --9, --Bi Vente
               10, --PC (No trig)
               11, --VC (No trig)
               12, --PRVC (No trig)
               --13, --PS/CPAP (trig)
               14, --VC (trig)
               15 --, PRVC (trig)
               --16, --PC in NIV
               --17, --PS/CPAP in NIV
               --18 --NAVA
          )
      )
  group by l.admissionid, CAST(ceil((l.measuredat - sc.admittedat)/(1000*60*60)) AS INT64)
)

, vent_info_1 AS
(
    SELECT ie.admissionid
    , MAX(
        CASE WHEN v.admissionid IS NOT NULL THEN 1 ELSE 0 END
    ) AS vent
    FROM older_study_cohort_ams ie
    LEFT JOIN vent_info_10 v
    ON ie.admissionid = v.admissionid
    AND v.hr >= 0 and v.hr < 24
    GROUP BY ie.admissionid
)

, vent_cohort_time_0 as (
    select po.admissionid
    , CAST(ceil((po.start - sc.admittedat)/(1000*60*60)) AS INT64) as start_hr
    , CAST(ceil((po.stop - sc.admittedat)/(1000*60*60)) AS INT64) as stop_hr
    from `physionet-data.amsterdamdb.processitems` po
    inner join older_study_cohort_ams sc
    on sc.admissionid = po.admissionid 
    where itemid = 12634
    and po.admissionid in (select admissionid from vent_info_1 where vent = 1)  
)

, vent_cohort_time_1 as
(
  select admissionid
  , GENERATE_ARRAY(start_hr, stop_hr) as hr
  from vent_cohort_time_0
)

, vent_cohort_time as (
  select admissionid, hr
  from vent_cohort_time_1 
  cross join unnest(vent_cohort_time_1.hr) as hr
)

, fio2_bg_0 AS (
    SELECT n.admissionid
    , CAST(ceil((n.measuredat - sc.admittedat)/(1000*60*60)) AS INT64) as hr
    , l.valueid, l.value AS O2_device
    , CASE WHEN n.itemid IN 
    (
        --FiO2 settings on respiratory support
        6699, --FiO2 %: setting on Evita ventilator
        12279, --O2 concentratie --measurement by Servo-i/Servo-U ventilator
        12369, --SET %O2: used with BiPap Vision ventilator
        16246 --Zephyros FiO2: Non-invasive ventilation
    ) THEN TRUE ELSE FALSE END AS ventilatory_support
    , CASE WHEN n.itemid IN 
    (
        --FiO2 settings on respiratory support
        6699, --FiO2 %: setting on Evita ventilator
        12279, --O2 concentratie --measurement by Servo-i/Servo-U ventilator
        12369, --SET %O2: used with BiPap Vision ventilator
        16246 --Zephyros FiO2: Non-invasive ventilation
    ) THEN 
        CASE WHEN NOT n.value IS NULL THEN n.value --use the settings
        ELSE 0.21 END
      ELSE -- estimate the FiO2
        CASE 
        WHEN l.valueid IN (
            2, -- Nasaal
            7 --O2-bril
        ) THEN CASE 
            WHEN n.value >= 1 AND n.value < 2 THEN 0.22
            WHEN n.value >= 2 AND n.value < 3 THEN 0.25
            WHEN n.value >= 3 AND n.value < 4 THEN 0.27
            WHEN n.value >= 4 AND n.value < 5 THEN 0.30
            WHEN n.value >= 5 THEN 0.35
            ELSE 0.21 END
        WHEN l.valueid IN (
            1, --Diep Nasaal
            3, --Kapje
            8, --Kinnebak
            9, --Nebulizer
            4, --Kunstneus
            18, --Spreekcanule
            19 --Spreekklepje
        ) THEN CASE
            WHEN n.value >= 1 AND n.value < 2 THEN 0.22 -- not defined by NICE
            WHEN n.value >= 2 AND n.value < 3 THEN 0.25
            WHEN n.value >= 3 AND n.value < 4 THEN 0.27
            WHEN n.value >= 4 AND n.value < 5 THEN 0.30
            WHEN n.value >= 5 AND n.value < 6 THEN 0.35
            WHEN n.value >= 6 AND n.value < 7 THEN 0.40
            WHEN n.value >= 7 AND n.value < 8 THEN 0.45
            WHEN n.value >= 8 THEN 0.50
            ELSE 0.21 END
        WHEN l.valueid IN (
            10, --Waterset
            11, --Trach.stoma
            13, --Ambu
            14, --Guedel
            15, --DL-tube
            16, --CPAP
            17 --Non-Rebreathing masker
        ) THEN CASE
            WHEN n.value >= 6 AND n.value < 7 THEN 0.60
            WHEN n.value >= 7 AND n.value < 8 THEN 0.70
            WHEN n.value >= 8 AND n.value < 9 THEN 0.80
            WHEN n.value >= 9 AND n.value < 10 THEN 0.85
            WHEN n.value >= 10 THEN 0.90
            ELSE 0.21 END
        WHEN l.valueid IN (
            12 --B.Lucht
        ) THEN 0.21 ELSE 0.21 END
    END AS fio2
    FROM `physionet-data.amsterdamdb.numericitems` n
    INNER JOIN older_study_cohort_ams sc 
    ON n.admissionid = sc.admissionid
    LEFT JOIN `physionet-data.amsterdamdb.listitems` l 
    ON n.admissionid = l.admissionid 
    AND n.measuredat = l.measuredat 
    AND l.itemid = 8189 -- Toedieningsweg (Oxygen device)
    WHERE n.itemid IN (
        --Oxygen Flow settings without respiratory support
        8845, -- O2 l/min
        10387, --Zuurstof toediening (bloed)
        18587, --Zuurstof toediening

        --FiO2 settings on respiratory support
        6699, --FiO2 %: setting on Evita ventilator
        12279, --O2 concentratie --measurement by Servo-i/Servo-U ventilator
        12369, --SET %O2: used with BiPap Vision ventilator
        16246 --Zephyros FiO2: Non-invasive ventilation
    )
    --measurements within 24 hours of ICU stay:
    AND (n.measuredat - sc.admittedat) >= -1000*60*60*6
    and (n.measuredat - sc.admittedat) < 1000*60*60*24
    AND n.value > 0 --ignore stand by values from Evita ventilator
)

, fio2_bg_1 as (
    select admissionid, hr, valueid, O2_device
    , ventilatory_support
    , case when fio2 > 100 then null  
    when fio2 >= 20 and fio2 <= 100 then fio2/100 
    else 
        case 
        when fio2 > 1 then null 
        when fio2 < 0.2 then null 
        else fio2 end 
    end as fio2
    from fio2_bg_0
)

, fio2_bg as (
    select admissionid, hr, max(100*fio2) as fio2
    from fio2_bg_1
    group by admissionid, hr
)

, pao2_bg_1 as (
    SELECT 
        pao2.admissionid,
        CASE pao2.unitid 
            WHEN 152 THEN pao2.value * 7.50061683 -- Conversion: kPa to mmHg
            ELSE pao2.value 
        END AS pao2,
        f.value AS specimen,
        CASE 
            WHEN pao2.registeredby NOT LIKE '%Systeem%' THEN TRUE
            ELSE FALSE
        END AS manual_entry,
        pao2.measuredat, 
        CAST(ceil((pao2.measuredat - sc.admittedat)/(1000*60*60)) AS INT64) as hr
    FROM `physionet-data.amsterdamdb.numericitems` pao2
    INNER JOIN older_study_cohort_ams sc ON
        pao2.admissionid = sc.admissionid
    LEFT JOIN `physionet-data.amsterdamdb.freetextitems` f ON
        pao2.admissionid = f.admissionid AND
        pao2.measuredat = f.measuredat AND
        f.itemid = 11646 --Afname (bloed): source of specimen
    WHERE 
        pao2.itemid IN (
            7433, --PO2
            9996, --PO2 (bloed)
            21214 --PO2 (bloed) - kPa
        ) 
    --measurements within 24 hours of ICU stay (use 30 minutes before admission to allow for time differences):
    AND (pao2.measuredat - sc.admittedat) >= -1000*60*60*6
    and (pao2.measuredat - sc.admittedat) < 1000*60*60*24
    AND (f.value LIKE '%art.%' OR f.value like '%ART.%' OR f.value IS NULL)  -- source is arterial or undefined (assume arterial)
    and pao2.value > 0
)

, pao2_bg as (
    select admissionid, hr, avg(pao2) as pao2
    from pao2_bg_1
    group by admissionid, hr
)

, paco2_bg_1 as (
    SELECT 
        paco2.admissionid,
        case when paco2.itemid = 21213 then paco2.value * 7.50061683 -- Conversion: kPa to mmHg
             else paco2.value
        end as paco2,
        f.value AS specimen,
        CASE 
            WHEN paco2.registeredby NOT LIKE '%Systeem%' THEN TRUE
            ELSE FALSE
        END AS manual_entry,
        paco2.measuredat,
        CAST(ceil((paco2.measuredat - sc.admittedat)/(1000*60*60)) AS INT64) as hr
    FROM `physionet-data.amsterdamdb.numericitems` paco2
    INNER JOIN older_study_cohort_ams sc ON
        paco2.admissionid = sc.admissionid
    LEFT JOIN `physionet-data.amsterdamdb.freetextitems` f ON
        paco2.admissionid = f.admissionid AND
        paco2.measuredat = f.measuredat AND
        f.itemid = 11646 --Afname (bloed): source of specimen
    WHERE 
        paco2.itemid IN (
            6846, --PCO2
            9990, --pCO2 (bloed)
            21213 --PCO2 (bloed) - kPa
        ) 
    --measurements within 24 hours of ICU stay (use 30 minutes before admission to allow for time differences):
    AND (paco2.measuredat - sc.admittedat) >= -1000*60*60*6
    AND (paco2.measuredat - sc.admittedat) < 1000*60*60*24 
    AND (f.value LIKE '%art.%' OR f.value like '%ART.%' OR f.value IS NULL)  -- source is arterial or undefined (assume arterial)
    and paco2.value > 0
)

, paco2_bg as (
    select admissionid, hr, avg(paco2) as paco2
    from paco2_bg_1
    group by admissionid, hr
)

, pao2_fio2_paco2_info_0 as (
  select distinct admissionid, hr
  from (
    select distinct admissionid, hr
    from fio2_bg
    union all
    select distinct admissionid, hr
    from pao2_bg
    union all
    select distinct admissionid, hr
    from paco2_bg    
  )
)

, pao2_fio2_paco2_info_1 as (
	select ct.admissionid, ct.hr
	, round(f.fio2) as fio2, round(po.pao2,1) as pao2 
	, round(pc.paco2,1) as paco2 
	, case 
	when fio2 is not null and pao2 is not null and ct.hr = vt.hr
	then round(100*pao2/fio2,0) else null end as pao2fio2ratio_vent
	, case 
	when fio2 is not null and pao2 is not null and ct.hr != vt.hr
	then round(100*pao2/fio2,0) else null end as pao2fio2ratio_novent
	from pao2_fio2_paco2_info_0 ct 
	left join fio2_bg f 
	on ct.admissionid = f.admissionid 
	and ct.hr = f.hr 
	left join pao2_bg po 
	on ct.admissionid = po.admissionid 
	and ct.hr = po.hr
	left join paco2_bg pc 
	on ct.admissionid = pc.admissionid 
	and ct.hr = pc.hr
  left join vent_cohort_time vt 
  on vt.admissionid = ct.admissionid
)

, pao2_fio2_paco2_info as (
    select pf.admissionid, min(pao2fio2ratio_novent) as pao2fio2ratio_novent
    , min(pao2fio2ratio_vent) as pao2fio2ratio_vent
    , min(pao2) as pao2_min
    , max(paco2) as paco2_max
    , max(fio2) as fio2_max
    from pao2_fio2_paco2_info_1 pf 
    group by pf.admissionid
)

, lab_info_part10 as 
(
  select a.admissionid
  , CAST(ceil((n.measuredat - a.admittedat)/(1000*60*60)) AS INT64) as hr
  , CASE
        WHEN itemid IN (
            9559, --Anion-Gap (bloed)
            8492 --AnGap
        ) THEN 'ANION GAP'
        WHEN itemid IN (
            9937, --Alb.Chem (bloed)
            6801 --Albumine chemisch
        ) THEN 'ALBUMIN'
        -- WHEN itemid = 51144 THEN 'BANDS'
        WHEN itemid IN (
            9992, --Act.HCO3 (bloed)
            6810 --HCO3
        ) THEN 'BICARBONATE'
        WHEN itemid IN (
            9945, --Bilirubine (bloed)
            6813 --Bili Totaal
        ) THEN 'BILIRUBIN'
        WHEN itemid IN (
            9941, --Kreatinine (bloed) µmol/l
            6836, --Kreatinine µmol/l (erroneously documented as µmol)
            14216 --KREAT enzym. (bloed) µmol/l
        ) THEN 'CREATININE'
        WHEN itemid in (
          6819, -- Chloor
          9930 -- Chloor (bloed)
        ) THEN 'CHLORIDE' 
        WHEN itemid IN (
            9947, --Glucose (bloed)
            6833, --Glucose Bloed 
            9557 --Glucose Astrup
        ) THEN 'GLUCOSE'
        WHEN itemid IN (
            11423, --Ht (bloed)
            11545, --Ht(v.Bgs) (bloed)
            6777 --Hematocriet
        ) THEN 'HEMATOCRIT'
        WHEN itemid IN (
            10053, --Lactaat (bloed)
            6837, --Laktaat
            9580 --Laktaat Astrup
        ) THEN 'LACTATE'
        WHEN itemid IN (
            9964, --Thrombo's (bloed)
            6797, --Thrombocyten
            10409, --Thrombo's citr. bloed (bloed)
            14252 --Thrombo CD61 (bloed)
        ) THEN 'PLATELET'
        WHEN itemid IN (
            9927, --Kalium (bloed) mmol/l
            9556, --Kalium Astrup mmol/l
            6835, --Kalium mmol/l
            10285 --K (onv.ISE) (bloed) mmol/l
        ) THEN 'POTASSIUM'
        WHEN itemid in (
          11944, -- APTT  (bloed)
          17982 -- APTT (bloed)
        ) THEN 'PTT'
        WHEN itemid in (
          11893, -- Prothrombinetijd (bloed)
          11894 -- Prothrombinetijd  (bloed)
        ) THEN 'INR'
        WHEN itemid = 6789 -- Protrombinetijd
        THEN 'PT'
        WHEN itemid IN (
            9924, --Natrium (bloed)
            6840, --Natrium
            9555, --Natrium Astrup
            10284 --Na (onv.ISE) (bloed)
        ) THEN 'SODIUM'
        WHEN itemid IN (
            9943, --Ureum (bloed)
            6850 --Ureum
        ) THEN 'BUN'
        WHEN itemid IN (
            9965, --Leuco's (bloed) 10^9/l
            6779 --Leucocyten 10^9/l
        ) THEN 'WBC'
        WHEN itemid IN (
            11978, --ALAT (bloed)
            6800 --ALAT
        ) THEN 'ALT'
        WHEN itemid IN (
            11990, --ASAT (bloed)
            6806 --ASAT
        ) THEN 'AST'
        WHEN itemid IN (
            11984, --Alk.Fosf. (bloed)
            6803 --Alk. Fosfatase
        ) THEN 'Alkaline_phosphatase'
        WHEN itemid IN (
            9952, --Magnesium (bloed)
            6839 --Magnesium
        ) THEN 'Magnesium'
        WHEN itemid IN (
            8115, -- Troponine  ng/ml
            10407 -- TroponineT (bloed) ug/ml
        ) THEN 'Troponin'
    WHEN itemid = 14249 -- NT-proBNP (bloed)
    THEN 'BNP'
        WHEN itemid IN (
            6776, -- Fibrinogeen  ng/ml
            9989, -- Fibrinogeen (bloed)
            10175 -- Fibrinogeen  (bloed)
        ) THEN 'Fibrinogen'
    WHEN itemid in (
      --6780, -- Lymfocyten
      6781, -- Lymfocyten
      11846, -- Lymfocyten % (bloed)
      14258 -- Lymfocyten (bloed)
    ) THEN 'Lymphocytes'
    WHEN itemid IN (
            6786, -- Neutrofielen
            -- 14254, -- separately process
            11856 -- Neutro's % app (bloed)
    ) THEN 'Neutrophils'
    WHEN itemid IN (9933, 6817) THEN 'Calcium'
    WHEN itemid in (9994, 6807) THEN 'baseexcess'
    WHEN itemid in (12310, 6848) THEN 'pH'
    else null end as label 
    , -- add in some sanity checks on the values
        -- the where clause below requires all valuenum to be > 0, so these are only upper limit checks
        CASE
        WHEN itemid in (9559, 8492) and value > 10000 THEN null -- mmol/l 'ANION GAP'
        WHEN itemid in (9937, 6801) and value > 10*10 THEN null -- g/L 'ALBUMIN'
        WHEN itemid in (9992, 6810) and value > 10000 THEN null -- mmol/l 'BICARBONATE'
        WHEN itemid in (9945, 6813) and value > 150*17.1 THEN null -- µmol/l 'BILIRUBIN'
        WHEN itemid in (6819, 9930) and value > 10000 THEN null -- mmol/l 'CHLORIDE'                    
        WHEN itemid in (9941, 6836, 14216) and value >  150*88.4 THEN null -- µmol/l 'CREATININE'
        WHEN itemid in (9947, 6833, 9557) and value > 0.0555*10000 THEN null -- mg/dL 'GLUCOSE'
        WHEN itemid in (11423, 11545, 6777) and value >   100 THEN null -- % 'HEMATOCRIT'       
        WHEN itemid in (10053, 6837, 9580) and value >    50 THEN null -- mmol/L 'LACTATE'
        WHEN itemid in (9964, 6797, 10409, 14252) and value > 10000 THEN null -- 10^9/l 'PLATELET'
        WHEN itemid in (9927, 9556, 6835, 10285) and value >  30 THEN null -- mmol/L 'POTASSIUM'
        WHEN itemid in (11944, 17982) and value >   150 THEN null -- sec 'PTT'
        WHEN itemid in (11893, 11894) and value >    50 THEN null -- 'INR'
        WHEN itemid = 6789 and value >   150 THEN null -- sec 'PT'
        WHEN itemid in (9924, 6840, 9555, 10284) and value >   200 THEN null -- mEq/L == mmol/L 'SODIUM'
        WHEN itemid in (9943, 6850) and value >  300/2.8 THEN null -- 1mmol/L == 2.8mg/dL 'BUN'
        WHEN itemid in (9965, 6779) and value >  1000 THEN null -- 10^9/l = K/uL 'WBC'
        WHEN itemid in (11978, 6800) and value >  20000 THEN null -- ie/L = iu/L 'ALT'
        WHEN itemid in (11990, 6806) and value >  20000 THEN null -- ie/L = iu/L 'AST'
        WHEN itemid in (11984, 6803) and value >  10000 THEN null -- ie/L = iu/L 'Alkaline_phosphatase'
        WHEN itemid in (9952, 6839) and value >  50/2.4305 THEN null -- 1 mmol/L = 2.4305 mg/dL 'Magnesium'
        WHEN itemid in (8115, 10407) and value >  100 THEN null -- 1 ug/L = 1 ng/ml = ng/ml 'Troponin'
        WHEN itemid = 14249 and value >  100000 THEN null -- ng/l = pg/ml 'BNP'
        WHEN itemid in (6776, 9989, 10175) and value >  5000/100 THEN null -- 1 g/L = 100 mg/dL 'Fibrinogen'
        WHEN itemid in (6781, 11846, 14258) and value >  100 THEN null -- 'Lymphocytes'
        WHEN itemid in (6786, 11856) and (value >  100 or value < 0) THEN null -- % 'Neutrophils'
        When itemid in (9933, 6817) and (value > 10000*0.25) THEN null -- 1 mg/dL = 0.25 mmol/L 
      ELSE value end as valuenum
  from `physionet-data.amsterdamdb.admissions` a
  left join `physionet-data.amsterdamdb.numericitems` n
  on a.admissionid = n.admissionid
  and n.measuredat >= a.admittedat - 6*1000*60*60
  and n.measuredat < a.admittedat + 24*1000*60*60
  and n.itemid in 
  (
       -- label_, label_Englis, units
            9559, --Anion-Gap (bloed)  |  ANION GAP
            8492, --AnGap  |  ANION GAP
            --  mmol/l  =  mEq/L
            9937, --Alb.Chem (bloed)  |  ALBUMIN 
            6801,--Albumine chemisch  |  ALBUMIN
            -- 10 g/l = 1 g/dl
            9992, --Act.HCO3 (bloed)  |  BICARBONATE
            6810, --HCO3  |  BICARBONATE  |  BICARBONATE
            -- mmol/l = mEq/L 
            9945, --Bilirubine (bloed)  |  BILIRUBIN
            6813, --Bili Totaal  |  BILIRUBIN
            -- 17.1 µmol/l = mg/dL
            9941, --Kreatinine (bloed) µmol/l  |  CREATININE
            6836, --Kreatinine µmol/l (erroneously documented as µmol)  |  CREATININE
            14216, --KREAT enzym. (bloed) µmol/l  |  CREATININE
            -- 88.4 µmol/l = mg/dL
            6819, -- Chloor |  CHLORIDE 
            9930, -- Chloor (bloed) |  CHLORIDE 
            -- mmol/l = mEq/L
            9947, --Glucose (bloed) |  GLUCOSE 
            6833, --Glucose Bloed  |  GLUCOSE
            9557, --Glucose Astrup |  GLUCOSE  
            -- 0.0555 mmol/L = 1 mg/dL
            11423, --Ht (bloed)  |  HEMATOCRIT   
            11545, --Ht(v.Bgs) (bloed)  |  HEMATOCRIT 
            6777, --Hematocriet  |  HEMATOCRIT  
            10053, --Lactaat (bloed)  |  LACTATE 
            6837, --Laktaat  |  LACTATE 
            9580, --Laktaat Astrup  |  LACTATE 
            9964, --Thrombo's (bloed)  |  PLATELET  
            6797, --Thrombocyten  |  PLATELET 
            10409, --Thrombo's citr. bloed (bloed)  |  PLATELET 
            14252, --Thrombo CD61 (bloed)  |  PLATELET  
            -- 10^9/l = 1k/uL
            9927, --Kalium (bloed) mmol/l  |  POTASSIUM 
            9556, --Kalium Astrup mmol/l  |  POTASSIUM 
            6835, --Kalium mmol/l  |  POTASSIUM  
            10285, --K (onv.ISE) (bloed) mmol/l  |  POTASSIUM 
            -- 1 mmoI/L = 1 mEq/L
            11944, -- APTT  (bloed)  |  PTT  
            17982, -- APTT (bloed)  |  PTT  
            11893, -- Prothrombinetijd (bloed)  |  INR  
            11894, -- Prothrombinetijd  (bloed)  |  INR  
            6789,
            -- Protrombinetijd  |  PT    
            9924, --Natrium (bloed)  |  SODIUM   
            6840, --Natrium  |  SODIUM    
            9555, --Natrium Astrup  |  SODIUM   
            10284, --Na (onv.ISE) (bloed)  |  SODIUM    
            9943, --Ureum (bloed)  |  BUN  
            6850, --Ureum  |  BUN  
            -- 1mmol/L = 2.8mg/dL 
            9965, --Leuco's (bloed) 10^9/l  |  WBC 
            6779, --Leucocyten 10^9/l  |  WBC 
            -- 10^9/l = K/uL
            11978, --ALAT (bloed)  |  ALT 
            6800, --ALAT  |  ALT 
            11990, --ASAT (bloed)  |  AST 
            6806,--ASAT  |  AST 
            11984, --Alk.Fosf. (bloed)  |  Alkaline phosphatase  
            6803, --Alk. Fosfatase  |  Alkaline phosphatase  
            9952, --Magnesium (bloed)  |  Magnesium  |  
            6839, --Magnesium  |  Magnesium  |  
            -- 1 mmol/L = 2.4305 mg/dL
            8115, -- Troponine  ng/ml  |  Troponin  | 
            10407, -- TroponineT (bloed) ug/ml  |  Troponin  | 
            14249, 
            -- NT-proBNP (bloed)  |  BNP  | 
            6776, -- Fibrinogeen  ng/ml  |  Fibrinogen  | 
            9989, -- Fibrinogeen (bloed)  |  Fibrinogen  | 
            10175, -- Fibrinogeen  (bloed)  |  Fibrinogen  | 
            -- 1 g/L = 100 mg/dL
            --6780, -- Lymfocyten  |  Lymphocytes  |  
            6781, -- Lymfocyten  |  Lymphocytes  |  
            11846, -- Lymfocyten % (bloed)  |  Lymphocytes  |  
            14258,-- Lymfocyten (bloed)  |  Lymphocytes  |  
            6786, -- Neutrofielen  |  Neutrophils  | 
            14254, -- Neutro's % app (bloed) 
            11856, -- Neutro's % app (bloed)  |  Neutrophils  | 
            9933, -- Calcium
            6817, -- Calcium
            9994, 6807, -- baseexcess
            12310, 6848 -- ph
  )
  and value is not null and value > 0
  where islabresult is true
  and a.admissionid in (select admissionid from older_study_cohort_ams)
)

, lab_info_part11 as (
    SELECT
      pvt_l.admissionid, pvt_l.hr
      , round(avg(CASE WHEN label = 'ANION GAP' THEN valuenum ELSE null END),1) as aniongap
      , round(avg(CASE WHEN label = 'ALBUMIN' THEN valuenum/10 ELSE null END),1) as albumin
      , round(avg(CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE null END),1) as bicarbonate
      , round(avg(CASE WHEN label = 'BILIRUBIN' THEN valuenum/17.1 ELSE null END),1) as bilirubin
      , round(avg(CASE WHEN label = 'CREATININE' THEN valuenum/88.4 ELSE null END),2) as creatinine
      , round(avg(CASE WHEN label = 'CHLORIDE' THEN valuenum ELSE null END),1) as chloride
      , round(avg(CASE WHEN label = 'GLUCOSE' THEN valuenum/0.0555 ELSE null END),1) as glucose
      , round(avg(CASE WHEN label = 'HEMATOCRIT' then
                    case when valuenum < 1 then 100*valuenum else valuenum end 
                 ELSE null END),1) as hematocrit
      , round(avg(CASE WHEN label = 'LACTATE' THEN valuenum ELSE null END),1) as lactate
      , round(avg(CASE WHEN label = 'PLATELET' THEN valuenum ELSE null END),1) as platelet
      , round(avg(CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE null END),1) as potassium
      , round(avg(CASE WHEN label = 'PTT' THEN valuenum ELSE null END),1) as ptt
      , round(avg(CASE WHEN label = 'INR' THEN valuenum ELSE null END),1) as inr
      , round(avg(CASE WHEN label = 'PT' THEN valuenum ELSE null END),1) as pt
      , round(avg(CASE WHEN label = 'SODIUM' THEN valuenum ELSE null END),1) as sodium
      , round(avg(CASE WHEN label = 'BUN' THEN 2.8*valuenum ELSE null end),1) as bun
      , round(avg(CASE WHEN label = 'WBC' THEN valuenum ELSE null end),1) as wbc
      , round(avg(CASE WHEN label = 'ALT' THEN valuenum ELSE null END),1) as alt
      , round(avg(CASE WHEN label = 'AST' THEN valuenum ELSE null END),1) as ast
      , round(avg(CASE WHEN label = 'Alkaline_phosphatase' THEN valuenum ELSE null end),1) as alkaline_phosphatase
      , round(avg(CASE WHEN label = 'Magnesium' THEN 2.4305*valuenum ELSE null end),1) as magnesium
      , round(avg(CASE WHEN label = 'Troponin' THEN valuenum ELSE null END),1) as troponin
      , round(avg(CASE WHEN label = 'BNP' THEN valuenum ELSE null end),1) as bnp
      , round(avg(CASE WHEN label = 'Fibrinogen' THEN 100*valuenum ELSE null end),1) as fibrinogen
      , round(avg(CASE WHEN label = 'Lymphocytes' THEN valuenum ELSE null end),1) as lymphocytes
      , round(avg(CASE WHEN label = 'Neutrophils' and valuenum < 1 THEN 100*valuenum 
                 when label = 'Neutrophils' and valuenum > 1 THEN valuenum 
                 ELSE null end),1) as neutrophils
      , round(avg(CASE WHEN label = 'Calcium' THEN valuenum/0.25 ELSE null END),1) as calcium
      , round(avg(CASE WHEN label = 'baseexcess' THEN valuenum ELSE null END), 2) as baseexcess
      , round(avg(CASE WHEN label = 'pH' THEN valuenum ELSE null END),2) as ph
    from lab_info_part10 pvt_l  
    group by pvt_l.admissionid, pvt_l.hr
)

, lab_info_part1 as (
  select admissionid, max(creatinine) as creatinine_max
  , max(bilirubin) as bilirubin_max, min(platelet) as platelet_min
  , max(bun) as bun_max, max(wbc) as wbc_max, max(glucose) as glucose_max
  , max(hematocrit) as hematocrit_max, max(potassium) as potassium_max
  , max(sodium) as sodium_max, min(bicarbonate) as bicarbonate_min
  , max(bicarbonate) as bicarbonate_max, min(albumin) as albumin_min
  , max(alt) as alt_max, max(ast) as ast_max, max(alkaline_phosphatase) as alp_max
  , max(pt) as pt_max, max(ptt) as ptt_max, min(inr) as inr_min
  , min(chloride) as chloride_min
  , max(lactate) as lactate_max, min(baseexcess) as baseexcess_min
  , max(troponin) as troponin_max, max(lymphocytes) as lymphocytes_max
  , min(lymphocytes) as lymphocytes_min, min(neutrophils) as neutrophils_min
  , max(magnesium) as magnesium_max, min(fibrinogen) as fibrinogen_min
  , max(bnp) as bnp_max, max(aniongap) as aniongap_max
  from lab_info_part11
  group by admissionid
)

select sc.admissionid, pao2fio2ratio_novent, pao2fio2ratio_vent
, pao2_min, paco2_max, fio2_max
, creatinine_max, bilirubin_max, platelet_min
, bun_max, wbc_max, glucose_max, hematocrit_max, potassium_max
, sodium_max, bicarbonate_min, bicarbonate_max, albumin_min
, alt_max, ast_max, alp_max, pt_max, ptt_max, inr_min
, chloride_min, lactate_max, baseexcess_min
, troponin_max, lymphocytes_max, lymphocytes_min, neutrophils_min
, magnesium_max, fibrinogen_min, bnp_max, aniongap_max
from older_study_cohort_ams sc 
left join pao2_fio2_paco2_info pfp 
on sc.admissionid = pfp.admissionid
left join lab_info_part1 lab 
on sc.admissionid = lab.admissionid
order by sc.admissionid;



drop table if exists `db_name.older_study_ams_vt_tr`;
create table `db_name.older_study_ams_vt_tr` as

with older_study_cohort_ams as (
    select *
    from `db_name.older_study_ams_initial`
)

, resp_rate_info_0 AS (
    SELECT
        n.admissionid,
        n.itemid,
        n.item,
        n.value,
        CASE
            WHEN NOT n.registeredby IS NULL THEN TRUE
            ELSE FALSE
        END as validated,
        CAST(ceil((n.measuredat - a.admittedat)/(1000*60*60)) AS INT64) as hr,
        ROW_NUMBER() OVER(
            PARTITION BY n.admissionid, n.measuredat
            ORDER BY
                CASE itemid
                    WHEN 8873 THEN 1 --Ventilator measurements
                    WHEN 12266 THEN 2 --Ventilator measurements
                    ELSE 3  --Patient monitor measurements
                END
            ) AS priority
    FROM `physionet-data.amsterdamdb.numericitems` n
    LEFT JOIN `physionet-data.amsterdamdb.admissions` a ON
        n.admissionid = a.admissionid
    WHERE itemid IN (
        --Evita Parameters
        8873, --Ademfrequentie Evita: measurement by Evita ventilator, most accurate
        --7726, --Ademfrequentie Spontaan: measurement by Evita ventilator, spontaneous breaths/min, distiction not needed for 'actual' respiratory rate
        --9654, --Ademfrequentie Spontaan(2): measurement by 2nd simultaneously used Evita ventilator (very uncommon), spontaneous breaths/min, distiction not needed for 'actual' respiratory rate

        --Servo-i/Servo-u Parameters
        --12283, --Adem Frequentie (Set): setting on Servo-i/Servo-U ventilator, not needed for 'actual' respiratory rate
        --12322, --Adem Frequentie (Set) (2): setting on 2nd simultaneously used Servo-i/Servo-U ventilator (uncommon), not needed for 'actual' respiratory rate
        12266, --Ademfreq.: measurement by Servo-i/Servo-U ventilator, most accurate
        --12348, --Ademfreq.(2): measurement by 2nd simultaneously used Servo-i/Servo-U ventilator (uncommon), no additional information
        --12577 --Ademfreq. Spontaan nieuw --from Servo-i/Servo-U ventilator, spontaneous breaths/min, distiction not needed for 'actual' respiratory rate

        --Patient monitor
        8874 --Ademfrequentie Monitor: measurement by patient monitor using ECG-impedance, less accurate
        )
    and n.admissionid in (select admissionid from older_study_cohort_ams)
    AND (n.measuredat - a.admittedat) <= 1000*60*60*24 --measurements within 24 hours
    and (n.measuredat - a.admittedat) >= 0
)

, resp_rate_info as (
  SELECT admissionid, hr, value 
  FROM resp_rate_info_0
  WHERE value > 0 and value < 70
)

, gcs_components_info_0 AS (
    SELECT
        eyes.admissionid,
        --eyes.itemid,
        --eyes.item,
        --eyes.value,
        --eyes.valueid,
        CASE eyes.itemid
            WHEN 6732 THEN 5 - eyes.valueid     --Actief openen van de ogen
            WHEN 13077 THEN eyes.valueid        --A_Eye
            WHEN 14470 THEN eyes.valueid - 4    --RA_Eye
            WHEN 16628 THEN eyes.valueid - 4    --MCA_Eye
            WHEN 19635 THEN eyes.valueid - 4    --E_EMV_NICE_24uur
            WHEN 19638 THEN eyes.valueid - 8    --E_EMV_NICE_Opname
        END AS eyes_score,
        --motor.value,
        --motor.valueid,
        CASE motor.itemid
            WHEN 6734 THEN 7 - motor.valueid    --Beste motore reactie van de armen
            WHEN 13072 THEN motor.valueid       --A_Motoriek
            WHEN 14476 THEN motor.valueid - 6   --RA_Motoriek
            WHEN 16634 THEN motor.valueid - 6   --MCA_Motoriek
            WHEN 19636 THEN motor.valueid - 6   --M_EMV_NICE_24uur
            WHEN 19639 THEN motor.valueid - 12  --M_EMV_NICE_Opname
        END AS motor_score,
        --verbal.value,
        --verbal.valueid,
        CASE verbal.itemid
            WHEN 6735 THEN 6 - verbal.valueid   --Beste verbale reactie
            WHEN 13066 THEN verbal.valueid      --A_Verbal
            WHEN 14482 THEN verbal.valueid - 5  --RA_Verbal
            WHEN 16640 THEN verbal.valueid - 5  --MCA_Verbal
            WHEN 19637 THEN verbal.valueid - 9 --V_EMV_NICE_24uur
            WHEN 19640 THEN verbal.valueid - 15 --V_EMV_NICE_Opname
        END AS verbal_score,
        eyes.registeredby,
        CAST(ceil((eyes.measuredat - a.admittedat)/(1000*60*60)) AS INT64) as hr
    FROM `physionet-data.amsterdamdb.listitems` eyes
    LEFT JOIN `physionet-data.amsterdamdb.admissions` a ON
        eyes.admissionid = a.admissionid
    LEFT JOIN `physionet-data.amsterdamdb.listitems` motor ON
        eyes.admissionid = motor.admissionid AND
        eyes.measuredat = motor.measuredat AND
        motor.itemid IN (
            6734, --Beste motore reactie van de armen
            13072, --A_Motoriek
            14476, --RA_Motoriek
            16634, --MCA_Motoriek
            19636, --M_EMV_NICE_24uur
            19639 --M_EMV_NICE_Opname
        )
    LEFT JOIN `physionet-data.amsterdamdb.listitems` verbal ON
        eyes.admissionid = verbal.admissionid AND
        eyes.measuredat = verbal.measuredat AND
        verbal.itemid IN (
            6735, --Beste verbale reactie
            13066, --A_Verbal
            14482, --RA_Verbal
            16640, --MCA_Verbal
            19637, --V_EMV_NICE_24uur
            19640 --V_EMV_NICE_Opname
        )
    WHERE eyes.itemid IN (
        6732, --Actief openen van de ogen
        13077, --A_Eye
        14470, --RA_Eye
        16628, --MCA_Eye
        19635, --E_EMV_NICE_24uur
        19638 --E_EMV_NICE_Opname
        )
    -- measurements within 24 hours of ICU stay:
    AND (eyes.measuredat - a.admittedat) <= 1000*60*60*24 AND (eyes.measuredat - a.admittedat) >= -1000*60*60*6
    and eyes.admissionid in (select admissionid from older_study_cohort_ams)
)

, gcs_components_info_1 AS (
    SELECT *,
        eyes_score + motor_score + (
            CASE
                WHEN verbal_score < 1 THEN 1
                ELSE verbal_score
            END
        ) AS gcs_score
    FROM gcs_components_info_0
)

, gcs_components_info AS (
    SELECT admissionid, hr, eyes_score as gcseyes
    , motor_score as gcsmotor, verbal_score as gcsverbal
    , gcs_score 
    FROM gcs_components_info_1
)

, vital_uo_info as (
	SELECT vt.admissionid
	, CAST(ceil((vt.measuredat - sc.admittedat)/(1000*60*60)) AS INT64) as hr 
	, case 
	when itemid = 6640 and value > 0 and value < 300 then 1 -- HeartRate
	when itemid in (
    6641, -- ABP systolisch
    6678,	-- Niet invasieve bloeddruk systolisch
    8841	-- ABP systolisch II
  ) and value > 0 and value < 400 then 2 -- SysBP
	when itemid in (
    6642, --ABP gemiddeld
    6679, --Niet invasieve bloeddruk gemiddeld
    8843 --ABP gemiddeld II
  ) and value > 0 and value < 300 then 3 -- MeanBP
	-- when itemid = 8874 and value > 0 and value < 70 then 4 -- RespRate | separate part
	when itemid in (8658, --Temp Bloed
	                8659, --Temperatuur Perifeer 2
	                8662, --Temperatuur Perifeer 1
	                13058, --Temp Rectaal
	                13059, --Temp Lies
	                13060, --Temp Axillair
	                13061, --Temp Oraal
	                13062, --Temp Oor
	                13063, --Temp Huid
	                13952, --Temp Blaas
	                16110 --Temp Oesophagus
	                ) 
        and ROUND((((value * 1.8) - 32 )/1.8), 1) > 10 
        and ROUND((((value * 1.8) - 32 )/1.8), 1) < 50  then 5 -- TempC
        -- ROUND((((value * 1.8) - 32 )/1.8), 1) -- unit
	when itemid = 6709 and value > 0 and value <= 100 then 6 -- SpO2
	when itemid in (
        8794, --UrineCAD
        8796, --UrineSupraPubis
        8798, --UrineSpontaan
        8800, --UrineIncontinentie
        8803, --UrineUP
        10743, --Nefrodrain li Uit
        10745, --Nefrodrain re Uit
        19921, --UrineSplint Li
        19922 --UrineSplint Re
  ) and value >= 0 then 7 -- urineoutput
    --when itemid in (6643, 6680, 8842) and value > 0 then 8 -- dbp
	else null end as VitalID
	, case when itemid in (8658, 8659, 8662, 13058, 13059, 13060, 
        13061, 13062, 13063, 13952, 16110
        ) then ROUND((((value * 1.8) - 32 )/1.8), 1) 
    else value end as valuenum
	from `physionet-data.amsterdamdb.numericitems` vt 
	inner join older_study_cohort_ams sc
	on sc.admissionid = vt.admissionid
	and vt.measuredat >= sc.admittedat 
	and (vt.measuredat - sc.admittedat) < 1000*60*60*24 --measurements within 24 hours
	where vt.itemid in (
    6640, 6641, 6678,	8841,
    6642, 6679, 8843, 
    8658, 8659, 8662, 13058, 
    13059, 13060, 13061, 13062, 
    13063, 13952, 16110, 6709, 
    8794, 8796, 8798, 8800, 8803, 
    10743, 10745, 19921, 19922
	)
  and vt.value > 0
  union all 
  select admissionid, hr, 4 as VitalID, value as valuenum
  from resp_rate_info
)

, vital_gcs_uo_info as (
  select sc.admissionid, g.gcs_min
  , round(her.heart_rate_mean, 1) as heart_rate_mean
  , round(s.sbp_mean, 1) as sbp_mean
  , round(m.mbp_mean, 1) as mbp_mean
  , round(r.resp_rate_mean, 1) as resp_rate_mean
  , round(t.temperature_mean, 1) as temperature_mean
  , round(sp.spo2_min, 1) as spo2_min
  , uo.urineoutput
  from older_study_cohort_ams sc 
  left join (
      select admissionid, min(gcs_score) as gcs_min
      from gcs_components_info
      group by admissionid
  ) g 
  on sc.admissionid = g.admissionid
  left join (
      select admissionid, avg(valuenum) as heart_rate_mean
      from vital_uo_info
      where VitalID =1
      group by admissionid
  ) her
  on sc.admissionid = her.admissionid
  left join (
      select admissionid, avg(valuenum) as sbp_mean
      from vital_uo_info
      where VitalID = 2
      group by admissionid
  ) s 
  on sc.admissionid = s.admissionid
  left join (
      select admissionid, avg(valuenum) as mbp_mean
      from vital_uo_info
      where VitalID = 3
      group by admissionid
  ) m 
  on sc.admissionid = m.admissionid
  left join (
      select admissionid, avg(valuenum) as resp_rate_mean
      from vital_uo_info
      where VitalID = 4
      group by admissionid
  ) r 
  on sc.admissionid = r.admissionid
  left join (
      select admissionid, avg(valuenum) as temperature_mean
      from vital_uo_info
      where VitalID = 5
      group by admissionid      
  ) t 
  on sc.admissionid = t.admissionid
  left join (
      select admissionid, min(valuenum) as spo2_min
      from vital_uo_info
      where VitalID = 6
      group by admissionid 
  ) sp 
  on sc.admissionid = sp.admissionid
  left join (
      select admissionid, sum(valuenum) as urineoutput
      from vital_uo_info
      where VitalID = 7
      group by admissionid 
  ) uo 
  on sc.admissionid = uo.admissionid
)

, dosing_info_0 AS (
    SELECT  
        dg.admissionid, 
        itemid,
        item,
        (start - ad.admittedat)/(1000*60*60) AS start_hr, 
        (stop - ad.admittedat)/(1000*60*60) AS stop_hr, 
        duration,
        rate,
        rateunit,
        dose,
        doseunit,
        doseunitid,
        doserateperkg,
        doserateunitid,
        doserateunit,
        CASE
            WHEN weightgroup LIKE '59' THEN 55
            WHEN weightgroup LIKE '60' THEN 65
            WHEN weightgroup LIKE '70' THEN 75
            WHEN weightgroup LIKE '80' THEN 85
            WHEN weightgroup LIKE '90' THEN 95
            WHEN weightgroup LIKE '100' THEN 105
            WHEN weightgroup LIKE '110' THEN 115
            ELSE 80 --mean weight for all years
        END as patientweight
    FROM `physionet-data.amsterdamdb.drugitems` dg
    LEFT JOIN `physionet-data.amsterdamdb.admissions` ad
    ON dg.admissionid = ad.admissionid
    WHERE ordercategoryid = 65 -- continuous i.v. perfusor
    AND itemid IN (
            7179, -- Dopamine (Inotropin)
            7178, -- Dobutamine (Dobutrex)
            6818, -- Adrenaline (Epinefrine)
            7229  -- Noradrenaline (Norepinefrine)
        )
    AND rate > 0 and dose > 0
    and dg.admissionid in (select admissionid from older_study_cohort_ams)
)

, dosing_info_1 as (
    SELECT 
        admissionid,
        itemid,
        case when itemid = 7179 then 'dopamine'
        when itemid = 7178 then 'dobutamine'
        when itemid = 6818 then 'epinephrine'
        when itemid = 7229 then 'norepinephrine'
        else null end as label,
        item,
        duration,
        rate,
        rateunit,
        start_hr,
        stop_hr,
        rate as gamma
    FROM dosing_info_0
    WHERE
        -- medication given within 24 hours of ICU stay:
        start_hr < 24 AND stop_hr >= 0
)

, dosing_info as (
    select admissionid, label, start_hr, stop_hr, gamma
    from dosing_info_1
)

, vent_info_0 as (
    select po.admissionid
    , CAST(ceil((po.start - sc.admittedat)/(1000*60*60)) AS INT64) as start_hr
    , CAST(ceil((po.stop - sc.admittedat)/(1000*60*60)) AS INT64) as stop_hr
    from `physionet-data.amsterdamdb.processitems` po
    inner join older_study_cohort_ams sc
    on sc.admissionid = po.admissionid 
    where itemid = 12634  
)

, vent_info_1 as (
    select admissionid
    , case when start_hr <= 0 and stop_hr > 0 then 1
    when start_hr > 0 and start_hr < 24 then 1
    else 0 end as vent
    from vent_info_0
)

, vent_info as (
    select admissionid, max(vent) as vent
    from vent_info_1
    group by admissionid
)

select sc.admissionid, gcs_min, heart_rate_mean
, sbp_mean, mbp_mean, resp_rate_mean
, temperature_mean, spo2_min, urineoutput
, case when sc.admissionid = dop.admissionid then 1 else 0 end as dopamine
, case when sc.admissionid = dob.admissionid then 1 else 0 end as dobutamine
, case when sc.admissionid = ep.admissionid then 1 else 0 end as epinephrine
, case when sc.admissionid = nop.admissionid then 1 else 0 end as norepinephrine
, case when vent = 1 then 1 else 0 end as vent
from older_study_cohort_ams sc 
left join vital_gcs_uo_info vg 
on sc.admissionid = vg.admissionid
left join (
    select distinct admissionid
    from dosing_info
    where gamma > 0
    and label = 'dopamine'
) dop 
on sc.admissionid = dop.admissionid
left join (
    select distinct admissionid
    from dosing_info
    where gamma > 0
    and label = 'dobutamine'
) dob 
on sc.admissionid = dob.admissionid
left join (
    select distinct admissionid
    from dosing_info
    where gamma > 0
    and label = 'epinephrine'
) ep 
on sc.admissionid = ep.admissionid
left join (
    select distinct admissionid
    from dosing_info
    where gamma > 0
    and label = 'norepinephrine'
) nop 
on sc.admissionid = nop.admissionid
left join vent_info vti 
on sc.admissionid = vti.admissionid
order by sc.admissionid;




drop table if exists `db_name.older_study_ams_older`;
create table `db_name.older_study_ams_older` as

with older_study_cohort_ams as (
    select *
    from `db_name.older_study_ams_initial`
)

, activity_info as (
    select sc.admissionid
    , max(case when sc.admissionid = stand.admissionid then 1 else 0 end) as stand
    , max(case when sc.admissionid = sit.admissionid then 1 else 0 end) as sit
    , max(case when sc.admissionid = bed.admissionid then 1 else 0 end) as bed
    , max(case when sc.admissionid = eva.admissionid then 1 else 0 end) as activity_eva_flag
    from older_study_cohort_ams sc
    left join (
        select distinct admissionid
        from `physionet-data.amsterdamdb.listitems`
        where admissionid in (select admissionid from older_study_cohort_ams)
        and itemid = 19233 -- Fysio BBS stand
        and value is not null
        and measuredat >= - 24*1000*60*60 -- at most consider one day before ICU admitted
        and measuredat < 24*1000*60*60
        union all
        select distinct admissionid
        from `physionet-data.amsterdamdb.listitems`
        where admissionid in (select admissionid from older_study_cohort_ams)
        and itemid = 15149 -- Mobilisatie keuze
        and value in ('kan staan met hulp 2 personen', 'kan staan met hulp 1 persoon')
        and measuredat >= - 24*1000*60*60 -- at most consider two days before ICU admitted
        and measuredat < 24*1000*60*60
        union all
        select distinct admissionid
        from `physionet-data.amsterdamdb.listitems`
        where admissionid in (select admissionid from older_study_cohort_ams)
        and itemid = 19237 -- Fysio BBS zit-stand
        and value is not null
        and measuredat >= - 24*1000*60*60 -- at most consider two days before ICU admitted
        and measuredat < 24*1000*60*60
    ) stand 
    on sc.admissionid = stand.admissionid 
    left join (
        select distinct admissionid
        from `physionet-data.amsterdamdb.listitems`
        where admissionid in (select admissionid from older_study_cohort_ams)
        and itemid = 19235 -- Fysio BBS zit
        and value is not null
        and measuredat >= - 24*1000*60*60 -- at most consider one day before ICU admitted
        and measuredat < 24*1000*60*60
        union all 
        select distinct admissionid
        from `physionet-data.amsterdamdb.listitems`
        where admissionid in (select admissionid from older_study_cohort_ams)
        and itemid = 9233 -- Houding patiënt
        and value in ('Halfzittend', 'Stoel', 'Zittend')
        and measuredat >= - 24*1000*60*60 -- at most consider one day before ICU admitted
        and measuredat < 24*1000*60*60 
        union all
        select distinct admissionid
        from `physionet-data.amsterdamdb.listitems`
        where admissionid in (select admissionid from older_study_cohort_ams)
        and itemid = 19596 -- Mobiliseren bed/stoel
        and value in ('in stoel')
        and measuredat >= - 24*1000*60*60 -- at most consider one day before ICU admitted
        and measuredat < 24*1000*60*60                       
    ) sit
    on sc.admissionid = sit.admissionid
    left join (
        select distinct admissionid
        from `physionet-data.amsterdamdb.listitems`
        where admissionid in (select admissionid from older_study_cohort_ams)
        and itemid in (
            19249, 19250, 19251, 19252, 19253,
            19254, 19255, 19256, 19257, 19258,
            19259, 19260
        ) -- MRC Sum Score Links
        and value is not null
        and measuredat >= - 24*1000*60*60 -- at most consider one day before ICU admitted
        and measuredat < 24*1000*60*60
        union all
        select distinct admissionid
        from `physionet-data.amsterdamdb.listitems`
        where admissionid in (select admissionid from older_study_cohort_ams)
        and itemid in (
            19249, 19250, 19251, 19252, 19253,
            19254, 19255, 19256, 19257, 19258,
            19259, 19260
        ) -- MRC Sum Score Links
        or (
            itemid = 9233 -- Houding patiënt
            and value not in ('Halfzittend', 'Stoel', 'Zittend')
        )
        and measuredat >= - 24*1000*60*60 -- at most consider one day before ICU admitted
        and measuredat < 24*1000*60*60
        union all
        select distinct admissionid
        from `physionet-data.amsterdamdb.listitems`
        where admissionid in (select admissionid from older_study_cohort_ams)
        and itemid = 19596 -- Mobiliseren bed/stoel
        and value in ('in bed')
        and measuredat >= - 24*1000*60*60 -- at most consider one day before ICU admitted
        and measuredat < 24*1000*60*60                  
    ) bed
    on sc.admissionid = bed.admissionid
    left join (
        select distinct admissionid
        from `physionet-data.amsterdamdb.listitems`
        where admissionid in (select admissionid from older_study_cohort_ams)
        and itemid in (
            19233 -- Fysio BBS stand
            , 15149 -- Mobilisatie keuze
            , 19237 -- Fysio BBS zit-stand
            , 19235 -- Fysio BBS zit
            , 9233 -- Houding patiënt
            , 19596 -- Mobiliseren bed/stoel
            , 19249, 19250, 19251, 19252, 19253 -- MRC Sum Score Links
            , 19254, 19255, 19256, 19257, 19258
            , 19259, 19260 -- MRC Sum Score Links
        ) 
        and value is not null
        and measuredat >= - 24*1000*60*60 -- at most consider one day before ICU admitted
        and measuredat < 24*1000*60*60        
    ) eva
    on sc.admissionid = eva.admissionid
    group by sc.admissionid    
)

, score_info as (
    select sc.admissionid
    , sf.sofa, 1 / (1 + exp(- (-3.3890 + 0.2439*(sf.sofa) ))) as sofa_prob
    , sp.sapsii as saps,  sp.sapsii_prob as saps_prob
    , ap.apsiii, ap.apsiii_prob
    , oa.oasis, oa.oasis_PROB as oasis_prob
    , null as cci_score
    from older_study_cohort_ams sc
    left join `db_name.sofafirstday_ams` sf
    on sc.admissionid = sf.admissionid
    left join `db_name.sapsii_firstday_ams` sp
    on sc.admissionid = sp.admissionid
    left join `db_name.oasis_firstday_ams` oa
    on sc.admissionid = oa.admissionid
    left join `db_name.apsiiifirstday_ams` ap
    on sc.admissionid = ap.admissionid        
)

select sc.admissionid
, stand, sit, bed, activity_eva_flag
, apsiii, apsiii_prob, oasis, oasis_prob, saps, saps_prob, sofa, sofa_prob
, 0 as cci_score
, 0 as code_status
, 0 as code_status_eva_flag
, 0 as pre_icu_los_day

from older_study_cohort_ams sc 
left join activity_info ai 
on sc.admissionid = ai.admissionid
left join score_info si 
on sc.admissionid = si.admissionid
order by sc.admissionid;



drop table if exists `db_name.older_study_ams`;
create table `db_name.older_study_ams` as

with older_study_cohort_ams as (
    select *
    from `db_name.older_study_ams_initial`
)

, exclude_id_vt_info as (
  select admissionid
  from (
      select admissionid
      , case when heart_rate_mean > 0 then 0 else 1 end as hr_flag
      , case when mbp_mean > 0 then 0 else 1 end as mbp_flag
      , case when resp_rate_mean > 0 then 0 else 1 end as rr_flag
      , case when gcs_min > 0 then 0 else 1 end as gcs_flag
      , case when temperature_mean > 0 then 0 else 1 end as t_flag
      , case when spo2_min > 0 then 0 else 1 end as spo2_flag
      , case when sbp_mean > 0 then 0 else 1 end as sbp_flag    
      from `db_name.older_study_ams_vt_tr`
  )
  where (hr_flag + mbp_flag + rr_flag + gcs_flag + t_flag + spo2_flag + sbp_flag) > 0
)

, older_study_ams_0 as (
    select sc.admissionid as id, sc.patientid
    , first_careunit, los_icu_day
    , death_hosp, deathtime_icu_hour
    , admission_type, admissionyeargroup as anchor_year_group
    , gender, agegroup, heightgroup, weightgroup
    --, admittedat, dischargedat
    , electivesurgery
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

    , gcs_min --, gcseyes, gcsmotor, gcsverbal
    , urineoutput, spo2_min
    , heart_rate_mean, mbp_mean, sbp_mean
    , resp_rate_mean, temperature_mean

    , case when norepinephrine = 1 then 1 else 0 end as norepinephrine
    , case when epinephrine = 1 then 1 else 0 end as epinephrine
    , case when dopamine = 1 then 1 else 0 end as dopamine
    , case when dobutamine = 1 then 1 else 0 end as dobutamine

    , case when stand = 1 then 1 else 0 end as activity_stand
    , case when sit = 1 and (stand is null or stand = 0) then 1 else 0 end as activity_sit
    , case when bed = 1 and (stand is null or stand = 0) and (sit is null or sit = 0) then 1 else 0 end as activity_bed
    , case when activity_eva_flag = 1 then 1 else 0 end as activity_eva_flag

    , apsiii, apsiii_prob, oasis, oasis_prob, saps, saps_prob, sofa, sofa_prob, cci_score
    , code_status, code_status_eva_flag, pre_icu_los_day

    from older_study_cohort_ams sc
    left join `db_name.older_study_ams_lab` lab
    on sc.admissionid = lab.admissionid
    left join `db_name.older_study_ams_vt_tr` vt_tr
    on sc.admissionid = vt_tr.admissionid
    left join `db_name.older_study_ams_older` ol
    on sc.admissionid = ol.admissionid
    where sc.admissionid not in (select admissionid from exclude_id_vt_info)
)

, older_study_ams_1 as (
  select *, ROW_NUMBER() OVER (ORDER BY id) as id_new
  from older_study_ams_0
)

-- since we set mimic as start, eicu next. So ams start from 100000 to keep the unique id
select (id_new + 100000) as id
, id as admissionid
, patientid
, first_careunit, los_icu_day
, death_hosp, deathtime_icu_hour
, admission_type, anchor_year_group
, gender, agegroup, heightgroup, weightgroup
, electivesurgery
, vent
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
, gcs_min --, gcseyes, gcsmotor, gcsverbal
, urineoutput, spo2_min
, heart_rate_mean, mbp_mean, sbp_mean
, resp_rate_mean, temperature_mean
, norepinephrine
, epinephrine
, dopamine
, dobutamine
, activity_stand
, activity_sit
, activity_bed
, activity_eva_flag
, apsiii, apsiii_prob, oasis, oasis_prob, saps, saps_prob, sofa, sofa_prob, cci_score
, code_status, code_status_eva_flag, pre_icu_los_day
from older_study_ams_1
order by id_new;



-- drop no need tables
drop table if exists `db_name.older_study_ams_initial`;
drop table if exists `db_name.older_study_ams_lab`;
drop table if exists `db_name.older_study_ams_vt_tr`;
drop table if exists `db_name.older_study_ams_older`;