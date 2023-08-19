-- ------------------------------------------------------------------
-- Title: Simplified Acute Physiology Score II (SAPS II)
-- This query extracts the simplified acute physiology score II.
-- This score is a measure of patient severity of illness.
-- The score is calculated on the first day of each ICU patients' stay.
-- ------------------------------------------------------------------

-- Reference for SAPS II:
--    Le Gall, Jean-Roger, Stanley Lemeshow, and Fabienne Saulnier.
--    "A new simplified acute physiology score (SAPS II) based on a European/North American multicenter study."
--    JAMA 270, no. 24 (1993): 2957-2963.

-- Variables used in SAPS II:
--  Age, GCS
--  VITALS: Heart rate, systolic blood pressure, temperature
--  FLAGS: ventilation/cpap
--  IO: urine output
--  LABS: PaO2/FiO2 ratio, blood urea nitrogen, WBC, potassium, sodium, HCO3

drop table if exists `db_name.sapsii_firstday_ams`;
create table `db_name.sapsii_firstday_ams` as

with demographic_total_ams_0 as (
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
  --, ceil((dischargedat - admittedat)/3600000) as icu_los_hours
  -- , round((- admittedat)/(24*3600000),3) as los_icu_admit_day
  , round((dischargedat - admittedat)/(24*3600000),3) as los_icu_day
  , case when dateofdeath > admittedat and abs(round((dischargedat - dateofdeath)/(24*3600000),3)) <= 2 
  then ceil(dateofdeath/3600000) else null end as deathtime_icu_hour
  , case when dateofdeath > admittedat and abs(round((dischargedat - dateofdeath)/(24*3600000),3)) <= 2 then 1 else 0 end as death_hosp
  , case 
  when gender = 'Man' then 'M'
  when gender = 'Vrouw' then 'F'
  else null end as gender
  , agegroup, weightgroup, lengthgroup as heightgroup
  FROM `physionet-data.amsterdamdb.admissions`
  --where admissioncount = 1 -- first_hospital + icu 
)

, demographic_total_ams as (
  select sc.admissionid, first_careunit
  , gender, agegroup, death_hosp
  , admittedat, admissionyeargroup, dischargedat
  -- , ethnicity, icu_admit_day -- unknown
  -- , bmi -- need to calculate separately
  , admission_type, heightgroup, weightgroup
  , los_icu_day, deathtime_icu_hour --, icu_los_hours
  from demographic_total_ams_0 sc 
  order by sc.admissionid
)

, cpap_0 as (
  SELECT 
      l.admissionid
      , CAST(ceil((l.measuredat - sc.admittedat)/(1000*60*60)) AS INT64) as hr
  FROM `physionet-data.amsterdamdb.listitems` l
  inner join demographic_total_ams sc 
  on l.admissionid = sc.admissionid 
  WHERE 
      (
          itemid = 9534  --Type beademing Evita 1
          AND valueid IN (
              8, --CPAP continuous positive airway pressure
              9 --CPAP_ASB
          )
      )
      OR (
          itemid = 6685 --Type Beademing Evita 4
          AND valueid IN (
              10, --CPAP
              11 --CPAP/ASB
          )
      )
      OR (
          itemid = 8189 --Toedieningsweg O2
          AND valueid = 16 --CPAP
      ) 
      OR (
          itemid IN (
              12290, --Ventilatie Mode (Set) - Servo-I and Servo-U ventilators
              12347 --Ventilatie Mode (Set) (2) Servo-I and Servo-U ventilators
          )
          AND valueid IN (
               13 --PS/CPAP (trig)
          )
      )
      OR (
        itemid = 12376 --Mode (Bipap Vision)
        AND valueid IN (
            1 --CPAP
        )
      )
  group by l.admissionid, CAST(ceil((l.measuredat - sc.admittedat)/(1000*60*60)) AS INT64)    
)

, cpap as (
    select distinct admissionid, hr
    from cpap_0
)

-- extract a flag for surgical service
-- this combined with "elective" defines elective/non-elective surgery
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
  and admissionid in (select admissionid from demographic_total_ams)
)

, surgflag_info as (
    select sc.admissionid, max(case when surgical = 1 then 1 else 0 end) as surgical
    from demographic_total_ams sc 
    left join surgflag_info_0 si 
    on sc.admissionid = si.admissionid
    group by sc.admissionid
)

-- icd-9 diagnostic codes are our best source for comorbidity information
-- unfortunately, they are technically a-causal
-- however, this shouldn't matter too much for the SAPS II comorbidities
-- aids | lymphoma | metastatic cancer

-- the chronic health points-calculation has currently not been implemented because they have not been documented consistently
-- we set to zero to easily calculate
, comorb as
(
  select admissionid, 0 as aids, 0 as hem, 0 as mets
  from demographic_total_ams
)

, vent_info as (
    select po.admissionid
    , CAST(ceil((po.start - sc.admittedat)/(1000*60*60)) AS INT64) as start_hr
    , CAST(ceil((po.stop - sc.admittedat)/(1000*60*60)) AS INT64) as stop_hr
    from `physionet-data.amsterdamdb.processitems` po
    inner join demographic_total_ams sc
    on sc.admissionid = po.admissionid 
    where itemid = 12634 
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
    INNER JOIN demographic_total_ams sc 
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
    INNER JOIN demographic_total_ams sc ON
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

, pao2_fio2_info_0 as (
  select distinct admissionid, hr
  from (
    select distinct admissionid, hr
    from fio2_bg
    union all
    select distinct admissionid, hr
    from pao2_bg    
  )
)

, pao2_fio2_info as (
	select ct.admissionid, ct.hr
	, round(f.fio2) as fio2, round(po.pao2,1) as pao2
    , case 
	when fio2 is not null and pao2 is not null then round(100*pao2/fio2,0) else null end as pao2fio2ratio  
	from pao2_fio2_info_0 ct 
	left join fio2_bg f 
	on ct.admissionid = f.admissionid 
	and ct.hr = f.hr 
	left join pao2_bg po 
	on ct.admissionid = po.admissionid 
	and ct.hr = po.hr
)

, pafi1 as
(
  -- join blood gas to ventilation durations to determine if patient was vent
  -- also join to cpap table for the same purpose
  select bg.admissionid, bg.hr
  , pao2fio2
  , case when vd.admissionid is not null then 1 else 0 end as vent
  , case when cp.admissionid is not null then 1 else 0 end as cpap
  from (
      select admissionid, hr
      , round(pao2fio2ratio,0) as pao2fio2
      from pao2_fio2_info
      where fio2 is not null
      and pao2 is not null
  ) bg
  left join vent_info vd
    on bg.admissionid = vd.admissionid
    and bg.hr >= vd.start_hr
    and bg.hr <= vd.stop_hr
  left join cpap cp
    on bg.admissionid = cp.admissionid
    and bg.hr = cp.hr
)

, pafi2 as
(
  -- get the minimum PaO2/FiO2 ratio *only for ventilated/cpap patients*
  select admissionid
  , min(pao2fio2) as pao2fio2_vent_min
  from pafi1
  where vent = 1 or cpap = 1
  group by admissionid
)

, vitals_uo_firstday as (
	SELECT vt.admissionid
	, CAST(ceil((vt.measuredat - sc.admittedat)/(1000*60*60)) AS INT64) as hr 
	, case 
	when itemid = 6640 and value > 0 and value < 300 then 1 -- HeartRate
	when itemid in (
    6641, -- ABP systolisch
    6678,	-- Niet invasieve bloeddruk systolisch
    8841	-- ABP systolisch II
  ) and value > 0 and value < 400 then 2 -- SysBP
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
	                ) and ROUND((((value * 1.8) - 32 )/1.8), 1) > 10 and ROUND((((value * 1.8) - 32 )/1.8), 1) < 50  then 3 -- TempC
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
    ) and value >= 0 then 4 -- urineoutput
	else null end as VitalID
	, case when itemid in (8658, 8659, 8662, 13058, 13059, 13060, 
        13061, 13062, 13063, 13952, 16110) then ROUND((((value * 1.8) - 32 )/1.8), 1)
    else value end as valuenum
	from `physionet-data.amsterdamdb.numericitems` vt 
	inner join demographic_total_ams sc
	on sc.admissionid = vt.admissionid
	and vt.measuredat >= sc.admittedat 
	and (vt.measuredat - sc.admittedat) < 1000*60*60*24 --measurements within 24 hours
	where vt.itemid in (
    6640, 6641, 6678, 8841,
    8658, 8659, 8662, 13058, 
    13059, 13060, 13061, 13062, 
    13063, 13952, 16110,
    8794, 8796, 8798, 8800, 8803, 
    10743, 10745, 19921, 19922
	)
  and vt.value > 0
)

, vitalsfirstday as (
    select sc.admissionid, heartrate_max, heartrate_min
	, sysbp_max, sysbp_min
	, tempc_max, tempc_min
    from demographic_total_ams sc 
    left join (
        select admissionid, max(valuenum) as heartrate_max
        , min(valuenum) as heartrate_min
        from vitals_uo_firstday
        where VitalID = 1
        group by admissionid
    ) her 
    on sc.admissionid = her.admissionid
    left join (
        select admissionid, max(valuenum) as sysbp_max
        , min(valuenum) as sysbp_min
        from vitals_uo_firstday
        where VitalID = 2
        group by admissionid
    ) s 
    on sc.admissionid = s.admissionid
    left join (
        select admissionid, max(valuenum) as tempc_max
        , min(valuenum) as tempc_min
        from vitals_uo_firstday
        where VitalID = 3
        group by admissionid
    ) t 
    on sc.admissionid = t.admissionid        
)

, uofirstday as (
	select sc.admissionid, urineoutput
	from demographic_total_ams sc 
    left join (
        select admissionid, sum(valuenum) as urineoutput
        from vitals_uo_firstday
        where VitalID = 4
        group by admissionid
    ) her 
    on sc.admissionid = her.admissionid
)

, lab_info_0 as (
  select a.admissionid
  , CAST(ceil((n.measuredat - a.admittedat)/(1000*60*60)) AS INT64) as hr
  , CASE
        WHEN itemid IN (
            9992, --Act.HCO3 (bloed)
            6810 --HCO3
        ) THEN 'BICARBONATE'
        WHEN itemid IN (
            9945, --Bilirubine (bloed)
            6813 --Bili Totaal
        ) THEN 'BILIRUBIN'
        WHEN itemid IN (
            9927, --Kalium (bloed) mmol/l
            9556, --Kalium Astrup mmol/l
            6835, --Kalium mmol/l
            10285 --K (onv.ISE) (bloed) mmol/l
        ) THEN 'POTASSIUM'
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
        else null end as label 
    , -- add in some sanity checks on the values
        -- the where clause below requires all valuenum to be > 0, so these are only upper limit checks
        CASE
        WHEN itemid in (9992, 6810) and value > 10000 THEN null -- mmol/l 'BICARBONATE'
        WHEN itemid in (9945, 6813) and value > 150*17.1 THEN null -- µmol/l 'BILIRUBIN'
        WHEN itemid in (9927, 9556, 6835, 10285) and value >  30 THEN null -- mmol/L 'POTASSIUM'
        WHEN itemid in (9924, 6840, 9555, 10284) and value >   200 THEN null -- mEq/L == mmol/L 'SODIUM'
        WHEN itemid in (9943, 6850) and value >  300/2.8 THEN null -- 1mmol/L == 2.8mg/dL 'BUN'
        WHEN itemid in (9965, 6779) and value >  1000 THEN null -- 10^9/l = K/uL 'WBC'
      ELSE value end as valuenum
  from `physionet-data.amsterdamdb.admissions` a
  left join `physionet-data.amsterdamdb.numericitems` n
  on a.admissionid = n.admissionid
  and n.measuredat >= a.admittedat - 6*1000*60*60
  and n.measuredat < a.admittedat + 24*1000*60*60
  and n.itemid in 
  (
            9992, --Act.HCO3 (bloed)  |  BICARBONATE
            6810, --HCO3  |  BICARBONATE  |  BICARBONATE
            -- mmol/l = mEq/L 
            9945, --Bilirubine (bloed)  |  BILIRUBIN
            6813, --Bili Totaal  |  BILIRUBIN
            9927, --Kalium (bloed) mmol/l  |  POTASSIUM 
            9556, --Kalium Astrup mmol/l  |  POTASSIUM 
            6835, --Kalium mmol/l  |  POTASSIUM  
            10285, --K (onv.ISE) (bloed) mmol/l  |  POTASSIUM    
            9924, --Natrium (bloed)  |  SODIUM   
            6840, --Natrium  |  SODIUM    
            9555, --Natrium Astrup  |  SODIUM   
            10284, --Na (onv.ISE) (bloed)  |  SODIUM    
            9943, --Ureum (bloed)  |  BUN  
            6850, --Ureum  |  BUN  
            -- 1mmol/L = 2.8mg/dL 
            9965, --Leuco's (bloed) 10^9/l  |  WBC 
            6779 --Leucocyten 10^9/l  |  WBC 
  )
  and value is not null and value > 0
  where islabresult is true
  and a.admissionid in (select admissionid from demographic_total_ams)
)

, lab_info_1 as (
    SELECT
      pvt_l.admissionid, pvt_l.hr
      , round(avg(CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE null END),1) as bicarbonate
      , round(avg(CASE WHEN label = 'BILIRUBIN' THEN valuenum/17.1 ELSE null END),1) as bilirubin
      , round(avg(CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE null END),1) as potassium
      , round(avg(CASE WHEN label = 'SODIUM' THEN valuenum ELSE null END),1) as sodium
      , round(avg(CASE WHEN label = 'BUN' THEN 2.8*valuenum ELSE null end),1) as bun
      , round(avg(CASE WHEN label = 'WBC' THEN valuenum ELSE null end),1) as wbc
    from lab_info_0 pvt_l  
    group by pvt_l.admissionid, pvt_l.hr
)

, lab_info as (
  select admissionid
  , min(bun) as bun_min, max(bun) as bun_max
  , min(wbc) as wbc_min, max(wbc) as wbc_max
  , min(potassium) as potassium_min, max(potassium) as potassium_max
  , min(sodium) as sodium_min, max(sodium) as sodium_max
  , min(bicarbonate) as bicarbonate_min, max(bicarbonate) as bicarbonate_max
  , min(bilirubin) as bilirubin_min, max(bilirubin) as bilirubin_max
  from lab_info_1
  group by admissionid    
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
    and eyes.admissionid in (select admissionid from demographic_total_ams)
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

, gcs_info as (
    select admissionid, min(gcs_score) as mingcs
    from gcs_components_info
    group by admissionid
)

, cohort as
(
  select ie.admissionid
        -- the casts ensure the result is numeric.. we could equally extract EPOCH from the interval
        -- however this code works in Oracle and Postgres
        , ie.agegroup

        , vital.heartrate_max
        , vital.heartrate_min
        , vital.sysbp_max
        , vital.sysbp_min
        , vital.tempc_max
        , vital.tempc_min

        -- this value is non-null iff the patient is on vent/cpap
        , pf.pao2fio2_vent_min

        , uo.urineoutput

        , labs.bun_min
        , labs.bun_max
        , labs.wbc_min
        , labs.wbc_max
        , labs.potassium_min
        , labs.potassium_max
        , labs.sodium_min
        , labs.sodium_max
        , labs.bicarbonate_min
        , labs.bicarbonate_max
        , labs.bilirubin_min
        , labs.bilirubin_max

        , gcs.mingcs

        , comorb.aids
        , comorb.hem
        , comorb.mets

        , case
            when ie.admission_type in ('planned') and sf.surgical = 1
              then 'ScheduledSurgical'
            when ie.admission_type in ('unplanned') and sf.surgical = 1
              then 'UnscheduledSurgical'
            else 'Medical'
          end as admissiontype


  FROM demographic_total_ams ie
  -- join to above views
  left join pafi2 pf
    on ie.admissionid = pf.admissionid
  left join surgflag_info sf
    on ie.admissionid = sf.admissionid
  left join comorb
    on ie.admissionid = comorb.admissionid
  left join gcs_info gcs
    on ie.admissionid = gcs.admissionid
  left join vitalsfirstday vital
    on ie.admissionid = vital.admissionid
  left join uofirstday uo
    on ie.admissionid = uo.admissionid
  left join lab_info labs
    on ie.admissionid = labs.admissionid
)

, scorecomp as
(
    select
    cohort.*
    -- Below code calculates the component scores needed for SAPS
    , case when agegroup = '18-39' then 0
        when agegroup in ('50-59', '40-49') then 7
        when agegroup = '60-69' then 12
        when agegroup = '70-79' then (15+16)/2
        when agegroup = '80+' then 18
        when agegroup is null then null
        end as age_score

    , case
        when heartrate_max is null then null
        when heartrate_min <   40 then 11
        when heartrate_max >= 160 then 7
        when heartrate_max >= 120 then 4
        when heartrate_min  <  70 then 2
        when  heartrate_max >= 70 and heartrate_max < 120
            and heartrate_min >= 70 and heartrate_min < 120
        then 0
        end as hr_score

    , case
        when  sysbp_min is null then null
        when  sysbp_min <   70 then 13
        when  sysbp_min <  100 then 5
        when  sysbp_max >= 200 then 2
        when  sysbp_max >= 100 and sysbp_max < 200
            and sysbp_min >= 100 and sysbp_min < 200
            then 0
        end as sysbp_score

    , case
        when tempc_max is null then null
        when tempc_min <  39.0 then 0
        when tempc_max >= 39.0 then 3
        end as temp_score

    , case
        when pao2fio2_vent_min is null then null
        when pao2fio2_vent_min <  100 then 11
        when pao2fio2_vent_min <  200 then 9
        when pao2fio2_vent_min >= 200 then 6
        end as pao2fio2_score

    , case
        when urineoutput is null then null
        when urineoutput <   500.0 then 11
        when urineoutput <  1000.0 then 4
        when urineoutput >= 1000.0 then 0
        end as uo_score

    , case
        when bun_max is null then null
        when bun_max <  28.0 then 0
        when bun_max <  84.0 then 6
        when bun_max >= 84.0 then 10
        end as bun_score

    , case
        when wbc_max is null then null
        when wbc_min <   1.0 then 12
        when wbc_max >= 20.0 then 3
        when wbc_max >=  1.0 and wbc_max < 20.0
        and wbc_min >=  1.0 and wbc_min < 20.0
            then 0
        end as wbc_score

    , case
        when potassium_max is null then null
        when potassium_min <  3.0 then 3
        when potassium_max >= 5.0 then 3
        when potassium_max >= 3.0 and potassium_max < 5.0
        and potassium_min >= 3.0 and potassium_min < 5.0
            then 0
        end as potassium_score

    , case
        when sodium_max is null then null
        when sodium_min  < 125 then 5
        when sodium_max >= 145 then 1
        when sodium_max >= 125 and sodium_max < 145
        and sodium_min >= 125 and sodium_min < 145
            then 0
        end as sodium_score

    , case
        when bicarbonate_max is null then null
        when bicarbonate_min <  15.0 then 5
        when bicarbonate_min <  20.0 then 3
        when bicarbonate_max >= 20.0
        and bicarbonate_min >= 20.0
            then 0
        end as bicarbonate_score

    , case
        when bilirubin_max is null then null
        when bilirubin_max  < 4.0 then 0
        when bilirubin_max  < 6.0 then 4
        when bilirubin_max >= 6.0 then 9
        end as bilirubin_score

    , case
        when mingcs is null then null
            when mingcs <  3 then null -- erroneous value/on trach
            when mingcs <  6 then 26
            when mingcs <  9 then 13
            when mingcs < 11 then 7
            when mingcs < 14 then 5
            when mingcs >= 14
            and mingcs <= 15
            then 0
            end as gcs_score

        , case
            when aids = 1 then 17
            when hem  = 1 then 10
            when mets = 1 then 9
            else 0
        end as comorbidity_score

        , case
            when admissiontype = 'ScheduledSurgical' then 0
            when admissiontype = 'Medical' then 6
            when admissiontype = 'UnscheduledSurgical' then 8
            else null
        end as admissiontype_score

    from cohort
)

-- Calculate SAPS II here so we can use it in the probability calculation below
, score as
(
  select s.*
  -- coalesce statements impute normal score of zero if data element is missing
  , coalesce(age_score,0)
  + coalesce(hr_score,0)
  + coalesce(sysbp_score,0)
  + coalesce(temp_score,0)
  + coalesce(pao2fio2_score,0)
  + coalesce(uo_score,0)
  + coalesce(bun_score,0)
  + coalesce(wbc_score,0)
  + coalesce(potassium_score,0)
  + coalesce(sodium_score,0)
  + coalesce(bicarbonate_score,0)
  + coalesce(bilirubin_score,0)
  + coalesce(gcs_score,0)
  + coalesce(comorbidity_score,0)
  + coalesce(admissiontype_score,0)
    as sapsii
  from scorecomp s
)

select ie.admissionid
, sapsii
, 1 / (1 + exp(- (-7.7631 + 0.0737*(sapsii) + 0.9971*(ln(sapsii + 1))) )) as sapsii_prob
, age_score
, hr_score
, sysbp_score
, temp_score
, pao2fio2_score
, uo_score
, bun_score
, wbc_score
, potassium_score
, sodium_score
, bicarbonate_score
, bilirubin_score
, gcs_score
, comorbidity_score
, admissiontype_score
FROM demographic_total_ams ie
left join score s
  on ie.admissionid = s.admissionid
order by ie.admissionid;