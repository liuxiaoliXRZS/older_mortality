-- ------------------------------------------------------------------
-- Title: Oxford Acute Severity of Illness Score (OASIS)
-- This query extracts the Oxford acute severity of illness score based on eICU Collabrate Database.
-- This score is a measure of severity of illness for patients in the ICU.
-- The score is calculated the score of each ICU patients' stay on admission.
-- ------------------------------------------------------------------

-- Reference for OASIS:
--    Johnson, Alistair EW, Andrew A. Kramer, and Gari D. Clifford.
--    "A new severity of illness scale using a subset of acute physiology and chronic health evaluation data elements shows comparable predictive accuracy*."
--    Critical care medicine 41, no. 7 (2013): 1711-1718.

-- Reference for oasis.sql of MIMIC-III by Alistair

-- Variables used in OASIS:
--  Heart rate, GCS, MAP, Temperature, Respiratory rate, Ventilation status
--  Urine output
--  Elective surgery
--  Pre-ICU in-hospital length of stay
--  Age

-- By Xiaoli Liu
-- 2022.01.06

drop table if exists `db_name.oasis_firstday_ams`;
create table `db_name.oasis_firstday_ams` as

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
  , admittedat, admissionyeargroup, dischargedat, dateofdeath
  -- , ethnicity, icu_admit_day -- unknown
  -- , bmi -- need to calculate separately
  , admission_type, heightgroup, weightgroup
  , los_icu_day, deathtime_icu_hour --, icu_los_hours
  from demographic_total_ams_0 sc 
  order by sc.admissionid
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

, vent_info_0 as (
    select po.admissionid
    , CAST(ceil((po.start - sc.admittedat)/(1000*60*60)) AS INT64) as start_hr
    , CAST(ceil((po.stop - sc.admittedat)/(1000*60*60)) AS INT64) as stop_hr
    from `physionet-data.amsterdamdb.processitems` po
    inner join demographic_total_ams sc
    on sc.admissionid = po.admissionid 
    where itemid = 12634 
)

, vent_info as (
    select sc.admissionid
    , max(
        case when sc.admissionid = vt.admissionid and start_hr <= 0 and stop_hr > 0 then 1
        when sc.admissionid = vt.admissionid and start_hr >= 0 and start_hr < 24 then 1
        else 0 end
    ) as mechvent
    from demographic_total_ams sc 
    left join vent_info_0 vt 
    on sc.admissionid = vt.admissionid
    group by sc.admissionid 
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
    and n.admissionid in (select admissionid from demographic_total_ams)
    AND (n.measuredat - a.admittedat) <= 1000*60*60*24 --measurements within 24 hours
    and (n.measuredat - a.admittedat) >= 0
)

, resp_rate_info as (
  SELECT admissionid, hr, value 
  FROM resp_rate_info_0
  WHERE value > 0 and value < 70
)

, vitals_uo_firstday as (
	SELECT vt.admissionid
	, CAST(ceil((vt.measuredat - sc.admittedat)/(1000*60*60)) AS INT64) as hr 
	, case 
	when itemid = 6640 and value > 0 and value < 300 then 1 -- HeartRate
	when itemid in (
    6642, --ABP gemiddeld
    6679, --Niet invasieve bloeddruk gemiddeld
    8843 --ABP gemiddeld II
  ) and value > 0 and value < 300 then 2 -- MeanBP	
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
    6640, 6642, 6679, 8843,
    8658, 8659, 8662, 13058, 
    13059, 13060, 13061, 13062, 
    13063, 13952, 16110,
    8794, 8796, 8798, 8800, 8803, 
    10743, 10745, 19921, 19922
	)
  and vt.value > 0
  union all
  select admissionid, hr, 5 as VitalID, value as valuenum
  from resp_rate_info
)

, vitalsfirstday as (
    select sc.admissionid, heartrate_max, heartrate_min
	, meanbp_max, meanbp_min
    , resprate_max, resprate_min
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
        select admissionid, max(valuenum) as meanbp_max
        , min(valuenum) as meanbp_min
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
    left join (
        select admissionid, max(valuenum) as resprate_max
        , min(valuenum) as resprate_min
        from vitals_uo_firstday
        where VitalID = 5
        group by admissionid
    ) rr 
    on sc.admissionid = rr.admissionid            
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

, cohort as
(
    select ie.admissionid
        , ie.admittedat as intime
        , ie.dischargedat as outtime
        , ie.dateofdeath as deathtime
        , null as preiculos -- can't acquire
        , agegroup
        , gcs.mingcs
        , vital.heartrate_max
        , vital.heartrate_min
        , vital.meanbp_max
        , vital.meanbp_min
        , vital.resprate_max
        , vital.resprate_min
        , vital.tempc_max
        , vital.tempc_min
        , vent.mechvent
        , uo.urineoutput

        , case
            when ie.admission_type in ('planned') and sf.surgical = 1
                then 1
            when ie.admission_type in ('unplanned') and sf.surgical = 1
                then 0
            else null
            end as electivesurgery

        -- age group
        , 'adult' as icustay_age_group

        -- mortality flags (can't acquire more detail)
        , death_hosp as icustay_expire_flag
        , death_hosp as hospital_expire_flag

    FROM demographic_total_ams ie
    left join surgflag_info sf
    on ie.admissionid = sf.admissionid
    -- join to custom tables to get more data....
    left join gcs_info gcs
    on ie.admissionid = gcs.admissionid
    left join vitalsfirstday vital
    on ie.admissionid = vital.admissionid
    left join uofirstday uo
    on ie.admissionid = uo.admissionid
    left join vent_info vent
    on ie.admissionid = vent.admissionid
)

, scorecomp as
(
    select co.admissionid
    , co.icustay_age_group
    , co.icustay_expire_flag
    , co.hospital_expire_flag

    -- Below code calculates the component scores needed for oasis
    , case when preiculos is null then null
        when preiculos < 10.2 then 5
        when preiculos < 297 then 3
        when preiculos < 1440 then 0
        when preiculos < 18708 then 1
        else 2 end as preiculos_score

    ,  case when agegroup is null then null
        when agegroup = '18-39' then (0+3)/2
        when agegroup = '40-49' then 3
        when agegroup = '50-59' then (3+6)/2
        when agegroup = '60-69' then 6
        when agegroup = '70-79' then (6+9)/2
        when agegroup = '80+' then (9+7)/2
        else 0 end as age_score

    ,  case when mingcs is null then null
        when mingcs <= 7 then 10
        when mingcs < 14 then 4
        when mingcs = 14 then 3
        else 0 end as gcs_score
    ,  case when heartrate_max is null then null
        when heartrate_max > 125 then 6
        when heartrate_min < 33 then 4
        when heartrate_max >= 107 and heartrate_max <= 125 then 3
        when heartrate_max >= 89 and heartrate_max <= 106 then 1
        else 0 end as heartrate_score
    ,  case when meanbp_min is null then null
        when meanbp_min < 20.65 then 4
        when meanbp_min < 51 then 3
        when meanbp_max > 143.44 then 3
        when meanbp_min >= 51 and meanbp_min < 61.33 then 2
        else 0 end as meanbp_score
    ,  case when resprate_min is null then null
        when resprate_min <   6 then 10
        when resprate_max >  44 then  9
        when resprate_max >  30 then  6
        when resprate_max >  22 then  1
        when resprate_min <  13 then 1 else 0
        end as resprate_score
    ,  case when tempc_max is null then null
        when tempc_max > 39.88 then 6
        when tempc_min >= 33.22 and tempc_min <= 35.93 then 4
        when tempc_max >= 33.22 and tempc_max <= 35.93 then 4
        when tempc_min < 33.22 then 3
        when tempc_min > 35.93 and tempc_min <= 36.39 then 2
        when tempc_max >= 36.89 and tempc_max <= 39.88 then 2
        else 0 end as temp_score
    ,  case when urineoutput is null then null
        when urineoutput < 671.09 then 10
        when urineoutput > 6896.80 then 8
        when urineoutput >= 671.09
        and urineoutput <= 1426.99 then 5
        when urineoutput >= 1427.00
        and urineoutput <= 2544.14 then 1
        else 0 end as urineoutput_score
    ,  case when mechvent is null then null
        when mechvent = 1 then 9
        else 0 end as mechvent_score
    ,  case when electivesurgery is null then null
        when electivesurgery = 1 then 0
        else 6 end as electivesurgery_score


    -- The below code gives the component associated with each score
    -- This is not needed to calculate oasis, but provided for user convenience.
    -- If both the min/max are in the normal range (score of 0), then the average value is stored.
    , preiculos
    , agegroup
    , mingcs as gcs
    ,  case when heartrate_max is null then null
        when heartrate_max > 125 then heartrate_max
        when heartrate_min < 33 then heartrate_min
        when heartrate_max >= 107 and heartrate_max <= 125 then heartrate_max
        when heartrate_max >= 89 and heartrate_max <= 106 then heartrate_max
        else (heartrate_min+heartrate_max)/2 end as heartrate
    ,  case when meanbp_min is null then null
        when meanbp_min < 20.65 then meanbp_min
        when meanbp_min < 51 then meanbp_min
        when meanbp_max > 143.44 then meanbp_max
        when meanbp_min >= 51 and meanbp_min < 61.33 then meanbp_min
        else (meanbp_min+meanbp_max)/2 end as meanbp
    ,  case when resprate_min is null then null
        when resprate_min <   6 then resprate_min
        when resprate_max >  44 then resprate_max
        when resprate_max >  30 then resprate_max
        when resprate_max >  22 then resprate_max
        when resprate_min <  13 then resprate_min
        else (resprate_min+resprate_max)/2 end as resprate
    ,  case when tempc_max is null then null
        when tempc_max > 39.88 then tempc_max
        when tempc_min >= 33.22 and tempc_min <= 35.93 then tempc_min
        when tempc_max >= 33.22 and tempc_max <= 35.93 then tempc_max
        when tempc_min < 33.22 then tempc_min
        when tempc_min > 35.93 and tempc_min <= 36.39 then tempc_min
        when tempc_max >= 36.89 and tempc_max <= 39.88 then tempc_max
        else (tempc_min+tempc_max)/2 end as temp
    ,  urineoutput
    ,  mechvent
    ,  electivesurgery
    from cohort co
)

, score as
(
    select s.*
        , coalesce(age_score,0)
        + coalesce(preiculos_score,0)
        + coalesce(gcs_score,0)
        + coalesce(heartrate_score,0)
        + coalesce(meanbp_score,0)
        + coalesce(resprate_score,0)
        + coalesce(temp_score,0)
        + coalesce(urineoutput_score,0)
        + coalesce(mechvent_score,0)
        + coalesce(electivesurgery_score,0)
        as oasis
    from scorecomp s
)

select
  admissionid
  , icustay_age_group
  , hospital_expire_flag
  , icustay_expire_flag
  , oasis
  -- Calculate the probability of in-hospital mortality
  , 1 / (1 + exp(- (-6.1746 + 0.1275*(oasis) ))) as oasis_PROB
  , agegroup, age_score
  , preiculos, preiculos_score
  , gcs, gcs_score
  , heartrate, heartrate_score
  , meanbp, meanbp_score
  , resprate, resprate_score
  , temp, temp_score
  , urineoutput, urineoutput_score
  , mechvent, mechvent_score
  , electivesurgery, electivesurgery_score
from score
order by admissionid;