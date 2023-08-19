-- Charlson Comorbidity Scoring
-- Reference : Deyo R A, Cherkin D C, Ciol M A. Adapting a clinical comorbidity index for use with ICD-9-CM administrative databases.[J]. 
--             Journal of Clinical Epidemiology, 1992, 45(6):613-619.
-- 2019.04.29 & 2021.10.06 & 2021.12.11 & 2022.01.07
-- Xiaoli Liu

drop table if exists `db_name.charlson_comorbidity_eicu`;
create table `db_name.charlson_comorbidity_eicu` as

with diagnosis_new_0 as (
  SELECT patientunitstayid, SPLIT(icd9code, ',') as icd9code
  FROM `physionet-data.eicu_crd.diagnosis`
)

, diagnosis_new_1 as (
    select patientunitstayid, icd9code
    from diagnosis_new_0
    cross join unnest(diagnosis_new_0.icd9code) as icd9code
)

, diagnosis_new as (
	select patientunitstayid, replace(icd9code, ' ', '') as icd9_code
	from diagnosis_new_1
)

, com AS
(
    SELECT
        patientunitstayid

        -- Myocardial infarction
        , MAX(CASE WHEN
            SUBSTR(icd9_code, 1, 3) IN ('410','412','I21','I22') THEN 1
            WHEN icd9_code = 'I25.5' THEN 1             
            ELSE 0 END) AS myocardial_infarct

        -- Congestive heart failure
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) = '428'
            OR
            icd9_code in ('402.01', '402.91', '404.91', '404.93')
            --SUBSTR(icd9_code, 1, 5) IN ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493')
            OR
            icd9_code in ('425.4', '425.5', '425.7', '425.8', '425.9')
            OR
            SUBSTR(icd9_code, 1, 3) IN ('I43','I50')
            OR
            icd9_code in ('I11.0', 'I13.2', 'I25.5', 'I42.0', 'I42.6', 'I42.8') THEN 1
            ELSE 0 END) AS congestive_heart_failure

        -- Peripheral vascular disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('440','441', 'I70','I71')
            OR
            icd9_code in ('437.3', '557.1', '557.9')
            -- SUBSTR(icd9_code, 1, 4) IN ('0930','4373','4471','5571','5579','V434')
            OR
            icd9_code in ('443.24','443.9', 'I73.9', 'K55.9')
            -- SUBSTR(icd9_code, 1, 4) BETWEEN '4431' AND '4439'
            THEN 1 
            ELSE 0 END) AS peripheral_vascular_disease

        -- Cerebrovascular disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) BETWEEN '430' AND '438'
            OR
            icd9_code in ('G45.9')
            OR
            SUBSTR(icd9_code, 1, 2) = 'I6'
            --OR SUBSTR(icd9_code, 1, 5) = '36234'
            THEN 1 
            ELSE 0 END) AS cerebrovascular_disease

        -- Dementia
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) = '290'
            OR
            icd9_code in ('294.10', 'F01.5', 'F02.8', 'F03', 'G30.9', 'F05')
            -- SUBSTR(icd9_code, 1, 4) IN ('2941','3312')
            THEN 1 
            ELSE 0 END) AS dementia

        -- Chronic pulmonary disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) BETWEEN '490' AND '505'
            OR
            icd9_code in ('416.8','416.9', 'I27.9')
            OR
            SUBSTR(icd9_code, 1, 2) IN ('J4', 'J6')
            --SUBSTR(icd9_code, 1, 4) IN ('4168','4169','5064','5081','5088')
            THEN 1 
            ELSE 0 END) AS chronic_pulmonary_disease

        -- Rheumatic disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) in ('725', 'M32', 'M33', 'M34', 'M35', 'M36')
            OR
            icd9_code in ('446.5','710.0','710.1','710.2','710.3','710.4',
                '714.0','714.1','714.2','714.8', 'M06.9')
            -- SUBSTR(icd9_code, 1, 4) IN ('4465','7100','7101','7102','7103','7104','7140','7141','7142','7148')
            THEN 1 
            ELSE 0 END) AS rheumatic_disease

        -- Peptic ulcer disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('531','532','533','534','K25','K26','K27','K28')
            THEN 1 
            ELSE 0 END) AS peptic_ulcer_disease

        -- Mild liver disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('570','571', 'B18','K73','K74')
            OR
            icd9_code in ('573.3', '573.4', '573.9', 'V42.7', 'K70.1', 'K70.3', 'K76.3', 'K76.9', 'Z94.4')
            --SUBSTR(icd9_code, 1, 4) IN ('0706','0709','5733','5734','5738','5739','V427')
            -- OR SUBSTR(icd9_code, 1, 5) IN ('07022','07023','07032','07033','07044','07054')
            THEN 1 
            ELSE 0 END) AS mild_liver_disease

        -- Diabetes without chronic complication
        , MAX(CASE WHEN
            SUBSTR(icd9_code, 1, 3) IN ('250')
            OR
            icd9_code in ('E10.1', 'E10.11', 'E10.65', 'E10.9', 'E11.65', 'E11.9', 'E13.00')
            --SUBSTR(icd9_code, 1, 4) IN ('2500','2501','2502','2503','2508','2509') 
            THEN 1 
            ELSE 0 END) AS diabetes_without_cc

        -- Diabetes with chronic complication (actually didn't have)
        , MAX(CASE WHEN 
            icd9_code IN ('250.4','250.5','250.6','250.7')
            THEN 1 
            ELSE 0 END) AS diabetes_with_cc

        -- Hemiplegia or paraplegia
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('342','343','G81','G82')
            OR
            icd9_code in ('344.00', '344.1')
            --SUBSTR(icd9_code, 1, 4) IN ('3341','3440','3441','3442','3443','3444','3445','3446','3449')
            THEN 1 
            ELSE 0 END) AS paraplegia

        -- Renal disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('582','585','586','V56', 'V42','N18','N19')
            -- OR SUBSTR(icd9_code, 1, 4) IN ('5880','V420','V451')
            -- OR SUBSTR(icd9_code, 1, 4) BETWEEN '5830' AND '5837'
            OR
            icd9_code in ('404.93')
            --SUBSTR(icd9_code, 1, 5) IN ('40301','40311','40391','40402','40403','40412','40413','40492','40493')          
            THEN 1 
            ELSE 0 END) AS renal_disease

        -- Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) BETWEEN '140' AND '172'
            OR
            icd9_code in ('174.9', '175.9', '179', '180.9', '183', '185', '186.9', '188.9', '189', '190.9', '191.9', '192.2', '192.9', '193', '194.3')
            -- SUBSTR(icd9_code, 1, 4) BETWEEN '1740' AND '1958'
            OR
            SUBSTR(icd9_code, 1, 3) BETWEEN '200' AND '208'
            OR 
            SUBSTR(icd9_code, 1, 3) = '238.6'
            OR
            SUBSTR(icd9_code, 1, 2) IN ('C5','C6','C7','C8','C9','C0','C1','C2')
            OR
            SUBSTR(icd9_code, 1, 3) IN ('C43', 'C32', 'C34', 'C37', 'C41')
            --SUBSTR(icd9_code, 1, 4) = '2386'
            THEN 1 
            ELSE 0 END) AS malignant_cancer

        -- Moderate or severe liver disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('456.0','456.1','456.2')
            OR
            icd9_code in ('572.2', '572.4', 'I85.01', 'K72.9', 'K72.91')
            --SUBSTR(icd9_code, 1, 3) BETWEEN '5722' AND '5728'
            THEN 1 
            ELSE 0 END) AS severe_liver_disease

        -- Metastatic solid tumor
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('196','197','198','199','C77','C78','C79','C80')
            THEN 1 
            ELSE 0 END) AS metastatic_solid_tumor

        -- AIDS/HIV
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('042','043','044','B20','B21','B22','B24')
            THEN 1 
            ELSE 0 END) AS aids
    FROM diagnosis_new
    GROUP BY patientunitstayid
)

, ag AS
(
    SELECT 
        patientunitstayid
        , age
        , CASE WHEN age <= 40 THEN 0
    WHEN age <= 50 THEN 1
    WHEN age <= 60 THEN 2
    WHEN age <= 70 THEN 3
    ELSE 4 END AS age_score
    FROM (
		SELECT patientunitstayid, cast(case when age = '> 89' then '91.4' else age end as numeric) as age
		FROM `physionet-data.eicu_crd_derived.icustay_detail`
		where age not like ''
	)
)

SELECT 
    com.patientunitstayid
    , ag.age_score
    , myocardial_infarct
    , congestive_heart_failure
    , peripheral_vascular_disease
    , cerebrovascular_disease
    , dementia
    , chronic_pulmonary_disease
    , rheumatic_disease
    , peptic_ulcer_disease
    , mild_liver_disease
    , diabetes_without_cc
    , diabetes_with_cc
    , paraplegia
    , renal_disease
    , malignant_cancer
    , severe_liver_disease 
    , metastatic_solid_tumor 
    , aids
    -- Calculate the Charlson Comorbidity Score using the original
    -- weights from Charlson, 1987.
    , coalesce(age_score,0)
    + coalesce(myocardial_infarct,0) + coalesce(congestive_heart_failure,0) + coalesce(peripheral_vascular_disease,0)
    + coalesce(cerebrovascular_disease,0) + coalesce(dementia,0) + coalesce(chronic_pulmonary_disease,0)
    + coalesce(rheumatic_disease,0) + coalesce(peptic_ulcer_disease,0)
    + coalesce(GREATEST(mild_liver_disease, 3*severe_liver_disease),0)
    + coalesce(GREATEST(2*diabetes_with_cc, diabetes_without_cc),0)
    + coalesce(GREATEST(2*malignant_cancer, 6*metastatic_solid_tumor),0)
    + coalesce(2*paraplegia,0) + coalesce(2*renal_disease,0) 
    + coalesce(6*aids,0)
    AS charlson_comorbidity_index
FROM com
LEFT JOIN ag
ON com.patientunitstayid = ag.patientunitstayid
;