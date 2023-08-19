-- Charlson Comorbidity Scoring
-- Reference : Deyo R A, Cherkin D C, Ciol M A. Adapting a clinical comorbidity index for use with ICD-9-CM administrative databases.[J]. 
--             Journal of Clinical Epidemiology, 1992, 45(6):613-619.
-- 2018.05.08 & 2021.10.07 & 2021.12.11 (refer MIMIC-IV)
-- Xiaoli Liu

drop table if exists `db_name.charlson_mimiciii`;
create table `db_name.charlson_mimiciii` as

WITH diag AS
(
    SELECT 
        hadm_id, icd9_code
    FROM `physionet-data.mimiciii_clinical.diagnoses_icd` diag
)

, com AS
(
    SELECT
        ad.hadm_id

        -- Myocardial infarction
        , MAX(CASE WHEN
            SUBSTR(icd9_code, 1, 3) IN ('410','412')
            THEN 1 
            ELSE 0 END) AS myocardial_infarct

        -- Congestive heart failure
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) = '428'
            OR
            SUBSTR(icd9_code, 1, 5) IN ('39891','40201','40211','40291','40401','40403',
                          '40411','40413','40491','40493')
            OR 
            SUBSTR(icd9_code, 1, 4) BETWEEN '4254' AND '4259'
            THEN 1 
            ELSE 0 END) AS congestive_heart_failure

        -- Peripheral vascular disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('440','441')
            OR
            SUBSTR(icd9_code, 1, 4) IN ('0930','4373','4471','5571','5579','V434')
            OR
            SUBSTR(icd9_code, 1, 4) BETWEEN '4431' AND '4439'
            THEN 1 
            ELSE 0 END) AS peripheral_vascular_disease

        -- Cerebrovascular disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) BETWEEN '430' AND '438'
            OR
            SUBSTR(icd9_code, 1, 5) = '36234'
            THEN 1 
            ELSE 0 END) AS cerebrovascular_disease

        -- Dementia
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) = '290'
            OR
            SUBSTR(icd9_code, 1, 4) IN ('2941','3312')
            THEN 1 
            ELSE 0 END) AS dementia

        -- Chronic pulmonary disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) BETWEEN '490' AND '505'
            OR
            SUBSTR(icd9_code, 1, 4) IN ('4168','4169','5064','5081','5088')
            THEN 1 
            ELSE 0 END) AS chronic_pulmonary_disease

        -- Rheumatic disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) = '725'
            OR
            SUBSTR(icd9_code, 1, 4) IN ('4465','7100','7101','7102','7103',
                                                  '7104','7140','7141','7142','7148')
            THEN 1 
            ELSE 0 END) AS rheumatic_disease

        -- Peptic ulcer disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('531','532','533','534')
            THEN 1 
            ELSE 0 END) AS peptic_ulcer_disease

        -- Mild liver disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('570','571')
            OR
            SUBSTR(icd9_code, 1, 4) IN ('0706','0709','5733','5734','5738','5739','V427')
            OR
            SUBSTR(icd9_code, 1, 5) IN ('07022','07023','07032','07033','07044','07054')
            THEN 1 
            ELSE 0 END) AS mild_liver_disease

        -- Diabetes without chronic complication
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 4) IN ('2500','2501','2502','2503','2508','2509') 
            THEN 1 
            ELSE 0 END) AS diabetes_without_cc

        -- Diabetes with chronic complication
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 4) IN ('2504','2505','2506','2507')
            THEN 1 
            ELSE 0 END) AS diabetes_with_cc

        -- Hemiplegia or paraplegia
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('342','343')
            OR
            SUBSTR(icd9_code, 1, 4) IN ('3341','3440','3441','3442',
                                                  '3443','3444','3445','3446','3449')
            THEN 1 
            ELSE 0 END) AS paraplegia

        -- Renal disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('582','585','586','V56')
            OR
            SUBSTR(icd9_code, 1, 4) IN ('5880','V420','V451')
            OR
            SUBSTR(icd9_code, 1, 4) BETWEEN '5830' AND '5837'
            OR
            SUBSTR(icd9_code, 1, 5) IN ('40301','40311','40391','40402','40403','40412','40413','40492','40493')          
            THEN 1 
            ELSE 0 END) AS renal_disease

        -- Any malignancy, including lymphoma and leukemia, except malignant neoplasm of skin
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) BETWEEN '140' AND '172'
            OR
            SUBSTR(icd9_code, 1, 4) BETWEEN '1740' AND '1958'
            OR
            SUBSTR(icd9_code, 1, 3) BETWEEN '200' AND '208'
            OR
            SUBSTR(icd9_code, 1, 4) = '2386'
            THEN 1 
            ELSE 0 END) AS malignant_cancer

        -- Moderate or severe liver disease
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 4) IN ('4560','4561','4562')
            OR
            SUBSTR(icd9_code, 1, 4) BETWEEN '5722' AND '5728'
            THEN 1 
            ELSE 0 END) AS severe_liver_disease

        -- Metastatic solid tumor
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('196','197','198','199')
            THEN 1 
            ELSE 0 END) AS metastatic_solid_tumor

        -- AIDS/HIV
        , MAX(CASE WHEN 
            SUBSTR(icd9_code, 1, 3) IN ('042','043','044')
            THEN 1 
            ELSE 0 END) AS aids
    FROM `physionet-data.mimiciii_clinical.admissions` ad
    LEFT JOIN diag
    ON ad.hadm_id = diag.hadm_id
    GROUP BY ad.hadm_id
)

, ag AS
(
    SELECT 
        hadm_id
        , admission_age as age
        , CASE WHEN admission_age <= 40 THEN 0
    WHEN admission_age <= 50 THEN 1
    WHEN admission_age <= 60 THEN 2
    WHEN admission_age <= 70 THEN 3
    ELSE 4 END AS age_score
    FROM `physionet-data.mimiciii_derived.icustay_detail`
)

SELECT 
    ad.subject_id
    , ad.hadm_id
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
    , age_score
    + myocardial_infarct + congestive_heart_failure + peripheral_vascular_disease
    + cerebrovascular_disease + dementia + chronic_pulmonary_disease
    + rheumatic_disease + peptic_ulcer_disease
    + GREATEST(mild_liver_disease, 3*severe_liver_disease)
    + GREATEST(2*diabetes_with_cc, diabetes_without_cc)
    + GREATEST(2*malignant_cancer, 6*metastatic_solid_tumor)
    + 2*paraplegia + 2*renal_disease 
    + 6*aids
    AS charlson_comorbidity_index
FROM `physionet-data.mimiciii_clinical.admissions` ad
LEFT JOIN com
ON ad.hadm_id = com.hadm_id
LEFT JOIN ag
ON com.hadm_id = ag.hadm_id
;