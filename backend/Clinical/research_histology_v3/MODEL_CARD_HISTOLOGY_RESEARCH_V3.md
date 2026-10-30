# LungInsight Histology Research Model v3

## Model identification

- Model: LungInsight TCGA LUAD-LUSC Research Histology Model
- Version: research-v3
- Audit generated: 2026-10-30T17:02:48.067712+00:00
- Artifact: cancer-type-luad-lusc-research-v3.joblib
- Framework: scikit-learn 1.9.1
- Intended status: research-only exploratory model

## Intended task

Exploratory binary classification of the TCGA-LUAD and TCGA-LUSC
source cohorts.

The model target was assigned from cohort membership:
- LUAD cohort -> LUAD
- LUSC cohort -> LUSC

## Training population

- TCGA-LUAD: 522
- TCGA-LUSC: 504
- Total: 1026

No MSK-CHORD, LIDC-IDRI, or other cohort was used for this model.

## Primary predictors

- Age at initial pathologic diagnosis
- Sex (`gender`)
- Tobacco smoking history indicator
- Tobacco smoking pack-years

Race is excluded from the primary model and was evaluated only as a
sensitivity feature.

## Excluded variables

- Ethnicity
- KPS
- ECOG
- ICD-O-3 histology
- ICD-O-3 site
- ICD-10
- Histologic diagnosis fields
- Treatment, response, and outcome variables
- Pathological stage
- Imaging measurements
- Brock malignancy probability

## Missing data

The original fitted pipelines use median imputation for numeric features
and an explicit missing category for categorical features.

Pack-years are not inferred from smoking status. Missing values remain
missing until handled by the fitted pipeline.

## Major data limitations

The two source cohorts are single-class:
- TCGA-LUAD contains LUAD labels only.
- TCGA-LUSC contains LUSC labels only.

Cohort identity and target are therefore perfectly confounded.

Random train/test splits and cross-validation cannot establish
independent histology discrimination. Metrics describe separation of
these supplied cohorts and must not be interpreted as estimates of
generalisation to an independent clinical population.

Pack-years have differential missingness between the cohorts. This
requires consideration when interpreting model behaviour.

Race documentation categories such as Not Available and Not Evaluated
are data documentation states, not biological or demographic categories.
Race is not part of the primary model.

## Intended use

Permitted:
- Technical research
- Exploratory analysis of the supplied TCGA cohorts
- Software pipeline testing
- Investigation of feature and missing-data sensitivity

Not permitted:
- Clinical diagnosis
- Clinical decision support
- Patient-level treatment decisions
- Clinical risk stratification
- Claims of independent histology discrimination
- Claims of validated performance in external populations

## Deployment status

This model is not integrated into the LungInsight API.

The existing v2 histology model and survival model remain separate and
unchanged.

## Validation status

Technical artifact loading and input handling were previously audited.
This robustness audit evaluates:
- Pack-years distribution and missingness
- Smoking source metadata consistency
- Race documentation-state sensitivity
- Alternative imputation configurations

These checks do not resolve cohort-target confounding.

## Required future evidence

Before any clinical use could be considered, an independent validation
design with both histology classes represented across the relevant
source populations would be required. Source harmonization, temporal
eligibility, calibration, and clinical applicability would also need
separate evaluation.

## Responsible interpretation

This is an exploratory cohort-discrimination research artifact, not a
clinically validated histology classifier.
