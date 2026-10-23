# Dataset feature inventory

This is a schema/sample inspection, not a full data-quality or label-validity audit.
Table files are sampled up to 5,000 rows; missingness, examples, and unique counts are sample-based.
Exact or normalized column-name overlap does not establish semantic equivalence.

## Dataset locations
- **MSK-CHORD-2024:** `C:\Users\emkap\Documents\Projects\LungInsight\datasets\msk_chord_2024`
  - Exists: `True`
  - Candidate tabular files found: `70`
- **TCGA-LUAD:** `C:\Users\emkap\Documents\Projects\LungInsight\datasets\TCGA-LUAD`
  - Exists: `True`
  - Candidate tabular files found: `411`
- **TCGA-LUSC:** `C:\Users\emkap\Documents\Projects\LungInsight\datasets\TCGA-LUSC`
  - Exists: `True`
  - Candidate tabular files found: `329`

## File inventory
dataset         inspection_status                                                                           
MSK-CHORD-2024  ERROR: ParserError: Error tokenizing data. C error: Expected 1 fields in line 3, saw 2\n          1
                ERROR: ParserError: Error tokenizing data. C error: Expected 1 fields in line 6, saw 25034\n      1
                ERROR: ParserError: Error tokenizing data. C error: Expected 1 fields in line 6, saw 25040\n      4
                ok                                                                                               64
TCGA-LUAD       ok                                                                                              411
TCGA-LUSC       ok                                                                                              329

## Feature counts
dataset
MSK-CHORD-2024    25324
TCGA-LUAD           342
TCGA-LUSC           306

### Semantic group counts
dataset         semantic_group                
MSK-CHORD-2024  age_or_demographic                    5
                binary_candidate                  13342
                categorical_or_low_cardinality    11903
                dataset_or_site_metadata             30
                date_or_time                          3
                identifier                            5
                molecular_or_genomic                 14
                performance_status                    2
                race_or_ethnicity                     4
                sex_or_gender                         2
                smoking                               2
                stage                                 4
                survival_or_followup                  2
                target_or_diagnosis_candidate         6
                treatment                             3
TCGA-LUAD       age_or_demographic                    5
                binary_candidate                      2
                categorical_or_low_cardinality      189
                dataset_or_site_metadata             15
                date_or_time                          6
                identifier                           14
                imaging_or_nodule_candidate           2
                molecular_or_genomic                  8
                performance_status                    2
                race_or_ethnicity                     2
                sex_or_gender                         1
                smoking                               5
                stage                                 4
                survival_or_followup                 30
                target_or_diagnosis_candidate        27
                treatment                            32
TCGA-LUSC       age_or_demographic                    4
                categorical_or_low_cardinality      170
                dataset_or_site_metadata             13
                date_or_time                          6
                identifier                           14
                imaging_or_nodule_candidate           2
                molecular_or_genomic                  7
                performance_status                    2
                race_or_ethnicity                     2
                sex_or_gender                         1
                smoking                               5
                stage                                 4
                survival_or_followup                 28
                target_or_diagnosis_candidate        20
                treatment                            28

## Inspection errors: 6
- MSK-CHORD-2024 | msk_chord_2024\case_lists\cases_all.txt | ParserError: Error tokenizing data. C error: Expected 1 fields in line 6, saw 25040

- MSK-CHORD-2024 | msk_chord_2024\case_lists\cases_cna.txt | ParserError: Error tokenizing data. C error: Expected 1 fields in line 6, saw 25034

- MSK-CHORD-2024 | msk_chord_2024\case_lists\cases_cnaseq.txt | ParserError: Error tokenizing data. C error: Expected 1 fields in line 6, saw 25040

- MSK-CHORD-2024 | msk_chord_2024\case_lists\cases_sequenced.txt | ParserError: Error tokenizing data. C error: Expected 1 fields in line 6, saw 25040

- MSK-CHORD-2024 | msk_chord_2024\case_lists\cases_sv.txt | ParserError: Error tokenizing data. C error: Expected 1 fields in line 6, saw 25040

- MSK-CHORD-2024 | msk_chord_2024\meta_study.txt | ParserError: Error tokenizing data. C error: Expected 1 fields in line 3, saw 2


## Output files
- `source_file_inventory.csv`
- `all_features_long.csv`
- `features_by_dataset.csv`
- `normalized_feature_overlap.csv`
- `feature_name_crosswalk.csv`
- `inspection_errors.json`
- `existing_model_audit_files.json`