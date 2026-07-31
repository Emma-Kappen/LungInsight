import pandas as pd

# cBioPortal clinical files have comment lines starting with "#" before the header
patient = pd.read_csv("datasets/msk_chord_2024/data_clinical_patient.txt", sep="\t", comment="#")
sample  = pd.read_csv("datasets/msk_chord_2024/data_clinical_sample.txt", sep="\t", comment="#")

# Check the exact column name/values first
print(sample.columns.tolist())
print(sample["CANCER_TYPE"].value_counts().head(20))