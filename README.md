# Comparative Evaluation of Three-Layer and BabelBrain-Based Skull Transmission Estimates

This repository contains scripts and notebooks for comparing three-layer model and BabelBrain simulation-based estimates of transcranial ultrasound stimulation (TUS) pressure transmission through the human skull.

The workflow includes:

1. Extraction of EEG-based scalp-normal trajectories from SimNIBS CHARM outputs.
2. CT- and ZTE-based skull-thickness estimation along EEG-normal trajectories.
3. EEG-trajectory-based BabelBrain acoustic simulations using CT- and ZTE-derived skull masks.
4. Three-layer analytical pressure-transmission modeling based on skull thickness and ultrasound frequency (not included in this repository).
5. Comparison of pressure transmission estimates across CT, ZTE, and analytical modeling approaches(not included in this repository).

---

## Acknowledgment

This repository includes code and workflow elements adapted from Zadeh et al. (2025):

**Zadeh AK, Puonti O, Sigurðsson B, Thielscher A, Monchi O, Pichardo S.**  

*Enhancing transcranial ultrasound stimulation planning with MRI-derived skull masks: a comparative analysis with CT-based processing.*  

Journal of Neural Engineering. 2025;22:016020.  
DOI: `10.1088/1741-2552/adab22`

The EEG trajectory-based BabelBrain simulation workflow and related processing structure were adapted from the workflow described by Zadeh et al. Additional comments, annotations, and modifications were added here to clarify the workflow and support comparison of CT-, ZTE-, and three-layer model-based estimates.

The three-layer analytical transmission model used for comparison is based on Attali et al. (2023):

**Attali D, Tiennot T, Schafer M, Fouragnan E, Sallet J, Caskey CF, Chen R, Darmani G, Bubrick EJ, Butler C, Stagg CJ, Klein-Flügge M, Verhagen L, Yoo SS, Butts Pauly K, Aubry JF.**  

*Three-layer model with absorption for conservative estimation of the maximum acoustic transmission coefficient through the human skull for transcranial ultrasound stimulation.*  

Brain Stimulation. 2023;16:48–55.  
DOI: `10.1016/j.brs.2022.12.005`

---

## Repository Overview

This repository is organized around three major components:

1. EEG trajectory generation
2. CT/ZTE skull-thickness extraction
3. BabelBrain and analytical pressure-transmission comparison

---

## Main Files

### `1_eeg_normals.py`

Extracts EEG 10–10 coordinate locations and local scalp-normal vectors from SimNIBS CHARM outputs.

This script:

- Loads the SimNIBS CHARM mesh.
- Reads EEG 10–10 electrode coordinates.
- Estimates the local scalp-normal vector near each EEG coordinate.
- Flips the normal vector inward.
- Exports EEG coordinates and inward normals as a CSV file.

Output columns:

```text
Name, R, A, S, Nx, Ny, Nz
```

These coordinates and inward normal vectors are used for trajectory-based skull-thickness estimation and BabelBrain simulations.

---

### `2_babelbrain_simulations_eeg.ipynb`

Runs EEG-trajectory-based BabelBrain acoustic simulations using CT- and ZTE-derived skull masks.

This notebook:

- Loads EEG-normal CSV files.
- Converts EEG-normal trajectories into Brainsight-style targets.
- Prepares CT and ZTE acoustic simulation inputs.
- Runs CT-based acoustic simulations.
- Runs ZTE-based acoustic simulations.
- Collects and summarizes BabelBrain output files.

The notebook was adapted from the Zadeh et al. workflow. Additional comments and section headings were added to clarify the processing steps.

---

### `3_CT_eeg_skull_thickness.py`

Estimates CT-derived skull thickness along EEG-normal trajectories.

This script:

- Searches CT subject folders.
- Loads each subject's CT image and EEG-normal CSV.
- Excludes predefined EEG locations.
- Samples CT intensity along each inward EEG-normal trajectory.
- Estimates skull thickness from raw CT cortical-bone peaks.
- Saves diagnostic plots, per-trajectory profiles, QC flags, and summary CSV files.

CT threshold:

```text
300 HU
```

Sampling window:

```text
+/- 40 mm around each EEG coordinate
```

---

### `4_ZTE_eeg_skull_thickness.py`

Estimates ZTE-derived skull thickness along EEG-normal trajectories.

This script:

- Searches ZTE subject folders.
- Loads each subject's ZTE image and EEG-normal CSV.
- Excludes predefined EEG locations.
- Samples ZTE intensity along each inward EEG-normal trajectory.
- Estimates skull thickness from raw ZTE high-intensity skull-profile peaks.
- Saves diagnostic plots, per-trajectory profiles, QC flags, and summary CSV files.

ZTE threshold:

```text
Native ZTE intensity > 300
```

Note: ZTE intensity is not Hounsfield units. The threshold is applied to native ZTE intensity values.

---

## Required Inputs

The workflow expects the following inputs for each participant:

### SimNIBS CHARM outputs

```text
m2m_<subject>/<subject>.msh
m2m_<subject>/eeg_positions/EEG10-10_Neuroelectrics.csv
```

### CT image

```text
*_CT.nii.gz
```

### ZTE image

```text
*_ZTE.nii.gz
```

### EEG-normal CSV files

Generated using:

```text
1_eeg_normals.py
```

### BabelBrain-compatible simulation inputs

CT- and ZTE-derived skull masks and acoustic simulation inputs are required for the BabelBrain simulation notebook.

---

## EEG-Normal Extraction

Run `1_eeg_normals.py` in a SimNIBS-compatible conda environment.

Example:

```bash
conda activate <path_to_simnibs>/simnibs_env/

python 1_eeg_normals.py \
~/Documents/TempForSim/SDR_0p42/m2m_SDR_0p42/SDR_0p42.msh \
~/Documents/TempForSim/SDR_0p42/m2m_SDR_0p42/eeg_positions/EEG10-10_Neuroelectrics.csv \
SDR_0p42_eeg_normals.csv
```

This generates normal trajectories for each EEG location and participant.

---

## Excluded EEG Locations

The following EEG locations are excluded from the final trajectory-based analyses:

```text
FP1, FP2, FPZ,
LPA, NZ, P9, P10, RPA,
AFZ, AF8, AF7, AF4, AF3,
IZ
```

After exclusions, the final workflow uses 53 retained EEG trajectories.

---

## Skull-Thickness Measurement

Skull thickness is estimated along each inward EEG-normal trajectory.

The general logic is:

1. Sample image intensity along the inward normal trajectory.
2. Identify high-intensity skull-profile segments.
3. Detect cortical skull-boundary peaks.
4. Estimate skull thickness as the distance from the outer skull boundary to the inner skull boundary.
5. Save per-trajectory profiles, diagnostic plots, and QC outputs.

For CT, the threshold is applied in HU.

For ZTE, the threshold is applied to native ZTE intensity.

---

## Frequency Note

Skull thickness is anatomical and is not frequency-dependent.

In the CT and ZTE skull-thickness scripts, the 250 kHz source folder may be used only to locate each subject's anatomical image and directory structure. The resulting skull-thickness estimates should not be interpreted as frequency-specific measurements.

---

## Output Files

The CT and ZTE skull-thickness scripts save:

```text
*_profile_<electrode>.csv
*_traj_<electrode>.png
*_skull_thickness_summary.csv
*_skull_thickness_less_than_2p5mm.csv
*_two_peak_span_rows.csv
*_flagged_QC_rows.csv
```

---

## Three-Layer Analytical Model

The analytical model estimates pressure transmission through a simplified three-layer structure:

```text
skin -> skull -> brain
```

The model incorporates:

- Tissue acoustic impedance
- Skull thickness
- Ultrasound frequency
- Skull absorption

This model provides a simplified pressure-transmission estimate that can be compared against subject-specific BabelBrain simulations.

---

## Dependencies

Core Python dependencies include:

```text
numpy
pandas
nibabel
matplotlib
scipy
simnibs
pathlib
shutil
re
```

BabelBrain is required for the acoustic simulation notebook.

---

## Notes

- CT-derived values are treated as the reference imaging-based skull model.
- ZTE-derived skull-thickness estimates are compared against CT-derived estimates.
- BabelBrain simulations are used to estimate trajectory-specific acoustic pressure transmission.
- The three-layer analytical model is used as a simplified comparison model.
- All scripts contain additional comments to clarify the workflow.

---

## Authors

Moon Jeong  
Dora Mackie

Credit to Zadeh et al (2025)

---

## References

1. Zadeh AK, Puonti O, Sigurðsson B, Thielscher A, Monchi O, Pichardo S. *Enhancing transcranial ultrasound stimulation planning with MRI-derived skull masks: a comparative analysis with CT-based processing.* Journal of Neural Engineering. 2025;22:016020. DOI: `10.1088/1741-2552/adab22`

2. Attali D, Tiennot T, Schafer M, Fouragnan E, Sallet J, Caskey CF, Chen R, Darmani G, Bubrick EJ, Butler C, Stagg CJ, Klein-Flügge M, Verhagen L, Yoo SS, Butts Pauly K, Aubry JF. *Three-layer model with absorption for conservative estimation of the maximum acoustic transmission coefficient through the human skull for transcranial ultrasound stimulation.* Brain Stimulation. 2023;16:48–55. DOI: `10.1016/j.brs.2022.12.005`
