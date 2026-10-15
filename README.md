# A Multimodal Physiological Dataset for Hand Gesture Classification for Prosthetic Hand Control

This GitHub repository contains the **code, processing and validation scripts, and supporting documentation** for the dataset described below. The complete dataset files are hosted separately on Figshare and are not included in this repository.

## Dataset Overview

The dataset supports research on motor execution (ME), motor imagery (MI), and hand gesture classification for prosthetic hand control. It combines three physiological measurement modalities:

- **Electroencephalography (EEG):** 13 channels
- **Surface electromyography (EMG):** 3 channels
- **Bioimpedance (BioZ):** Electrical bioimpedance measurements

EEG and EMG recordings were acquired using an OpenBCI Cyton and Daisy configuration. The first three acquisition channels were used for EMG, while the remaining channels were used for EEG analysis. BioZ measurements were acquired separately and should not be interpreted as sample-level synchronized with EEG and EMG.

## Repository Scope

This repository is intended for:

- Dataset organization and metadata handling
- Signal preprocessing and validation
- Quality-control procedures
- Reproducibility resources and processing scripts
- Documentation explaining the dataset and its use

**The full dataset is not stored in this GitHub repository.** Download the data from the archival Figshare record.

## Dataset Access

- **Archival repository:** [Figshare](https://doi.org/10.6084/m9.figshare.34421721)
- **Dataset DOI:** [10.6084/m9.figshare.34421721](https://doi.org/10.6084/m9.figshare.34421721)
- **Dataset license:** Creative Commons Attribution 4.0 International (CC BY 4.0)

The Figshare record is the designated source for the dataset files, metadata, accompanying documentation, and quality-control reports. Check the record for current publication status and the applicable terms of use.

## Dataset Contents

The dataset release is documented as containing:

- Raw motor execution recordings
- Raw motor imagery recordings
- EEG and EMG recordings
- Native BioZ `.spec` files
- Processed modality-specific files
- Metadata and cohort information
- Recording-level annotation manifests
- Quality-control reports
- SHA-256 checksum catalogs

The frozen Dataset V1.0 inventory reports:

| Component | Count |
|---|---:|
| Raw motor execution recordings | 840 |
| Raw motor imagery recordings | 525 |
| BioZ measurement units | 1,890 |
| Native BioZ `.spec` files | 3,780 |
| Processed motor execution EEG files | 840 |
| Processed motor imagery EEG files | 525 |
| Processed EMG files | 840 |
| Processed BioZ files | 3,780 |
| Total processed NPZ files | 5,985 |

Counts refer to recordings, measurement units, or files as indicated.

## Annotation Policy

The dataset documentation does not claim unsupported event, trigger, or repetition annotations. Where these details are absent from the original recordings, they are reported as unavailable. No event IDs or repetition IDs are fabricated. Recording-level class labels are used where supported by the available metadata.

## Integrity and Reproducibility

Dataset V1.0 underwent a formal freeze and release-integrity procedure. The release report records that raw and processed data were not modified and that preprocessing, feature extraction, machine-learning experiments, and fusion generation were not rerun during the freeze/release stage.

SHA-256 checksum catalogs are supplied with the release to help users verify file integrity. Follow the repository scripts and documentation for their intended scope and usage; review each script before running it and configure paths for your own environment.

## Citation

Please cite the dataset and its associated descriptor when using the data in publications or other research outputs. Verify the final citation against the metadata shown on Figshare after publication.

**Proposed dataset citation:**

Faruk, O., Sayed, M. A., Nahiyan, K. M. T., Kadir, M. A., Wadud, M. S. I., & Al Imran, M. I. (2026). *A Multimodal Physiological Dataset for Hand Gesture Classification for Prosthetic Hand Control*. Figshare. https://doi.org/10.6084/m9.figshare.34421721

## Versioning

The documented release is **Dataset V1.0 — Final Frozen Release**. Future changes should use explicit version identifiers and be documented transparently rather than silently modifying V1.0.

## Code License

Code in this GitHub repository is intended to be released under the **MIT License**, provided that the repository includes an MIT `LICENSE` file. The code license is separate from the dataset license. The dataset is governed by the terms specified on its Figshare record.

## Contact

**Omar Faruk**  
Department of Biomedical Physics & Technology  
University of Dhaka  
Dhaka, Bangladesh  

**Email:** [faruk@bmpt.du.ac.bd](mailto:faruk@bmpt.du.ac.bd)

For questions about the code, documentation, or dataset use, consult the Figshare record or contact the repository maintainer.
