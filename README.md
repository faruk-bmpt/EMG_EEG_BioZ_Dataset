# A multimodal Physiological Dataset for Hand Gesture Classification for Prosthetic Hand Control

A multimodal biosignal dataset developed for research on **motor execution, motor imagery, and hand gesture recognition**, integrating electroencephalography (EEG), surface electromyography (EMG), and bioimpedance (BioZ) measurements.

## Dataset Overview

The dataset contains multimodal biosignal recordings associated with motor execution, motor imagery, and hand gesture tasks

### Modalities

* **EEG** — 13 channels focused on sensory and motor cortical regions
* **EMG** — 3 channels
* **Bioimpedance (BioZ)** — electrical bioimpedance measurements
* **Motor Execution (ME)** recordings
* **Motor Imagery (MI)** recordings

The EEG acquisition used an OpenBCI Cyton + Daisy configuration. The recording montage followed the OpenBCI GUI 10–20-based channel configuration, with the analysis focused on channels covering the sensory and motor cortical regions. The first three acquisition channels were used for EMG.

## Dataset Contents

The released dataset includes:

* Raw motor-execution recordings
* Raw motor-imagery recordings
* EEG recordings
* EMG recordings
* Native BioZ `.spec` files
* Processed modality-specific files
* Metadata and cohort information
* Recording-level annotation manifests
* Quality-control reports
* Dataset documentation
* SHA-256 integrity checksums
* Release documentation

## Dataset V1.0

Dataset Version: **V1.0**

The dataset underwent a formal freeze and release-integrity procedure before publication.

The frozen source dataset contains:

* **840** motor-execution raw recordings
* **525** motor-imagery raw recordings
* **1,890** BioZ measurement units
* **3,780** native BioZ `.spec` files
* **840** execution EEG processed files
* **525** motor-imagery EEG processed files
* **840** EMG processed files
* **3,780** BioZ processed files
* **5,985** processed NPZ files in total

The release was validated using SHA-256 checksums.

## Important Annotation Note

The dataset release does **not** fabricate event, trigger, or repetition annotations.

Where such information was not available in the original recordings:

* Event annotations are reported as unavailable.
* Trigger annotations are reported as unavailable.
* Repetition annotations are reported as unavailable.
* No annotation inference was performed.
* No event IDs were fabricated.
* No repetition IDs were fabricated.

This policy is intended to preserve the provenance and scientific integrity of the original recordings.

## Data Integrity

Dataset V1.0 was formally frozen before release.

The release validation confirmed:

* Raw data modification: **False**
* Processed data modification: **False**
* Preprocessing rerun during freeze/release: **False**
* Feature extraction during freeze/release: **False**
* Machine learning during freeze/release: **False**
* Fusion generation during freeze/release: **False**

SHA-256 checksum catalogs are provided with the release to support file-integrity verification.

## Repository Contents

This GitHub repository is intended primarily for:

* Dataset documentation
* Processing and validation scripts
* Metadata descriptions
* Reproducibility resources
* Dataset release information
* Links to the archival dataset

The complete dataset is **not stored directly in this GitHub repository** because of its size. The archival dataset should be accessed through the associated Zenodo record.

## Data Access

**Zenodo:** *DOI/link to be added after publication*

The Zenodo record is the archival source for the released Dataset V1.0.

## Citation

Please cite the associated dataset descriptor and Zenodo record when using this dataset.

**Dataset citation:** *To be added after publication.*

## Versioning

Current release:

**Dataset V1.0 — Final Frozen Release**

Future releases, if required, will use explicit version identifiers and will not silently modify Dataset V1.0.

## License

The source code and scripts in this repository are released under the **MIT License**.

The dataset itself is governed by the data-use and licensing terms specified in the associated Zenodo record.

## Contact

**Omar Faruk**
Department of Biomedical Physics and Technology
University of Dhaka
Dhaka, Bangladesh

For questions regarding dataset access, documentation, or scientific use, please refer to the associated Zenodo record and dataset descriptor.
