A Multimodal Physiological Dataset for Hand Gesture Classification for Prosthetic Hand Control
A multimodal biosignal dataset developed to support research on motor execution, motor imagery, and hand gesture recognition for prosthetic hand control. The dataset integrates electroencephalography (EEG), surface electromyography (EMG), and bioimpedance (BioZ) measurements to facilitate research into physiological signal processing, multimodal analysis, and gesture classification.
Dataset Overview
The dataset comprises physiological recordings associated with motor execution (ME), motor imagery (MI), and hand gesture tasks. It provides complementary neural, muscular, and electrical bioimpedance measurements for investigating hand gesture classification and related prosthetic control applications.
Modalities and Acquisition
•	Electroencephalography (EEG): 13 channels
•	Surface electromyography (EMG): 3 channels
•	Bioimpedance (BioZ): Electrical bioimpedance measurements
•	Motor execution (ME): EEG and EMG recordings acquired during motor execution tasks
•	Motor imagery (MI): EEG recordings acquired during motor imagery tasks
EEG and EMG acquisition used an OpenBCI Cyton and Daisy configuration. The recording montage followed the OpenBCI GUI's 10–20-based channel configuration, with EEG analysis focused on channels covering sensory and motor cortical regions. The first three acquisition channels were used for EMG.
BioZ measurements were acquired separately using a dedicated bioimpedance measurement system. They represent a distinct acquisition modality and should not be interpreted as sample-level synchronized with the EEG and EMG recordings.
Dataset Contents
The released dataset includes:
•	Raw motor execution recordings
•	Raw motor imagery recordings
•	EEG recordings
•	EMG recordings
•	Native BioZ .spec files
•	Processed, modality-specific data files
•	Metadata and cohort information
•	Recording-level annotation manifests
•	Quality-control reports
•	Dataset documentation and release notes
•	SHA-256 checksum catalogs for file-integrity verification
Dataset V1.0 — Final Frozen Release
Dataset version: V1.0
Dataset V1.0 underwent a formal dataset-freeze and release-integrity validation procedure before release.
The frozen source dataset contains:
Data component	Number of files or units
Raw motor execution recordings	840
Raw motor imagery recordings	525
BioZ measurement units	1,890
Native BioZ .spec files	3,780
Processed motor execution EEG files	840
Processed motor imagery EEG files	525
Processed EMG files	840
Processed BioZ files	3,780
Total processed NPZ files	5,985
The counts refer to the corresponding recordings, measurement units, or files, as indicated. The release includes SHA-256 checksum catalogs to support verification of file integrity.
Annotation Policy
The release does not fabricate event, trigger, or repetition annotations.
Where the original recordings do not provide the relevant information:
•	Event annotations are reported as unavailable.
•	Trigger annotations are reported as unavailable.
•	Repetition annotations are reported as unavailable.
•	No event or repetition information has been inferred or fabricated.
•	No artificial event IDs or repetition IDs have been assigned.
The dataset uses recording-level class labels where supported by the available metadata. This conservative annotation policy preserves data provenance and scientific integrity and avoids introducing unsupported temporal information.
Data Integrity and Reproducibility
Dataset V1.0 was formally frozen before release. The final release-integrity procedure reported the following:
•	Raw data modified during release: No
•	Processed data modified during release: No
•	Preprocessing rerun during freeze or release: No
•	Feature extraction performed during freeze or release: No
•	Machine-learning experiments performed during freeze or release: No
•	Fusion generation performed during freeze or release: No
SHA-256 checksum catalogs are included to support independent verification of the released files. These checksums help users determine whether files have changed relative to the catalog; they do not, by themselves, establish the scientific validity of the measurements.
Repository Contents
This GitHub repository provides supporting resources for working with the dataset, including:
•	Dataset documentation
•	Data-processing and validation scripts
•	Metadata descriptions
•	Reproducibility resources
•	Dataset release information
•	Links to the archival dataset
Because of its size, the complete dataset is not stored directly in this GitHub repository. The dataset files are distributed through the associated Figshare record.
Data Access
Archival repository: Figshare
Dataset DOI: https://doi.org/10.6084/m9.figshare.34421721
The Figshare record is the designated archival source for Dataset V1.0. It provides access to the released dataset files, accompanying documentation, metadata, and quality-control reports.
Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0).
Users may access, share, and adapt the dataset in accordance with the CC BY 4.0 license, subject to its attribution requirements and any applicable ethical or legal restrictions documented in the release.
Citation
Please cite the dataset and its associated descriptor when using these resources in research publications, presentations, or derivative work.
Proposed dataset citation:
Faruk, O., Sayed, M. A., Nahiyan, K. M. T., Kadir, M. A., Wadud, M. S. I., & Al Imran, M. I. (2026). A Multimodal Physiological Dataset for Hand Gesture Classification for Prosthetic Hand Control. Figshare. https://doi.org/10.6084/m9.figshare.34421721
Verify the final author formatting, publication year, title, and DOI against the citation generated by Figshare after publication.
Versioning
Current release: Dataset V1.0 — Final Frozen Release
Any future updates will be identified through explicit versioning. Changes will be documented transparently rather than silently modifying the contents of Dataset V1.0.
Code License
The source code and scripts in this GitHub repository are released under the MIT License, as specified by the repository's license file.
The code license is separate from the dataset license. Users should consult the Figshare record and the dataset documentation for the applicable terms governing use and redistribution of the dataset.
Contact
Omar Faruk
Department of Biomedical Physics & Technology
University of Dhaka
Dhaka, Bangladesh
Email: faruk@bmpt.du.ac.bd
For questions regarding dataset access, documentation, or scientific use, please refer to the associated Figshare record or contact the corresponding author.
