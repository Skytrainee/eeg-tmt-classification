# EEG-TMT Classification: FFT and Wavelet-Based Cognitive Task Analysis

This repository contains the source code used in our study on EEG-based cognitive task classification across different experimental environments (Paper–Pencil, Tablet, and VR).  
We combine **Fast Fourier Transform (FFT)** and **Wavelet Transform (WT)** for feature extraction, and employ multiple machine learning and deep learning models (SVM, CNN, PC-LSTM) for classification.

---

##  Overview

- **Objective**: To explore EEG signal characteristics and classification performance under different cognitive task modalities.
- **Tasks**: Trail Making Test (TMT-A and TMT-B)
- **Feature Extraction**: FFT and Wavelet Transform
- **Models Used**: SVM, CNN, PC-LSTM
- **Environments**: Paper–Pencil (PP-TMT), Tablet (Tablet-TMT), and Virtual Reality (VR-TMT)

---

##  Repository Structure
eeg-tmt-classification/
│
├── data/ # (Not publicly available – requires request)
├── src/
│ ├── preprocessing/ # EEG preprocessing scripts
│ ├── feature_extraction/ # FFT & Wavelet feature computation
│ ├── models/ # CNN, PC-LSTM, and SVM training scripts
│ └── analysis/ # Statistical analysis and visualization
│
├── results/ # Classification results and figures
├── requirements.txt # Python dependencies
└── README.md # Project documentation
---

Data Availability

The EEG datasets used in this study are not publicly available due to:

Participant privacy and ethical review constraints

Institutional data export regulations

However, the datasets can be shared with qualified researchers for non-commercial academic use upon reasonable request.
Please contact the corresponding author for data access inquiries.

Contact
For questions, collaboration, or data requests, please contact:
[Shikai Liu] — [1661687958@qq.com]
