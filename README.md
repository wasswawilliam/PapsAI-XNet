# PapsAI-XNet
Explainable hybrid deep learning framework integrating attention-guided CNNs and ANFIS for automated cervical cytology classification with transparent AI-assisted diagnosis.
# PapsAI XNet

**Explainable Hybrid Deep Learning for Automated Cervical Cytology Classification**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Status](https://img.shields.io/badge/Status-Research-green)

Official implementation of **PapsAI XNet**, an explainable hybrid deep learning framework that combines **attention-guided Convolutional Neural Networks (CNNs)** with an **Adaptive Neuro-Fuzzy Inference System (ANFIS)** for automated cervical cytology classification.

PapsAI XNet was developed to improve both **diagnostic accuracy** and **clinical interpretability**, making artificial intelligence more trustworthy for cervical cancer screening, particularly in resource-constrained healthcare settings.

---

## Overview

Cervical cancer remains one of the leading causes of cancer-related mortality among women worldwide, with the greatest burden occurring in low- and middle-income countries where access to expert cytotechnologists is limited.

Although modern deep learning models achieve high diagnostic accuracy, most operate as black-box systems that provide limited insight into their decision-making process. PapsAI XNet addresses this limitation by integrating explainable neuro-fuzzy reasoning directly into the classification pipeline.

The proposed framework combines:

- Attention-guided CNN feature extraction
- Feature normalization
- Adaptive Neuro-Fuzzy Inference System (ANFIS)
- Explainable fuzzy rule reasoning
- Lightweight architecture suitable for deployment on low-resource digital microscopy platforms

---

## Key Features

- Explainable AI for cervical cytology
- Hybrid CNN–ANFIS architecture
- Channel Attention Module (CAM)
- Feature normalization layer
- Gaussian fuzzy membership functions
- Compact fuzzy rule base
- High diagnostic accuracy
- Lightweight computational footprint
- Suitable for real-time deployment
- Designed for integration into the **PapsAI Digital Microscope** (https://papsai.org/)

---

## Framework Architecture

```
Pap-smear Image
        │
        ▼
Image Pre-processing
        │
        ▼
CNN Backbone
        │
Channel Attention Module
        │
Feature Normalization
        │
32-dimensional Feature Embedding
        │
        ▼
ANFIS Classifier
        │
Gaussian Membership Functions
        │
Fuzzy Rule Reasoning
        │
        ▼
Cytology Classification
```

---

## Performance

| Metric | PapsAI XNet |
|---------|------------:|
| Accuracy | **97.8%** |
| Sensitivity | **96.4%** |
| Specificity | **98.6%** |
| Precision | **97.1%** |
| F1-score | **96.7%** |

The framework consistently outperformed benchmark CNN models including:

- ResNet-18
- MobileNetV2

while providing intrinsic explainability through neuro-fuzzy reasoning.

---

## Repository Structure

```
PapsAI-XNet/
│
├── data/
├── datasets/
├── models/
├── src/
│   ├── cnn/
│   ├── attention/
│   ├── anfis/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── explainability/
│
├── notebooks/
├── configs/
├── outputs/
├── figures/
├── requirements.txt
├── train.py
├── evaluate.py
├── inference.py
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/PapsAI/PapsAI-XNet.git

cd PapsAI-XNet
```

Create a virtual environment

```bash
python -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

This repository uses the publicly available **Herlev Pap-smear Dataset**.

Please obtain the dataset from its original source and place it inside

```
datasets/Herlev/
```

The dataset is **not redistributed** in this repository due to licensing restrictions.

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python evaluate.py
```

---

## Inference

```bash
python inference.py --image sample.png
```

---

## Explainability

PapsAI XNet provides **intrinsic explainability** through:

- Gaussian Membership Functions
- Fuzzy Rule Surfaces
- Rule Activation Heatmaps
- Transparent Neuro-Fuzzy Reasoning

Unlike Grad-CAM or SHAP, explanations are generated directly by the classifier rather than as post-hoc visualizations.

---

## PapsAI Ecosystem

PapsAI XNet forms part of the broader **PapsAI Digital Microscope Ecosystem**, an ongoing research and innovation programme focused on developing affordable AI-powered digital pathology solutions for cervical cancer screening.

The PapsAI ecosystem includes:

- PapsAI Digital Microscope
- PapsAI XNet
- Whole-slide image analysis
- Clinical decision support
- Telepathology
- AI-assisted diagnosis

Future versions, including **PapsAI XNet V2**, will extend this work to whole-slide image analysis and integrated clinical workflows.

Learn more:

https://papsai.org

---

## Citation

If you use this work, please cite:

```
Wasswa W., et al.

Explainable Hybrid Deep Learning for Automated Cervical Cytology Classification.

Nature Scientific Reports.

2026.
```

---

## License

This project is released under the **Apache License 2.0**.

See the LICENSE file for details.

---

## Acknowledgements

We acknowledge all researchers who contributed to cervical cytology datasets and the development of explainable artificial intelligence for healthcare.

---

## Contact

Dr. William Wasswa

Mbarara University of Science and Technology

Uganda

Website:must.ac.ug
https://papsai.org

Email:wwasswa@must.ac.ug

---

## Disclaimer

This software is provided for **research purposes only**.

It is **not intended for direct clinical diagnosis** and should not be used as a substitute for professional medical judgment without appropriate regulatory approval and clinical validation.
