# Scientific Research at UFRJ: Galaxies and AI 🌌🤖

Welcome to the **`scientific-research-at-ufrj-galaxies-and-AI`** repository!

This repository serves as a centralized hub for my research, studies, and code development applying **Machine Learning (ML) and Deep Learning** techniques to astrophysical phenomena. The primary focus of this project is on galaxy morphological classification and cluster-finding algorithms, conducted as part of my scientific research at the **Universidade Federal do Rio de Janeiro (UFRJ)**.

---

## 📖 Project Overview

Modern astronomical surveys (like the Dark Energy Survey - DES, SDSS, etc.) generate massive datasets that cannot be analyzed manually. This project explores how Artificial Intelligence can automate and improve upon traditional astrophysical analysis.

The core objectives of this repository are to:

1. Replicate, understand, and build upon state-of-the-art ML models for **Galaxy Morphological Classification** (e.g., distinguishing spiral from elliptical galaxies using Convolutional Neural Networks).
2. Explore ML applications for **Galaxy Cluster-finding** using catalog data.
3. Serve as a study log blending fundamental astronomical concepts with modern data science.

## 📚 Foundational Literature & References

The code, notebooks, and models in this repository are deeply inspired by and built upon the following key texts and papers:

### **1. Galaxy Morphological Classification**

* **Morfometryka System:** Exploring morphometric coefficients (Concentration, Asymmetry, Smoothness, Entropy, Spirality) and Linear Discriminant Analysis (LDA) for classification.
> *Reference:* Ferrari, F., de Carvalho, R. R., & Trevisan, M. (2015). "MORFOMETRYKA—A NEW WAY OF ESTABLISHING MORPHOLOGICAL CLASSIFICATION OF GALAXIES." *The Astrophysical Journal*.


* **Deep Learning on DES Data:** Pushing the limits of automated morphological classifications and applying Convolutional Neural Networks (CNNs) to the Dark Energy Survey (DES) Year 3 data.
> *References:* > - Vega-Ferrero, J., et al. (2021). "Pushing automated morphological classifications to their limits with the Dark Energy Survey."
> * Cheng, T.-Y., et al. (2021). "Galaxy morphological classification catalogue of the Dark Energy Survey Year 3 data with convolutional neural networks."
> 
> 



### **2. Galaxy Cluster Detection**

* **COSMIC Algorithm:** Using machine intelligence to identify the brightest cluster galaxies and estimate cluster richness in optical catalogs.
> *Reference:* Tian, D.-C., et al. (2025). "COSMIC: A Galaxy Cluster–Finding Algorithm Using Machine Learning."



### **3. Astronomical Foundations**

* **Astrophysics Fundamentals:** Grounding the machine learning output in physical reality and fundamental astrophysical theory.
> *Reference:* Karttunen, H., et al. "Fundamental Astronomy, 5th Edition."



---

## 🗂️ Repository Structure

```text
scientific-research-at-ufrj-galaxies-and-AI/
│
├── data/                  # Datasets (ignored in git), catalogs, and image samples (e.g., SDSS, DES subsets)
├── docs/                  # Summaries of papers, literature reviews, and study notes
├── notebooks/             # Jupyter notebooks for EDA, model training, and visualization
│   ├── 01_morfometryka_exploration.ipynb
│   ├── 02_cnn_galaxy_classification.ipynb
│   └── 03_cluster_finding_cosmic.ipynb
├── src/                   # Python scripts (data preprocessing, model architectures, metrics)
│   ├── data_loader.py
│   ├── models/            # PyTorch/TensorFlow model definitions
│   └── utils.py           # Helper functions for FITS file handling and metric plotting
├── requirements.txt       # Project dependencies (Astropy, PyTorch/TensorFlow, scikit-learn, etc.)
└── README.md              # This file

```

---

## 🛠️ Tools & Technologies

This research leverages standard tools in both the astronomical and data science communities:

* **Data Processing & Astronomy:** `Astropy`, `NumPy`, `Pandas`, `SciPy`
* **Machine Learning:** `scikit-learn` (for LDA, Random Forests, metrics)
* **Deep Learning:** `TensorFlow` / `Keras` or `PyTorch` (for CNNs)
* **Visualization:** `Matplotlib`, `Seaborn`

---

## 🚀 Getting Started

To run the notebooks and code in this repository locally:

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/scientific-research-at-ufrj-galaxies-and-AI.git
cd scientific-research-at-ufrj-galaxies-and-AI

```


2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```


3. **Install the dependencies:**
```bash
pip install -r requirements.txt

```



---

## 📍 About

This repository is maintained by a student/researcher at the **Universidade Federal do Rio de Janeiro (UFRJ)**, bridging the gap between theoretical astrophysics and modern computational intelligence.

**Research Advisor:** Professor Arianna Cortesi

*Viva a ciência brasileira!* 🇧🇷🔬
