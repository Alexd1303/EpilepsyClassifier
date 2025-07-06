
# 🧠 NeuroSpike: AI-Powered EEG-Based Epilepsy Classification System

A deep learning pipeline for the automated classification of epilepsy grades using EEG signal features.  
Developed for *AI4ST 2024–2025 – Signal and Imaging Acquisition and Modelling in Healthcare*

> **Authors**  
> Alessandro Dubini (885957)  
> Elia Leonardo Martin (886366)  
> Alice Menna (888364)

---

## 🧠 Clinical Motivation

Epilepsy affects over **50 million people worldwide**, according to the [World Health Organization](https://www.who.int/news-room/fact-sheets/detail/epilepsy). Up to **70%** of patients could live seizure-free with accurate classification and proper treatment.

However, diagnosis often relies on subjective EEG interpretation. NeuroSpike seeks to support clinicians by providing **automated epilepsy grade classification**, assisting in treatment decisions between **pharmacological** and **surgical** interventions.

---

## 🎯 System Objectives

- Assist in differentiating surgical vs. pharmacological treatment candidates
- Reduce misdiagnosis and avoid unnecessary surgery
- Improve patient quality of life and optimize healthcare resource usage
- Enable explainable, real-time clinical decision support from EEG data

---

## ⚙️ Signal Processing Pipeline

### 🔬 Preprocessing Stages

1. **Finite Impulse Response (FIR) Filtering**
2. **Signal Scaling**
3. **Zero-Crossing Rate Extraction**

Zero-crossing rate is a key discriminative feature:  
- **Mean z-crossings (Class 3):** 18.05  
- **Mean z-crossings (Class 4):** 40.64

Signal enhancement via convolution increases separability.

---

## 🧪 Classification Models

### 1. 🧩 Decision Tree

**Best Configuration:**
```python
DecisionTreeClassifier(max_depth=3, min_samples_split=10, criterion='entropy')
```

**Performance (20 runs):**
- Accuracy: **97.87%**
- Sensitivity: **96.74%**
- Specificity: **98.99%**

### 2. 🔺 Support Vector Machine (SVM)

**Best Configuration:**
```python
SVC(kernel='poly', C=0.1, gamma=0.1)
```

**Performance (20 runs):**
- Accuracy: **93.37%**
- Sensitivity: **92.24%**
- Specificity: **94.50%**

### 3. 🌲 Random Forest

**Best Configuration:**
```python
RandomForestClassifier(
    max_depth=10,
    min_samples_split=10,
    criterion='gini',
    n_estimators=50,
    max_features='sqrt',
    bootstrap=False
)
```

**Performance (20 runs):**
- Accuracy: **89.37%**
- Sensitivity: **93.39%**
- Specificity: **84.74%**

---

## 📊 Feature Importance

Top EEG features contributing to classification:  
- `zero_crossing[350]`: 86.17%  
- `zero_crossing[165]`: 5.07%  
- `zero_crossing[250]`: 3.20%

---

## 🌐 Web Application

- Real-time EEG upload and classification
- Classifier comparison and visualization interface

---

## 🔐 Ethics & Privacy

> This tool is designed for educational and research purposes. No identifiable personal or medical data is stored. NeuroSpike does not replace a clinical diagnosis and should be used as a support system only.

---

## 📚 References

1. WHO Epilepsy Fact Sheet – [Link](https://www.who.int/news-room/fact-sheets/detail/epilepsy)
2. Scikit-learn Documentation (Decision Trees, SVM, Random Forest)

---

## 🔭 Future Work

- Integrate temporal sequence modeling (e.g., LSTMs)
- Include additional physiological features (e.g., EMG, ECG)
- Cloud deployment with secure patient data handling
- Extend to seizure prediction and real-time monitoring

---

**NEUROSPIKE – Supporting Data-Driven Neurology**


