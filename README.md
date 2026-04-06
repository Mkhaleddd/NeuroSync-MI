# NeuroSync-MI: Dual-Pathway EEG Classification

A comparative study of **Convolutional Neural Networks (CNN)** and **Common Spatial Patterns (CSP)** for Motor Imagery classification using the BCI Competition IV dataset.

## 🚀 The Challenge
Classifying intent from 22-channel EEG data is notoriously difficult due to the low Signal-to-Noise Ratio (SNR). This project implements two distinct philosophies to solve the "Left Hand vs. Right Hand" classification task.

---

## 🧠 Approach 1: Spectrogram-CNN (Deep Learning)
This method transforms 1D brainwaves into 2D "Power Images."

1. **Preprocessing:** 8-30 Hz Bandpass filter using `scipy.signal.filtfilt` (Zero-phase).
2. **Feature Extraction:** Short-Time Fourier Transform (STFT) to generate $Z_{xx}$.
3. **Log-Power Scaling:** We apply $f(x) = \ln(1+x)$ to balance the power of Delta vs. Beta frequencies.
4. **Architecture:** A 2D-CNN (`EEGSpectrumNet`) utilizing `AdaptiveAvgPool2d` to extract spatial-temporal features.

---

## 🛠 Approach 2: CSP + LDA (Classical ML)
The "Gold Standard" for Motor Imagery.

* **CSP (Common Spatial Patterns):** Mathematically finds spatial filters that maximize the variance of one class while minimizing the other.
* **LDA (Linear Discriminant Analysis):** A robust classifier that finds the linear boundary between the extracted spatial features.

---

## 📈 Key Insights
* **The Log Transform:** Crucial for CNN stability. Without it, high-frequency signals are invisible to the model.
* **Batch Normalization:** Used in the CNN to handle the non-stationary nature of EEG data (signals that change over time).
* **Cross-Validation:** 5-Fold CV is used to ensure the accuracy isn't just a result of a "lucky" data split.

---

## 📦 Requirements
- PyTorch
- NumPy & SciPy
- Scikit-Learn
- MNE (Optional, for CSP visualization)
