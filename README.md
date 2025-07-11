# Dataverse-Africa-Cassava-Mosaic-Detector
Dataverse Africa Cassava Mosaic Detection system

# Cassava Mosaic Disease Detection System

This is a Proof-of-Concept (POC) for Dataverse Africa — a deep learning-based image classification system that detects Cassava Mosaic Disease using transfer learning with MobileNetV2.

## 🚀 Project Overview

Cassava is a vital crop in Africa, and early detection of diseases like Cassava Mosaic can greatly enhance food security. This system classifies cassava leaf images into:

1. **Cassava Mosaic** 🦠
2. **Healthy Cassava** 🌿
3. **Non-Cassava Plant** 🍀

The solution uses TensorFlow and MobileNetV2 for transfer learning, achieving over **98% accuracy** on the validation set.

---

## 📁 Project Structure

```
├── image.png                             # Background image used
├── cassava disease detection system.py   # Main Streamlit app
├── runtime                               # python version used
├── data_split/                           # Structured training/validation/test data
├── cassava_classifier_final.h5           # Best model saved as .h5
├── requirements.txt                      # Dependency list
├── .gitignore                            # Files to ignore in version control
└── README.md                             # This file
```

---

## 🛠 Features

- **Image Classification** using transfer learning (MobileNetV2)
- **Interactive Predictions** via Streamlit UI
- **Custom Image Upload Support**
- **Real-Time Accuracy & F1 Score Reporting**
- **Confusion Matrix & Visualization**

---

## ⚙️ Installation

```bash
git clone https://github.com/promibe/dataverse-africa-cassava-mosaic-detector.git
cd dataverse-africa-cassava-mosaic-detector
pip install -r requirements.txt
streamlit run "cassava disease detection system.py"
```

> Make sure to use **Python 3.10 or 3.11**, as TensorFlow does not yet support Python 3.13.

---

## 📷 Sample Predictions

You can test your own images by placing them in the `test_plants/` folder or using the upload button in the Streamlit UI.

---

## 🧠 Model Info

- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Fine-tuned:** Last 50 layers unfrozen for better generalization
- **Optimizer:** Adam with learning rate scheduler
- **Evaluation:** Precision, Recall, F1-Score, Confusion Matrix

---

## 📌 Author

Built by **Promise Ibediogwu Ekele** for Dataverse Africa.

---

## 📜 License

This project is for educational and non-commercial research purposes only.
