# 🚗 Car Damage Detection using Deep Learning

## 📌 Project Overview
This project presents a **Concept Demonstration** for automated car damage detection using **neural network and transfer learning**. The goal is to classify vehicle images into **six damage categories** based on front and rear damage conditions.

An interactive **Streamlit web application** is built to allow users to upload car images and receive **real-time damage classification results**.

---

## 🎯 Objectives
- Build and compare multiple CNN-based models for car damage classification
- Improve accuracy using **transfer learning**
- Deploy the best-performing model using **Streamlit**
- Validate feasibility for real-world use cases such as **insurance assessment and vehicle inspection**

---

## 🗂 Dataset Details
- **Total images:** ~2300  
- **Classes (6):**
  - `F_Breakage`
  - `F_Crushed`
  - `F_Normal`
  - `R_Breakage`
  - `R_Crushed`
  - `R_Normal`

---

## 🧠 Models Experimented

| Model | Description | Best Validation Accuracy |
|------|------------|--------------------------|
| Model 1 | Custom CNN | ~55% |
| Model 2 | CNN + Regularization | ~54% |
| Model 3 | ResNet18 (Transfer Learning) | ~66% |
| Model 4 | EfficientNet (Transfer Learning) | ~65% |
| ✅ Model 5 | **ResNet50 (Transfer Learning)** | **82.26%** |

---

## 🏆 Final Model
- **Architecture:** ResNet-50 (Transfer Learning)
- **Framework:** PyTorch
- **Validation Accuracy:** **82.26%**
- **Why ResNet50?**
  - Better feature extraction
  - Improved generalization on limited data
  - Reduced overfitting compared to custom CNNs
 

---

## 🛠 Tech Stack
- **Programming:** Python
- **Deep Learning:** PyTorch
- **Models:** CNN, ResNet18, EfficientNet, ResNet50
- **Data Processing:** NumPy, PIL
- **Visualization:** Matplotlib,Seaborn
- **Deployment:** Streamlit

---

## 🚀 Streamlit Application
### Features:
- Upload car images (JPG/PNG)
- Real-time damage classification
- Simple and interactive UI

---

## 🔧 Requirements

To run this project locally, ensure the following requirements are met:

### Software Requirements
- Python 3.8 or higher
- Git
- Streamlit

### Python Libraries
- PyTorch
- Torchvision
- NumPy
- Pandas
- Pillow
- Matplotlib
- Seaborn
- Scikit-learn

Install all dependencies using:
```bash
pip install -r requirements.txt

### Run the app locally:
streamlit run ./app.py
