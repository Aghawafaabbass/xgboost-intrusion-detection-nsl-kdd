# 🔐 Enhancing Intrusion Detection in Cybersecurity using XGBoost  
## A Performance Analysis on the NSL-KDD Dataset  


---

## 📌 Abstract

The growing sophistication of cyber threats demands advanced Intrusion Detection Systems (IDS) to safeguard network integrity. This repository presents a robust IDS built using **XGBoost** on the **NSL-KDD dataset**, an improved and balanced version of the KDD'99 dataset.

Leveraging XGBoost's strength in handling complex and imbalanced data, the project delivers high accuracy, precision, and interpretability. The implementation includes:

- ✅ Comprehensive preprocessing  
- ✅ Model training & evaluation  
- ✅ Feature importance & SHAP analysis  

> 💡 Designed for real-time cybersecurity use and analysis by security professionals.

---

## 🔑 Key Results

| **Metric**       | **Score**   |
|------------------|-------------|
| Accuracy         | 80.08%      |
| Precision        | 85.00%      |
| Recall           | 82.00%      |
| F1-Score         | 80.00%      |
| ROC-AUC          | 0.89        |
| PR-AUC           | 0.97        |
| Explainability   | ✅ SHAP + Feature Importance |

> 🧠 The model offers strong interpretability and excellent attack detection capability.

---

## 🧠 Key Contributions

- **Efficient IDS Design**: XGBoost-based IDS optimized for real-time use  
- **Comprehensive Preprocessing**: Label encoding, feature scaling, feature selection  
- **Interpretability**: SHAP + feature importance analysis  
- **Superior Performance**: Outperforms SVM, Random Forest, and Decision Trees  
- **Robustness**: <3% drop under noise or missing data (ablation tested)  
- **Error Analysis**: Identifies weaknesses (e.g., `nmap`, `guess_passwd` attacks)

---

## 📈 Results Summary

### 🔍 Performance by Class

- **Normal Traffic Recall**: 0.97  
- **Attack Traffic Precision**: 0.97  
- ⚠️ Slight overfitting (Train Accuracy: 99.99%) → consider early stopping

### 🧮 Confusion Matrix Insight

- False Negative Rate for attacks: ~32%  
- Indicates difficulty with covert attack detection (improvement needed)

---

## 🔬 Visualizations

| **Visualization**           | **File**                    | **Insight**                             |
|-----------------------------|-----------------------------|------------------------------------------|
| Confusion Matrix            | `confusion_matrix.png`      | Shows classification errors              |
| ROC Curve                   | `roc_curve.png`             | AUC ≈ 0.89                               |
| Precision-Recall Curve      | `precision_recall_curve.png`| PR-AUC ≈ 0.97                            |
| Feature Importance          | `feature_importance.png`    | Top 15 most important features           |
| Learning Curve              | `learning_curve.png`        | Detects overfitting                      |

---

## 🧪 Future Work

- Real-Time Detection via incremental learning  
- Cross-dataset validation (CIC-IDS2017, UNSW-NB15, TON_IoT)  
- Hybrid models (e.g., XGBoost + LSTM)  
- Advanced feature engineering  
- Better class imbalance handling  
- Edge deployment (model quantization)  
- Enhanced explainability (SHAP, LIME, dashboards)

---

## 📂 File Descriptions

| **File**                      | **Description**                                          |
|------------------------------|----------------------------------------------------------|
| `XGBoost_IDS_NSLKDD.ipynb`   | Jupyter Notebook (preprocessing, training, evaluation)   |
| `confusion_matrix.png`       | Confusion matrix on test data                            |
| `roc_curve.png`              | ROC Curve with AUC ≈ 0.89                                |
| `precision_recall_curve.png` | Precision-Recall Curve with AP ≈ 0.97                    |
| `feature_importance.png`     | Feature importance graph                                 |
| `learning_curve.png`         | Training vs Test accuracy curve                          |

---

## 📥 Dataset Access

Download the NSL-KDD dataset:

🔗 [University of New Brunswick – NSL-KDD Dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)  
(Files: `KDDTrain+.txt`, `KDDTest+.txt`)

--

## 🚀 How to Run the Project

### ✅ Option 1: Run in Google Colab

1. Open the notebook in Colab:  
   🔗 [Run in Google Colab](https://colab.research.google.com/github/Aghawafaabbass/xgboost-intrusion-detection-nsl-kdd/blob/main/XGBoost_IDS_NSLKDD.ipynb)
2. Upload the dataset files:  
   - `KDDTrain+.txt`  
   - `KDDTest+.txt`  
3. Run all cells:  
   - `Runtime > Run All` or press `Ctrl+F9`

---

### ✅ Option 2: Run Locally

#### 🛠️ Requirements

- Python 3.7+  
- Jupyter Notebook or JupyterLab

#### 📦 Install Libraries

```bash
pip install pandas numpy xgboost matplotlib seaborn scikit-learn
▶️ Launch Notebook
bash
Copy
Edit
jupyter notebook
Open XGBoost_IDS_NSLKDD.ipynb and run the cells after placing the dataset files (KDDTrain+.txt, KDDTest+.txt) in the same directory.

🏷️ Index Terms
XGBoost, Intrusion Detection System (IDS), NSL-KDD,
Network Security, Machine Learning, Explainable AI,
Cybersecurity, Anomaly Detection, Supervised Learning

👨‍🏫 Authors
Agha Wafa Abbas
Lecturer, School of Computing, Arden University, UK
Lecturer, IVY College of Management Sciences, Pakistan
📧 awabbas@arden.ac.uk | wafa.abbas.lhr@rootsivy.edu.pk


📜 Citation
Agha Wafa, A. (2025). Enhancing Intrusion Detection in Cybersecurity using XGBoost: A Performance Analysis on the NSL-KDD Dataset (1.0). Zenodo.
🔗 DOI: 10.5281/zenodo.16737594

🙏 Acknowledgements
Special thanks to:

University of New Brunswick – for maintaining the NSL-KDD dataset

Developers of open-source libraries:

XGBoost

Scikit-learn

Pandas

NumPy

Matplotlib

SHAP

ℹ️ About
This project delivers a scalable, interpretable, and high-performing Intrusion Detection System using XGBoost on the NSL-KDD dataset. It is intended for:

🔍 Researchers

🛡️ Cybersecurity professionals

🤖 Machine Learning enthusiasts

📄 License
© 2025 Agha Wafa Abbas — All rights reserved.


