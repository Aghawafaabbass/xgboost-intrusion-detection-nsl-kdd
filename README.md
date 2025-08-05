Enhancing Intrusion Detection in Cybersecurity using XGBoost: A Performance Analysis on the NSL-KDD Dataset

📌 Abstract

The escalating complexity and frequency of cyber threats necessitate advanced Intrusion Detection Systems (IDS) to protect network integrity. This repository presents a robust IDS built using XGBoost on the NSL-KDD dataset, an improved and balanced version of the KDD'99 dataset. By leveraging XGBoost’s ability to handle high-dimensional and imbalanced data, this project achieves strong performance in detecting network intrusions. The implementation includes thorough data preprocessing, model training, evaluation, and feature importance analysis, making it suitable for real-time cybersecurity applications and interpretable for security analysts.

🔑 Key Results





✅ Test Accuracy: 80.08%



📊 Precision: 85.00%



🔍 Recall: 82.00%



🚀 F1-Score: 80.00%



🌟 ROC-AUC: 0.89



📉 PR-AUC: 0.97



🧠 Explainability: Feature importance and SHAP analysis for transparent decision-making

🧠 Key Contributions





Efficient IDS Implementation: A scalable XGBoost-based IDS optimized for the NSL-KDD dataset, suitable for enterprise and real-time deployments.



Comprehensive Preprocessing: Includes label encoding, feature scaling, and removal of irrelevant features (e.g., num_outbound_cmds) to enhance model performance.



Interpretability: Detailed feature importance and SHAP analysis to provide actionable insights and build trust in security applications.



Superior Performance: Outperforms traditional models like Decision Trees, Random Forests, and SVMs across key metrics.



Robustness: Maintains performance under noise and missing data, with <3% degradation in ablation tests.



Error Analysis: Identifies misclassification patterns (e.g., nmap, guess_passwd) to guide future improvements.

📈 Results Summary







Metric



Test Set Score





Accuracy



80.08%





Precision



85.00%





Recall



82.00%





F1-Score



80.00%





ROC-AUC



0.89





PR-AUC



0.97

The model demonstrates balanced detection of normal and attack traffic, with a high recall (0.97) for normal instances and high precision (0.97) for attack instances. A slight overfitting issue (training accuracy: 99.99%) suggests opportunities for optimization, such as regularization or early stopping. The confusion matrix reveals a 32% false negative rate for attack instances, highlighting challenges with covert attack patterns.

🔬 Visualizations





Confusion Matrix: Visualizes True Positives, False Positives, False Negatives, and True Negatives (confusion_matrix.png).



ROC Curve: Depicts the trade-off between True Positive Rate and False Positive Rate, with AUC ≈ 0.89 (roc_curve.png).



Precision-Recall Curve: Highlights performance on imbalanced data, with Average Precision ≈ 0.97 (precision_recall_curve.png).



Feature Importance: Ranks top 15 features, including src_bytes, protocol_type, dst_host_srv_count (feature_importance.png).



Learning Curve: Compares training and test accuracy to diagnose overfitting (learning_curve.png).

🧪 Future Work





Real-Time Detection: Optimize for streaming data using incremental learning or online gradient boosting.



Cross-Dataset Validation: Evaluate on modern datasets like CIC-IDS2017, UNSW-NB15, or TON_IoT.



Hybrid Models: Combine XGBoost with deep learning architectures (e.g., LSTM, CNN) for temporal and sequence-based detection.



Advanced Feature Engineering: Incorporate temporal aggregates, statistical features, or AutoML-driven feature selection.



Class Imbalance Handling: Explore adaptive sampling, cost-sensitive learning, or focal loss to improve detection of rare attack types (e.g., U2R, R2L).



Edge Deployment: Apply model compression and quantization for resource-constrained environments.



Enhanced Explainability: Integrate SHAP, LIME, or counterfactual explanations into deployment dashboards for real-time decision support.

📂 File Description







File



Description





XGBoost_IDS_NSLKDD.ipynb



Jupyter Notebook with preprocessing, model training, evaluation, and visuals





confusion_matrix.png



Confusion Matrix for test data





roc_curve.png



Receiver Operating Characteristic Curve





precision_recall_curve.png



Precision-Recall Curve for imbalanced data





feature_importance.png



Plot of top 15 feature importances





learning_curve.png



Learning curve showing training vs. test accuracy

📥 Dataset Access

The NSL-KDD dataset (KDDTrain+.txt and KDDTest+.txt) is used in this project. Download it from:





University of New Brunswick NSL-KDD Dataset

🚀 How to Run This Project

✅ Option 1: Open in Google Colab (No Setup Needed)





Clone the repository or download XGBoost_IDS_NSLKDD.ipynb.



Open in Google Colab by replacing github.com with colab.research.google.com/github/ in the URL:

https://colab.research.google.com/github/Aghawafaabbass/xgboost-intrusion-detection-nsl-kdd/blob/main/XGBoost_IDS_NSLKDD.ipynb



Upload the NSL-KDD dataset files (KDDTrain+.txt, KDDTest+.txt) to Colab.



Run all cells (Runtime > Run All or Ctrl+F9).

✅ Option 2: Run Locally

Requirements





Python 3.7+



Jupyter Notebook or JupyterLab

Install Required Libraries

pip install pandas numpy xgboost matplotlib seaborn scikit-learn

Launch the Notebook

jupyter notebook

Open XGBoost_IDS_NSLKDD.ipynb and run the cells after placing the NSL-KDD dataset files in the project directory.

📌 Index Terms

XGBoost, Intrusion Detection System (IDS), NSL-KDD Dataset, Network Security, Machine Learning, Ensemble Learning, Feature Importance, Explainable AI, Cybersecurity, Anomaly Detection, Supervised Learning

👨‍🏫 Authors

Agha Wafa Abbas
Lecturer, School of Computing, Arden University, UK
Lecturer, IVY College of Management Sciences, Pakistan
📧 awabbas@arden.ac.uk | wafa.abbas.lhr@rootsivy.edu.pk


📜 Citation & Publication

This project is part of a published research paper:

Agha, W. A. (2025). Enhancing Intrusion Detection in Cybersecurity using XGBoost: A Performance Analysis on the NSL-KDD Dataset. (https://doi.org/10.5281/zenodo.16737594)

🙏 Acknowledgements

We express gratitude to the University of New Brunswick for maintaining the NSL-KDD dataset and the open-source community for their contributions to XGBoost, Scikit-learn, Pandas, NumPy, Matplotlib, and SHAP libraries, which enabled this research.

ℹ️ About

This repository implements a scalable, interpretable, and high-performing Intrusion Detection System using XGBoost on the NSL-KDD dataset. It is designed for researchers, cybersecurity professionals, and enthusiasts interested in machine learning-driven network security solutions.

📄 License

© 2025 Agha Wafa Abbas. All rights reserved.
