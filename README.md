# 📊 Customer Churn Prediction – Streamlit App V1.5

This repository contains a **Streamlit web application** for predicting customer churn using a **Random Forest model** trained on engineered customer behavioral, demographic, and interaction features.

The project includes:
- A full ML pipeline (preprocessing → feature engineering → training → evaluation → inference)
- A deployed-ready Streamlit application
- Saved model artifacts for inference
- Modularized notebooks for transparency and reproducibility

---

## 🚀 Demo

The app predicts whether a customer is **likely to churn or stay**, based on input features such as:
- Age
- Gender
- Tenure
- Subscription Type
- Contract Length
- Usage Frequency
- Support Calls
- Payment Behavior  
…and several engineered features.

---

## 📁 Repository Structure

customer-churn/
│
├── app.py # Streamlit application
├── requirements.txt # Package dependencies
├── .gitignore # Files ignored in GitHub
│
├── model/ # Model artifacts (safe for repo)
│ ├── rf_model.pkl
│ ├── scaler.pkl
│ ├── label_encoders.pkl
│ ├── feature_names.pkl
│
├── notebooks/ # Jupyter Notebooks (NOT needed for deployment)
│ ├── 01_data_load_and_basic_eda.ipynb
│ ├── 02_eda_analysis.ipynb
│ ├── 03_data_preprocessing_and_feature_engineering.ipynb
│ ├── 04_model_training.ipynb
│ ├── 05_model_evaluation.ipynb
│ ├── 06_model_export_and_inference.ipynb
│
├── data/ # Example input data (DO NOT upload large CSVs)
│ ├── sample_input.json
│
└── README.md


---

## 🧠 Machine Learning Workflow

### **1. Data Preprocessing**
- Missing value handling  
- Dropping unused identifiers  
- Encoding categorical features  
- One-hot encoding of subscription & contract types  
- Feature engineering:
  - Average Monthly Spend  
  - Support Intensity  
  - Recency/Tenure Ratio  
  - Age Group segmentation  

### **2. Balancing**
SMOTE is used to handle class imbalance.

### **3. Scaling**
`StandardScaler` is applied to numeric fields.

### **4. Model**
The chosen model is:
- **Random Forest Classifier**  
  - Tuned hyperparameters  
  - Balanced class weighting  

### **5. Saved Artifacts**
The following files support inference:

rf_model.pkl # The trained model
scaler.pkl # Feature scaler
label_encoders.pkl # Encoders (if any)
feature_names.pkl # Ordered list of columns used during training


---

## 🖥 Running the App Locally

### **1. Install dependencies**
pip install -r requirements.txt


### **2. Run Streamlit**
streamlit run app.py


### **3. Open browser**
[
](http://localhost:8501)

---

## ☁️ Deploy to Streamlit Cloud (Recommended)

1. Push this repository to **GitHub**  
2. Go to: https://share.streamlit.io  
3. Click **Deploy an app**  
4. Select your repo  
5. Set **Main file** → `app.py`  
6. Deploy 🎉  

Streamlit Cloud will:
- Install dependencies from `requirements.txt`
- Load model artifacts from `/model`
- Launch the app automatically

---

## 📈 Inputs and Prediction Output

### Input fields include:
- Age  
- Gender  
- Tenure  
- Usage Frequency  
- Support Calls  
- Payment Delay  
- Total Spend  
- Last Interaction  
- Subscription Type  
- Contract Length  

### The model returns:
- **Churn Probability** (0.00 – 1.00)  
- **Final Prediction** (Likely to churn / Not likely to churn)

---

## 🛡 Notes & Best Practices

- Training datasets (**train.csv**, **train_processed.csv**) are intentionally **NOT included** in the repo due to size limits and privacy.
- Do **NOT** upload large CSV files to GitHub.
- The app uses only `.pkl` model artifacts.

---

## 🧩 Future Enhancements

- Add SHAP explainability  
- Add batch prediction via CSV upload  
- Add authentication  
- Convert pipeline into a single `sklearn.Pipeline` object  
- Deploy using Docker or HuggingFace Spaces  

---

## 👨‍💻 Author

This project was auto-generated and improved by **ChatGPT AI** based on user specifications.

---

## ⭐ Support

If you like this project, please ⭐ star the repository on GitHub!

