📌 Insurance Cost Prediction
📖 Project Overview

This project focuses on predicting insurance costs using machine learning techniques based on customer demographic and health-related features. The goal is to build a reliable regression model that can estimate insurance charges accurately and demonstrate an end-to-end data science workflow.

🎯 Problem Statement

Insurance companies need to estimate customer insurance costs efficiently. Using historical data, this project aims to predict insurance charges based on factors such as age, BMI, smoking status, and region.

📊 Dataset Description

The dataset contains the following features:

age – Age of the insured person
sex – Gender
bmi – Body Mass Index
children – Number of dependents
smoker – Smoking status
region – Residential area
charges – Insurance cost (target variable)


🛠️ Tech Stack

Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
Jupyter Notebook



🔍 Project Workflow
Exploratory Data Analysis (EDA)
Data understanding
Outlier and distribution analysis
Feature relationships
Data Preprocessing
Handling categorical variables
Feature scaling
Pipeline creation
Model Training & Evaluation
Trained multiple regression models
Compared model performance using evaluation metrics
Model Selection
Selected the best-performing model
Saved the trained model for future use


🤖 Machine Learning Models Used

Linear Regression
Decision Tree Regressor
Support Vector Regressor (SVR)
Ensemble-based models (if applicable)


📈 Model Evaluation Metrics

R² Score
Mean Absolute Error (MAE)
(Final model selected based on best overall performance)


📂 Project Structure
Insurance-Cost-Prediction/
│
├── data/
│   └── insurance.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
│
├── models/
│   └── best_insurance_model.pkl
│
├── src/
│   └── preprocessing.py
│
├── README.md
└── requirements.txt


▶️ How to Run This Project

Clone the repository:
git clone https://github.com/chetan-thoke/Insurance-Cost-Prediction.git


Navigate to the project directory:
cd Insurance-Cost-Prediction


Install dependencies:
pip install -r requirements.txt

Run the Streamlit app
streamlit run app.py

Run notebooks:
EDA.ipynb for analysis
Model_Training.ipynb for training



🚀 Future Improvements

Hyperparameter tuning & cross-validation
Better feature engineering
Deployment enhancements with Docker or cloud hosting
Add CI/CD workflows

👤 Author

Chetan Thoke
Aspiring Data Scientist
🔗 GitHub: https://github.com/chetan-thoke

