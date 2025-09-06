# 🛒 Future of Retail - BigMart Sales Chatbot

This project explores the **future of retail** using **machine learning** and a **chatbot interface**.  
It predicts store sales and allows users to interact with the system through a chatbot for insights.

---

## 📂 Project Structure
- `app.py` → Main FastAPI application  
- `chatbot.py` → Chatbot implementation for store queries  
- `model.py` → Training & prediction logic (XGBoost model)  
- `utils.py` → Utility functions  
- `generate_eda_report.py` → Generates automated exploratory data analysis (EDA) reports  
- `model.pkl` → Pre-trained machine learning model  
- `bigmart_data.csv` → Dataset used for analysis  
- `rossmann_store.csv` & `rossmann_train.csv` → Additional datasets  
- `templates/` → HTML templates for web UI  
- `run_project.bat` → Script to quickly start the project on Windows  

---

## ⚙️ Features
- ✅ Sales prediction using **XGBoost Regression**  
- ✅ Interactive **chatbot interface** for store insights  
- ✅ Automated EDA report generation  
- ✅ Data preprocessing and visualization tools  
- ✅ FastAPI-powered backend with simple web interface  

---

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib)  
- **FastAPI** (backend)  
- **HTML, CSS, JS** (frontend via `templates/`)  
- **Git & GitHub**

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Chandhana2004/future-of-retail.git
   cd future-of-retail
2. Install dependencies:

pip install -r requirements.txt


(If requirements.txt is missing, manually install: fastapi uvicorn pandas numpy scikit-learn xgboost matplotlib)

3. Run the app:

uvicorn app:app --reload


Or simply double-click run_project.bat (on Windows).

4. Open in browser:
👉 http://127.0.0.1:8000

📊 Visualizations

Store sales trends

Product category analysis

Prediction vs Actual sales

📌 Future Improvements

Deploy on cloud (Heroku / AWS)

Improve UI/UX with modern frontend
