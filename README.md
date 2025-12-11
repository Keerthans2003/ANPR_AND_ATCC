# 🚦 Smart Traffic AI System – ANPR & ATCC 

An end-to-end **Smart Traffic Monitoring System** that combines:

- **ANPR (Automatic Number Plate Recognition)** from vehicle images
- **ATCC (Automatic Traffic & Vehicle Counting)** from traffic videos   

This project uses two publicly available datasets:

- **Large License Plate Dataset** for ANPR — https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset
- **Traffic Video Dataset** for ATCC — https://www.kaggle.com/datasets/arshadrahmanziban/traffic-video-dataset   

It follows a real-world AI deployment workflow:  
**Training & heavy processing in Google Colab (GPU)** → **Deployment & visualization in VS Code using Streamlit**

---

## ✨ Key Features

- 📹 Real-time vehicle detection from video  
- 🔢 License plate text detection using OCR  
- 🎛 Adjustable detection confidence threshold  
- 📊 Interactive analytics dashboard:
  - Vehicle count table  
  - Bar chart  
  - Pie chart  
  - Line plot  
  - Scatter plot  
- 📤 Support for uploading videos and images locally  
- 💾 SQLite database ready for logs  
- 🧪 Fully modular Colab → VS Code pipeline  

---

## 🧠 Models & Technologies

Detection & OCR Models

    - Vehicle Detection: YOLOv8n(via Ultralytics)  
    - License-Plate Detection (for ANPR): YOLOv8 / custom plate model
    - OCR: EasyOCR

Tech Stack

    -Python · Streamlit · OpenCV · Ultralytics YOLOv8 · EasyOCR · Pandas · Matplotlib · SQLite · Google Colab · VS Code  

---

## ☁️ Dataset & Processing Workflow

1. **ATCC** — Videos from *Traffic Video Dataset* processed frame-by-frame with YOLOv8 → Vehicle counts exported to CSV  
2. **ANPR** — Images from *Large License Plate Dataset* processed with YOLO + EasyOCR → Plate text saved to CSV  
3. Exported CSVs are used in VS Code for dashboard and analytics.

---

    
## 📁 Project Structure (VS Code)
   
    ANPR_AND_ATCC
    │
    ├── data/
    │   └── logs/
    │       ├── atcc_results.csv     ✅ ATCC output
    │       └── anpr_results.csv     ✅ ANPR output
    │
    ├── db/
    │   └── init_db.py               ✅ Loads both CSVs into ONE database
    │
    ├── streamlit_app/
    │   └── app.py                   ✅ Combined dashboard (2 tabs)
    │
    ├── traffic.db                   ✅ Auto-created database
    └── requirements.txt



# ⚙️ Steps to Run This Project in VS Code

## Step 1: Go to your project folder
    cd ANPR_AND_ATCC

## Step 2: Create virtual environment
    python -m venv venv

## Step 3: Activate virtual environment (PowerShell)
    .\venv\Scripts\activate  

## Step 4: Install all required packages
    pip install -r requirements.txt

## Step 5: Create database
    python db/init_db.py

# Step 6: Run the Streamlit dashboard
    python -m streamlit run streamlit_app/app.py

## Step 7: Open this in your browser
    http://localhost:8501
