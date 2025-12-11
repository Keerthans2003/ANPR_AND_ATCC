import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os
import time

# =============================
# PAGE CONFIG & THEME
# =============================
st.set_page_config(
    page_title="Smart Traffic AI", 
    page_icon="🚦",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    [data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Custom header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("# 🚦 Smart Traffic AI System")
    st.markdown("### Advanced Vehicle Detection & License Plate Recognition", unsafe_allow_html=True)

# =============================
# DATABASE & LOGGING SETUP
# =============================
DB_PATH = "../traffic.db"
CSV_LOGS_DIR = "../data/logs"

# Ensure directories exist
os.makedirs(CSV_LOGS_DIR, exist_ok=True)

def init_database():
    """Initialize SQLite database if it doesn't exist"""
    if not os.path.exists(DB_PATH):
        try:
            import sys
            sys.path.insert(0, '..')
            from db import init_db
            init_db.init_database()
        except:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS atcc (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                vehicle_type TEXT NOT NULL,
                count INTEGER NOT NULL,
                confidence REAL,
                video_file TEXT,
                processed_frames INTEGER
            )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS anpr (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                detected_plate TEXT NOT NULL,
                confidence REAL,
                image_file TEXT
            )''')
            conn.commit()
            conn.close()

def save_atcc_to_db(vehicle_counts, filename, confidence, frame_count):
    """Save ATCC results to database and CSV"""
    try:
        init_database()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        # Only save if vehicles were detected
        if len(vehicle_counts) > 0:
            for vehicle_type, count in vehicle_counts.items():
                cursor.execute('''INSERT INTO atcc (timestamp, vehicle_type, count, confidence, video_file, processed_frames)
                    VALUES (?, ?, ?, ?, ?, ?)''', (timestamp, vehicle_type, count, confidence, filename, frame_count))
            conn.commit()
            
            # Save to CSV
            csv_path = os.path.join(CSV_LOGS_DIR, 'atcc_results.csv')
            df_new = pd.DataFrame({
                'timestamp': [timestamp] * len(vehicle_counts),
                'vehicle_type': list(vehicle_counts.keys()),
                'count': list(vehicle_counts.values()),
                'confidence': [confidence] * len(vehicle_counts),
                'video_file': [filename] * len(vehicle_counts)
            })
            
            if os.path.exists(csv_path):
                try:
                    df_existing = pd.read_csv(csv_path)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(csv_path, index=False)
                except:
                    # If CSV is corrupted, overwrite it
                    df_new.to_csv(csv_path, index=False)
            else:
                df_new.to_csv(csv_path, index=False)
        
        conn.close()
        st.cache_data.clear()
    except Exception as e:
        st.warning(f"⚠️ Error saving ATCC logs: {e}")

def save_anpr_to_db(detected_plates, filename, confidence):
    """Save ANPR results to database and CSV"""
    try:
        init_database()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        # Only save if plates were detected
        if len(detected_plates) > 0:
            for plate in detected_plates:
                cursor.execute('''INSERT INTO anpr (timestamp, detected_plate, confidence, image_file)
                    VALUES (?, ?, ?, ?)''', (timestamp, plate, confidence, filename))
            conn.commit()
            
            # Save to CSV
            csv_path = os.path.join(CSV_LOGS_DIR, 'anpr_results.csv')
            df_new = pd.DataFrame({
                'timestamp': [timestamp] * len(detected_plates),
                'detected_plate': detected_plates,
                'confidence': [confidence] * len(detected_plates),
                'image_file': [filename] * len(detected_plates)
            })
            
            if os.path.exists(csv_path):
                try:
                    df_existing = pd.read_csv(csv_path)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(csv_path, index=False)
                except:
                    # If CSV is corrupted, overwrite it
                    df_new.to_csv(csv_path, index=False)
            else:
                df_new.to_csv(csv_path, index=False)
        
        conn.close()
        st.cache_data.clear()
    except Exception as e:
        st.warning(f"⚠️ Error saving ANPR logs: {e}")

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_models():
    """Load YOLO and OCR models with caching"""
    try:
        atcc_model = YOLO("../model/yolov8n.pt")
        anpr_model = YOLO("../model/yolov8n.pt")
        reader = easyocr.Reader(['en'])
        return atcc_model, anpr_model, reader
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

@st.cache_data
def get_dashboard_stats():
    """Get all dashboard statistics from database"""
    try:
        init_database()
        if not os.path.exists(DB_PATH):
            return {'vehicles': 0, 'plates': 0, 'analyses': 0, 'by_type': {}}
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total vehicles
        cursor.execute("SELECT COALESCE(SUM(count), 0) FROM atcc")
        total_vehicles = cursor.fetchone()[0]
        
        # Total plates
        cursor.execute("SELECT COUNT(*) FROM anpr")
        total_plates = cursor.fetchone()[0]
        
        # Total analyses
        cursor.execute("SELECT COUNT(DISTINCT video_file) FROM atcc")
        analyses = cursor.fetchone()[0]
        
        # Vehicles by type
        cursor.execute("SELECT vehicle_type, SUM(count) as total FROM atcc GROUP BY vehicle_type")
        by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        return {
            'vehicles': int(total_vehicles),
            'plates': int(total_plates),
            'analyses': int(analyses),
            'by_type': by_type
        }
    except:
        return {'vehicles': 0, 'plates': 0, 'analyses': 0, 'by_type': {}}

atcc_model, anpr_model, reader = load_models()

# =============================
# MAIN CONTENT TABS
# =============================
main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(["🎯 Analysis", "📊 Dashboard", "📈 History", "⚙️ Settings"])

# =============================
# TAB 1: ANALYSIS
# =============================
with main_tab1:
    col_left, col_right = st.columns([1, 3])
    
    with col_left:
        st.subheader("⚙️ Detection Settings")
        
        mode = st.radio(
            "Select Mode",
            ["ATCC – Vehicle Counting", "ANPR – License Plate Detection"],
            label_visibility="collapsed"
        )
        
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.05
        )
        
        st.metric("Current Setting", f"{confidence:.2f}")
        
        st.divider()
        
        uploaded_file = st.file_uploader(
            "📤 Upload File",
            type=["jpg", "png", "jpeg", "mp4", "avi", "mov"]
        )
        
        if uploaded_file:
            st.success(f"✅ File selected: {uploaded_file.name}")
        
        analyze = st.button("🚀 Start Analysis", key="analyze_btn", use_container_width=True)
    
    with col_right:
        if analyze and uploaded_file:
            if mode.startswith("ATCC"):
                st.subheader("🎥 Processing Video...")
                
                if atcc_model is None:
                    st.error("❌ Vehicle detection model not loaded. Please restart the app.")
                else:
                    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.tmp')
                    temp.write(uploaded_file.read())
                    temp.close()

                    try:
                        cap = cv2.VideoCapture(temp.name)
                        if not cap.isOpened():
                            st.error("❌ Error opening video file")
                        else:
                            stframe = st.empty()
                            vehicle_counts = Counter()
                            frame_count = 0
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            if total_frames == 0:
                                st.error("❌ Cannot read video frames")
                            else:
                                frames_processed = 0
                                
                                while True:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break

                                    frame_count += 1
                                    
                                    if frame_count % 5 == 0:
                                        try:
                                            results = atcc_model(frame, conf=confidence, verbose=False)[0]
                                            
                                            if results.boxes is not None and len(results.boxes) > 0:
                                                for box in results.boxes:
                                                    cls = int(box.cls)
                                                    if cls in results.names:
                                                        name = results.names[cls]
                                                        vehicle_counts[name] += 1
                                            
                                            frames_processed += 1
                                            annotated = results.plot()
                                            stframe.image(annotated, channels="BGR", use_column_width=True)
                                        except Exception as e:
                                            st.warning(f"⚠️ Frame processing error: {e}")
                                    
                                    progress = min(frame_count / total_frames, 1.0)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Processing frame {frame_count}/{total_frames} | Vehicles: {sum(vehicle_counts.values())}")

                        cap.release()
                    finally:
                        os.unlink(temp.name)

                    save_atcc_to_db(vehicle_counts, uploaded_file.name, confidence, frame_count)

                    st.divider()
                    total_vehicles = sum(vehicle_counts.values())
                    st.success(f"✅ Analysis Complete - {total_vehicles} vehicles detected")
                    
                    # Results section
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.subheader("📊 Vehicle Count")
                        if len(vehicle_counts) > 0:
                            df = pd.DataFrame(vehicle_counts.items(), columns=["Vehicle Type", "Count"])
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No vehicles detected in this video")
                    
                    with col_res2:
                        st.subheader("📈 Distribution")
                        if len(vehicle_counts) > 0:
                            fig, ax = plt.subplots(figsize=(6, 4))
                            labels = list(vehicle_counts.keys())
                            values = list(vehicle_counts.values())
                            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                            ax.bar(labels, values, color=colors)
                            ax.set_ylabel("Count", fontsize=11)
                            ax.set_xlabel("Vehicle Type", fontsize=11)
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
            
            else:  # ANPR Mode
                st.subheader("🔍 Processing Image...")
                
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)

                col_img1, col_img2 = st.columns(2)
                
                with col_img1:
                    st.image(img, channels="BGR", caption="Original Image", use_column_width=True)

                with col_img2:
                    if reader is not None:
                        st.info("🔄 Processing image for license plates...")
                        
                        detected_plates = []
                        
                        # Try multiple preprocessing techniques
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Technique 1: CLAHE + Bilateral Filter + Otsu
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        enhanced = clahe.apply(gray)
                        enhanced = cv2.bilateralFilter(enhanced, 11, 17, 17)
                        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        # Technique 2: Adaptive Thresholding
                        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                        
                        # Technique 3: Simple thresholding with morphological operations
                        _, thresh3 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel)
                        
                        # Run OCR on all preprocessed versions
                        try:
                            for i, thresh in enumerate([thresh1, thresh2, thresh3, gray], 1):
                                try:
                                    results = reader.readtext(thresh, detail=1)
                                    
                                    if results and len(results) > 0:
                                        for detection in results:
                                            if len(detection) >= 3:
                                                text = detection[1].strip()
                                                conf = detection[2]
                                                
                                                # Much lower confidence threshold - accept more detections
                                                if conf >= 0.1 and len(text) >= 2:
                                                    # Filter for plate-like patterns
                                                    if any(c.isalnum() for c in text):
                                                        detected_plates.append(text)
                                except:
                                    pass
                            
                            # Remove duplicates and very short matches
                            detected_plates = list(dict.fromkeys(detected_plates))
                            detected_plates = [p for p in detected_plates if len(p) >= 3]
                            
                            if len(detected_plates) > 0:
                                save_anpr_to_db(detected_plates, uploaded_file.name, confidence)
                                st.success(f"✅ {len(detected_plates)} plate(s) detected!")
                                
                                st.subheader("🔢 Detected Plates")
                                for i, plate in enumerate(detected_plates, 1):
                                    st.markdown(f"**{i}.** `{plate}`")
                            else:
                                st.warning("⚠️ No license plates detected. Try uploading a clearer image.")
                        except Exception as e:
                            st.error(f"OCR Error: {e}")
                    else:
                        st.error("❌ OCR model not loaded")
        else:
            st.info("👈 Select analysis mode and upload a file to begin")

# =============================
# TAB 2: DASHBOARD
# =============================
with main_tab2:
    stats = get_dashboard_stats()
    
    # KPI Metrics
    st.subheader("📊 Key Performance Indicators")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("🚗 Total Vehicles", stats['vehicles'])
    with metric_col2:
        st.metric("📋 License Plates", stats['plates'])
    with metric_col3:
        st.metric("📹 Analyses", stats['analyses'])
    with metric_col4:
        st.metric("🎯 Vehicle Types", len(stats['by_type']))
    
    st.divider()
    
    # Charts
    if stats['by_type']:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("📊 Vehicle Distribution (Bar)")
            vehicle_types = list(stats['by_type'].keys())
            vehicle_counts = list(stats['by_type'].values())
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = plt.cm.Spectral(np.linspace(0, 1, len(vehicle_types)))
            ax.bar(vehicle_types, vehicle_counts, color=colors)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_xlabel("Vehicle Type", fontsize=11)
            ax.set_title("Vehicles by Type", fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        with col_chart2:
            st.subheader("🥧 Vehicle Distribution (Pie)")
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = plt.cm.Spectral(np.linspace(0, 1, len(vehicle_types)))
            ax.pie(vehicle_counts, labels=vehicle_types, autopct="%1.1f%%", colors=colors, startangle=90)
            ax.set_title("Vehicles by Type", fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

# =============================
# TAB 3: HISTORY
# =============================
with main_tab3:
    hist_col1, hist_col2 = st.columns(2)
    
    with hist_col1:
        st.subheader("📹 ATCC Analysis History")
        try:
            csv_path = os.path.join(CSV_LOGS_DIR, 'atcc_results.csv')
            if os.path.exists(csv_path):
                df_atcc = pd.read_csv(csv_path)
                df_atcc['timestamp'] = pd.to_datetime(df_atcc['timestamp'])
                df_atcc = df_atcc.sort_values('timestamp', ascending=False).head(20)
                st.dataframe(df_atcc, use_container_width=True, hide_index=True)
                
                # Timeline chart
                if len(df_atcc) > 0:
                    st.subheader("📈 Vehicle Detections Over Time")
                    timeline = df_atcc.groupby(df_atcc['timestamp'].dt.date)['count'].sum()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(timeline.index, timeline.values, marker='o', linewidth=2, markersize=8, color='steelblue')
                    ax.set_xlabel("Date", fontsize=11)
                    ax.set_ylabel("Total Vehicles", fontsize=11)
                    ax.set_title("Daily Vehicle Count Trend", fontsize=12, fontweight='bold')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
            else:
                st.info("No ATCC history yet")
        except Exception as e:
            st.error(f"Error loading ATCC history: {e}")
    
    with hist_col2:
        st.subheader("🔢 ANPR Detection History")
        try:
            csv_path = os.path.join(CSV_LOGS_DIR, 'anpr_results.csv')
            if os.path.exists(csv_path):
                df_anpr = pd.read_csv(csv_path)
                df_anpr['timestamp'] = pd.to_datetime(df_anpr['timestamp'])
                df_anpr = df_anpr.sort_values('timestamp', ascending=False).head(20)
                st.dataframe(df_anpr, use_container_width=True, hide_index=True)
                
                # Most common plates
                if len(df_anpr) > 0:
                    st.subheader("🏆 Most Detected Plates")
                    top_plates = df_anpr['detected_plate'].value_counts().head(10)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.barh(top_plates.index, top_plates.values, color='coral')
                    ax.set_xlabel("Detection Count", fontsize=11)
                    ax.set_title("Top 10 Detected Plates", fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
            else:
                st.info("No ANPR history yet")
        except Exception as e:
            st.error(f"Error loading ANPR history: {e}")

# =============================
# TAB 4: SETTINGS
# =============================
with main_tab4:
    st.subheader("⚙️ System Settings & Export")
    
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        st.subheader("📥 Import / 📥 Export")
        
        export_type = st.selectbox("Select data to export", ["ATCC Results", "ANPR Results", "Both"])
        
        if st.button("📥 Generate Export", use_container_width=True):
            try:
                if export_type in ["ATCC Results", "Both"]:
                    csv_path = os.path.join(CSV_LOGS_DIR, 'atcc_results.csv')
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        st.download_button(
                            "📥 Download ATCC CSV",
                            df.to_csv(index=False),
                            "atcc_results.csv",
                            "text/csv"
                        )
                
                if export_type in ["ANPR Results", "Both"]:
                    csv_path = os.path.join(CSV_LOGS_DIR, 'anpr_results.csv')
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        st.download_button(
                            "📥 Download ANPR CSV",
                            df.to_csv(index=False),
                            "anpr_results.csv",
                            "text/csv"
                        )
            except Exception as e:
                st.error(f"Export error: {e}")
    
    with col_set2:
        st.subheader("🗄️ Database Info")
        
        try:
            init_database()
            if os.path.exists(DB_PATH):
                db_size = os.path.getsize(DB_PATH) / 1024  # KB
                st.metric("Database Size", f"{db_size:.2f} KB")
                
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM atcc")
                atcc_rows = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM anpr")
                anpr_rows = cursor.fetchone()[0]
                
                conn.close()
                
                st.metric("ATCC Records", atcc_rows)
                st.metric("ANPR Records", anpr_rows)
                
                if st.button("🔄 Clear Cache", use_container_width=True):
                    st.cache_data.clear()
                    st.success("✅ Cache cleared!")
        except:
            st.info("Database not initialized yet")
    
    st.divider()
    
    st.subheader("ℹ️ System Information")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.info(f"📂 Database: traffic.db")
    with col_info2:
        st.info(f"📊 Logs: data/logs/")
    with col_info3:
        st.info(f"🤖 Model: YOLOv8n")
