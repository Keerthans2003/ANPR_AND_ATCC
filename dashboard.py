from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import tempfile
import os
import sqlite3
from datetime import datetime
from ultralytics import YOLO
import easyocr
from collections import Counter
import base64
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
import matplotlib
import csv
import pandas as pd
matplotlib.use('Agg')

app = Flask(__name__, template_folder='.', static_folder='static')

# Database path
DB_PATH = "../traffic.db"
CSV_LOGS_DIR = "../data/logs"

# Ensure CSV logs directory exists
os.makedirs(CSV_LOGS_DIR, exist_ok=True)

# Load models
print("Loading YOLO models...")
try:
    atcc_model = YOLO("../model/yolov8n.pt")  # vehicle detector
    anpr_model = YOLO("../model/yolov8n.pt")  # plate detector (demo)
    reader = easyocr.Reader(['en'])
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"⚠️ Model loading warning: {e}")
    atcc_model = None
    anpr_model = None
    reader = None

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        mode = request.form.get('mode', 'ATCC')
        confidence = float(request.form.get('confidence', 0.4))
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        
        if mode == 'ATCC':
            # Video processing
            temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp.write(file_bytes)
            temp.close()
            
            result = process_video(temp.name, file.filename, confidence)
            os.unlink(temp.name)
            return jsonify(result)
        else:
            # Image processing for ANPR
            img = cv2.imdecode(file_bytes, 1)
            result = process_image(img, file.filename, confidence)
            return jsonify(result)
    
    except Exception as e:
        print(f"Error in analyze: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get aggregated data from database for dashboard"""
    try:
        if not os.path.exists(DB_PATH):
            return jsonify({
                'atcc_total': 0,
                'anpr_total': 0,
                'vehicle_breakdown': {},
                'recent_plates': []
            })
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get ATCC statistics
        cursor.execute("SELECT vehicle_type, SUM(count) as total FROM atcc GROUP BY vehicle_type")
        atcc_data = {row[0]: row[1] for row in cursor.fetchall()}
        atcc_total = sum(atcc_data.values())
        
        # Get ANPR statistics
        cursor.execute("SELECT COUNT(*) FROM anpr")
        anpr_total = cursor.fetchone()[0]
        
        cursor.execute("SELECT detected_plate FROM anpr ORDER BY timestamp DESC LIMIT 10")
        recent_plates = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'atcc_total': atcc_total,
            'anpr_total': anpr_total,
            'vehicle_breakdown': atcc_data,
            'recent_plates': recent_plates
        })
    except Exception as e:
        print(f"Error getting dashboard data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<mode>', methods=['GET'])
def export_csv(mode):
    """Export results as CSV"""
    try:
        if not os.path.exists(DB_PATH):
            return jsonify({'error': 'Database not found'}), 404
        
        conn = sqlite3.connect(DB_PATH)
        
        if mode == 'atcc':
            df = pd.read_sql_query("SELECT * FROM atcc", conn)
            filename = 'atcc_results.csv'
        elif mode == 'anpr':
            df = pd.read_sql_query("SELECT * FROM anpr", conn)
            filename = 'anpr_results.csv'
        else:
            return jsonify({'error': 'Invalid mode'}), 400
        
        conn.close()
        
        if df.empty:
            return jsonify({'error': 'No data to export'}), 404
        
        # Create CSV in memory
        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return output.getvalue(), 200, {
            'Content-Disposition': f'attachment; filename={filename}',
            'Content-Type': 'text/csv'
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video(video_path, filename, confidence):
    """Process video for ATCC (vehicle counting)"""
    if atcc_model is None:
        return {'error': 'Model not loaded', 'success': False}
    
    cap = cv2.VideoCapture(video_path)
    vehicle_counts = Counter()
    frame_count = 0
    processed_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame for speed
            continue
        
        # Resize for processing
        frame_small = cv2.resize(frame, (640, 480))
        results = atcc_model(frame_small, conf=confidence)[0]
        
        for box in results.boxes:
            cls = int(box.cls)
            name = results.names[cls]
            vehicle_counts[name] += 1
        
        annotated = results.plot()
        
        # Convert to base64 for display (only store a few frames)
        if len(processed_frames) < 5:
            _, buffer = cv2.imencode('.jpg', annotated)
            frame_b64 = base64.b64encode(buffer).decode()
            processed_frames.append(f"data:image/jpeg;base64,{frame_b64}")
    
    cap.release()
    
    # Save to database and CSV
    save_atcc_to_db(vehicle_counts, filename, confidence, frame_count)
    
    # Generate chart
    labels = list(vehicle_counts.keys())
    values = list(vehicle_counts.values())
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color='steelblue')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Vehicle Type', fontsize=12)
    ax.set_title('Vehicle Count Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    
    return {
        'success': True,
        'mode': 'ATCC',
        'vehicle_counts': dict(vehicle_counts),
        'total_vehicles': sum(vehicle_counts.values()),
        'frames': processed_frames,
        'chart': f"data:image/png;base64,{chart_b64}"
    }

def process_image(img, filename, confidence):
    """Process image for ANPR (license plate detection)"""
    # Convert to base64 for display
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode()
    
    # Run OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    results = reader.readtext(thresh) if reader else []
    detected_plates = [res[-2] for res in results] if results else []
    
    # Save to database and CSV
    save_anpr_to_db(detected_plates, filename, confidence)
    
    return {
        'success': True,
        'mode': 'ANPR',
        'image': f"data:image/jpeg;base64,{img_b64}",
        'detected_plates': detected_plates,
        'plate_count': len(detected_plates)
    }

def save_atcc_to_db(vehicle_counts, filename, confidence, frame_count):
    """Save ATCC results to database and CSV"""
    try:
        # Ensure database exists
        if not os.path.exists(DB_PATH):
            try:
                import sys
                sys.path.insert(0, '..')
                from db import init_db
                init_db.init_database()
            except:
                pass
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for vehicle_type, count in vehicle_counts.items():
            cursor.execute('''
                INSERT INTO atcc (timestamp, vehicle_type, count, confidence, video_file, processed_frames)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, vehicle_type, count, confidence, filename, frame_count))
        
        conn.commit()
        conn.close()
        
        # Also save to CSV
        csv_path = os.path.join(CSV_LOGS_DIR, 'atcc_results.csv')
        df_new = pd.DataFrame({
            'timestamp': [timestamp] * len(vehicle_counts),
            'vehicle_type': list(vehicle_counts.keys()),
            'count': list(vehicle_counts.values()),
            'confidence': [confidence] * len(vehicle_counts),
            'video_file': [filename] * len(vehicle_counts)
        })
        
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
        else:
            df_new.to_csv(csv_path, index=False)
        
        print(f"✅ ATCC results saved for {filename}")
    except Exception as e:
        print(f"⚠️ Error saving ATCC to DB: {e}")

def save_anpr_to_db(detected_plates, filename, confidence):
    """Save ANPR results to database and CSV"""
    try:
        # Ensure database exists
        if not os.path.exists(DB_PATH):
            try:
                import sys
                sys.path.insert(0, '..')
                from db import init_db
                init_db.init_database()
            except:
                pass
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for plate in detected_plates:
            cursor.execute('''
                INSERT INTO anpr (timestamp, detected_plate, confidence, image_file)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, plate, confidence, filename))
        
        conn.commit()
        conn.close()
        
        # Also save to CSV
        csv_path = os.path.join(CSV_LOGS_DIR, 'anpr_results.csv')
        df_new = pd.DataFrame({
            'timestamp': [timestamp] * len(detected_plates),
            'detected_plate': detected_plates,
            'confidence': [confidence] * len(detected_plates),
            'image_file': [filename] * len(detected_plates)
        })
        
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
        else:
            df_new.to_csv(csv_path, index=False)
        
        print(f"✅ ANPR results saved for {filename}")
    except Exception as e:
        print(f"⚠️ Error saving ANPR to DB: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
