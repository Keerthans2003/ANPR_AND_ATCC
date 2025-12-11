// Update confidence slider display
document.getElementById('confidence').addEventListener('input', (e) => {
    document.getElementById('confidence-value').textContent = e.target.value;
});

// Analyze button click handler
document.getElementById('analyze-btn').addEventListener('click', analyze);

// Refresh dashboard button
document.getElementById('refresh-dashboard-btn').addEventListener('click', refreshDashboard);

// Export buttons
document.getElementById('export-atcc-btn').addEventListener('click', () => exportData('atcc'));
document.getElementById('export-anpr-btn').addEventListener('click', () => exportData('anpr'));

// File input change handler
document.getElementById('file-input').addEventListener('change', (e) => {
    const fileName = e.target.files[0]?.name || 'No file selected';
    console.log('File selected:', fileName);
});

// Load dashboard on page load
window.addEventListener('load', refreshDashboard);

async function analyze() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    
    if (!file) {
        showStatus('Please select a file first', 'error');
        return;
    }
    
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const confidence = document.getElementById('confidence').value;
    
    // Validate file type
    const allowedAtccFormats = ['video/mp4', 'video/x-msvideo', 'video/quicktime'];
    const allowedAnprFormats = ['image/jpeg', 'image/png', 'image/jpg'];
    
    const isAtcc = mode === 'ATCC';
    const allowedFormats = isAtcc ? allowedAtccFormats : allowedAnprFormats;
    
    if (!allowedFormats.includes(file.type)) {
        showStatus(`Invalid file format for ${mode} mode`, 'error');
        return;
    }
    
    // Show loading
    showLoading(true);
    hideResults();
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('mode', mode);
        formData.append('confidence', confidence);
        
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Analysis failed');
        }
        
        const result = await response.json();
        
        if (result.success) {
            if (mode === 'ATCC') {
                displayAtccResults(result);
            } else {
                displayAnprResults(result);
            }
            showStatus('Analysis completed successfully!', 'success');
        } else {
            showStatus('Analysis failed', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showStatus('Error: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

function displayAtccResults(result) {
    const atccResults = document.getElementById('atcc-results');
    
    // Display frames
    const framesContainer = document.getElementById('frames-container');
    framesContainer.innerHTML = '';
    
    if (result.frames && result.frames.length > 0) {
        result.frames.forEach(frameData => {
            const img = document.createElement('img');
            img.src = frameData;
            img.alt = 'Processed frame';
            framesContainer.appendChild(img);
        });
    }
    
    // Display total vehicles
    document.getElementById('total-vehicles').textContent = result.total_vehicles || 0;
    
    // Display vehicle table
    const tbody = document.getElementById('vehicle-tbody');
    tbody.innerHTML = '';
    
    if (result.vehicle_counts && Object.keys(result.vehicle_counts).length > 0) {
        Object.entries(result.vehicle_counts).forEach(([vehicleType, count]) => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${vehicleType}</td><td>${count}</td>`;
            tbody.appendChild(row);
        });
    } else {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="2">No vehicles detected</td>';
        tbody.appendChild(row);
    }
    
    // Display chart
    if (result.chart) {
        document.getElementById('vehicle-chart').src = result.chart;
    }
    
    atccResults.style.display = 'block';
    document.getElementById('anpr-results').style.display = 'none';
    document.getElementById('results').style.display = 'block';
    document.getElementById('empty-state').style.display = 'none';
}

function displayAnprResults(result) {
    const anprResults = document.getElementById('anpr-results');
    
    // Display input image
    if (result.image) {
        document.getElementById('input-image').src = result.image;
    }
    
    // Display detected plates
    const platesContainer = document.getElementById('plates-container');
    platesContainer.innerHTML = '';
    
    if (result.detected_plates && result.detected_plates.length > 0) {
        result.detected_plates.forEach(plate => {
            const plateDiv = document.createElement('div');
            plateDiv.className = 'plate-item';
            plateDiv.textContent = plate;
            platesContainer.appendChild(plateDiv);
        });
    } else {
        const noPlatesDiv = document.createElement('div');
        noPlatesDiv.className = 'no-plates';
        noPlatesDiv.textContent = '⚠ No license plates detected';
        platesContainer.appendChild(noPlatesDiv);
    }
    
    anprResults.style.display = 'block';
    document.getElementById('atcc-results').style.display = 'none';
    document.getElementById('results').style.display = 'block';
    document.getElementById('empty-state').style.display = 'none';
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'flex' : 'none';
}

function hideResults() {
    document.getElementById('results').style.display = 'none';
}

function showStatus(message, type) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = `status ${type}`;
    
    if (type === 'success') {
        setTimeout(() => {
            statusEl.className = 'status';
        }, 5000);
    }
}

async function refreshDashboard() {
    try {
        const response = await fetch('/api/dashboard');
        const data = await response.json();
        
        // Update stats
        document.getElementById('total-vehicles-stat').textContent = data.atcc_total || 0;
        document.getElementById('total-plates-stat').textContent = data.anpr_total || 0;
        
        console.log('Dashboard updated:', data);
    } catch (error) {
        console.error('Error refreshing dashboard:', error);
    }
}

function exportData(mode) {
    const url = `/api/export/${mode}`;
    
    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error('Export failed');
            return response.blob();
        })
        .then(blob => {
            // Create download link
            const link = document.createElement('a');
            const filename = mode === 'atcc' ? 'atcc_results.csv' : 'anpr_results.csv';
            link.href = window.URL.createObjectURL(blob);
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            showStatus(`✅ ${filename} downloaded!`, 'success');
        })
        .catch(error => {
            console.error('Export error:', error);
            showStatus('Export failed', 'error');
        });
}
