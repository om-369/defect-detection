"""Dashboard for monitoring health check results."""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from flask import Flask, render_template, jsonify

app = Flask(__name__)

def load_health_data() -> Dict[str, Any]:
    """Load the latest health check data."""
    try:
        health_file = Path('monitoring/health_checks/latest.json')
        if not health_file.exists():
            return {
                'error': 'No health check data available',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        with open(health_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {
            'error': f'Error loading health data: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }

def load_historical_data(limit: int = 10) -> List[Dict[str, Any]]:
    """Load historical health check data."""
    try:
        health_dir = Path('monitoring/health_checks')
        if not health_dir.exists():
            return []
        
        # Get all health check files except latest.json
        files = [f for f in health_dir.glob('health_check_*.json')]
        files.sort(reverse=True)
        
        data = []
        for file in files[:limit]:
            with open(file, 'r') as f:
                data.append(json.load(f))
        return data
    except Exception:
        return []

@app.route('/')
def dashboard():
    """Render the main dashboard."""
    return render_template('dashboard.html')

@app.route('/api/health/current')
def current_health():
    """Get current health status."""
    return jsonify(load_health_data())

@app.route('/api/health/history')
def health_history():
    """Get historical health data."""
    return jsonify(load_historical_data())

if __name__ == '__main__':
    # Create monitoring directory if it doesn't exist
    Path('monitoring/health_checks').mkdir(parents=True, exist_ok=True)
    
    # Run the dashboard
    port = int(os.environ.get('DASHBOARD_PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
