# Live Metrics Anomaly Detection System

## Overview

The Live Metrics Anomaly Detection System is a real-time monitoring dashboard developed using Python (Flask), HTML, CSS, and JavaScript. It monitors CPU, Memory, and Disk usage, detects anomalies using the Z-Score algorithm, forecasts future trends using Holt's Method, and provides machine learning insights using the Isolation Forest algorithm.

## Features

* Real-time CPU, Memory, and Disk monitoring
* Z-Score based anomaly detection
* Holt's Method for future forecasting
* Isolation Forest for AI-based anomaly analysis
* Interactive charts and dashboards
* Historical data storage using SQLite
* Adjustable anomaly threshold and forecast horizon

## Technologies Used

* Python
* Flask
* HTML
* CSS
* JavaScript
* Chart.js
* SQLite
* psutil
* NumPy
* Scikit-learn

## Live Preview

Render Deployment:

https://live-metrics-anomaly-detector.onrender.com

> Note: The dashboard displays live metrics only when the local Python backend is running.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/live-metrics-anomaly-detector.git
cd live-metrics-anomaly-detector
```

Install the required packages:

```bash
pip install flask flask-cors psutil numpy scikit-learn gunicorn
```

Run the backend server:

```bash
python live_metrics_server.py
```

Open **index.html** in your browser.

## Why is the Python server required?

This project uses the Python **psutil** library to collect live CPU, Memory, and Disk usage. Web browsers cannot directly access system hardware information for security reasons. Therefore, the Flask backend must be running locally to provide live metrics to the dashboard.

## Project Structure

```text
live-metrics-anomaly-detector/
│
├── index.html
├── live_metrics_server.py
├── requirements.txt
├── Procfile
└── README.md
```

## Author

**Aliana Begum**
**Nega Sri**
Autointelli Internship Project – 2026
