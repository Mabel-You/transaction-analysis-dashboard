# Transaction Dashboard

An interactive web dashboard for analyzing transaction data and detecting anomalies using machine learning.

## Features

- Interactive web interface for transaction data visualization
- Real-time statistics and numerical analysis
- Interactive charts and data visualizations
- Anomaly detection using Isolation Forest algorithm

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.50.0 | Web interface and interactive controls |
| pandas | 2.3.2 | Transaction data loading and processing |
| numpy | 2.3.3 | Numerical calculations and statistics |
| plotly | 6.3.0 | Interactive charts and visualizations |
| scikit-learn | 1.7.2 | Isolation Forest for anomaly detection |

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mabel-You/transaction-analysis-dashboard.git
   cd transaction-analysis-dashboard
   ```

2. **Create a Python virtual environment**

   **For Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   **For macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the dashboard**
   ```bash
   streamlit run dashboard.py
   ```

2. **Access the dashboard**
   - The dashboard will automatically open in your default web browser
   - If it doesn't open automatically, navigate to `http://localhost:8501`

3. **Stop the dashboard**
   - Press `Ctrl + C` in the terminal to stop the server

## Project Structure

```
.
├── dashboard.py                # Main dashboard application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── financial_transactions.csv  # Transaction dataset
```
