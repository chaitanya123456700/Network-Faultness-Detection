NOTE: No need to run model again in this project folder or repo. in this folder itself the trained model is saved as model.pkl so no need to do any preprocessing or download the dataset just execute requirements.txt the execute dashboard(dash.html) with the command given below. follow the below commands given  for enabling venv and execute the project and after opening the dashboard in the chrome or any webserver first upload the test data which was in the dashboard/text_exported.csv which was given in the project folder itself then analyse the results 

Demo video Drive_Link : https://drive.google.com/drive/folders/1z30m0QmLVU_KZEn-Rs2WBDy9sKZMOW1D?usp=sharing

# Network Fault Detection using Graph Attention Networks (GAT)

A real-time network intrusion detection system using Graph Attention Networks on the NSL-KDD dataset, with an interactive web dashboard for visualization.

---

## 🎯 Project Overview

This project implements a **Graph Attention Network (GAT)** for detecting network intrusions and faults in real-time. It classifies network traffic into 5 categories:

- **Normal** — Legitimate traffic
- **DoS** (Denial of Service) — Flooding attacks
- **Probe** — Surveillance and reconnaissance
- **R2L** (Remote to Local) — Unauthorized access from remote
- **U2R** (User to Root) — Privilege escalation attacks

---

## ✨ Key Features

- 2-layer GAT with residual connections and Focal Loss
- SMOTE oversampling for class balancing
- Interactive real-time dashboard with force-directed graph rendering
- AI-powered threat analysis via Claude API integration

---

## 📁 Project Structure

```
NETWORK_FAULTNESS_DETECTION_CODE/
├── dashboard/
│   ├── dash.html               # Interactive web dashboard
│   └── test_export.csv         # Sample data
├── data/
│   ├── KDDTrain+.txt          # Training dataset
│   ├── KDDTest+.txt           # Test dataset
│   ├── text_data.csv          # Pre-processed data (load this in dashboard)
│   └── ...                    # Other processed files
├── models/
│   └── gat_model.py           # GAT architecture
├── outputs/
│   └── best_model.pth         # Pre-trained model weights
├── train.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/NETWORK_FAULTNESS_DETECTION_CODE.git
cd NETWORK_FAULTNESS_DETECTION_CODE
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard

```bash
cd dashboard
python -m http.server 8000
```

Then open your browser and go to: **http://localhost:8000/dash.html**

### 5. Load the Data

Once the dashboard is open, use the **CSV Upload** button and load the file from:

```
 dashboard/text_export.csv
```

That's it — no training required. The pre-trained model and data are already included in the repository.

---

## 🖥️ Dashboard Features

- **Real-time Graph Visualization** — Force-directed kNN graph rendering
- **Live Metrics** — Traffic classification counts and percentages
- **Threat Timeline** — 30-second rolling window of attack detection
- **Class Distribution** — Donut chart of traffic types
- **Confidence Scores** — Histogram of model confidence
- **AI Analyst** — Claude-powered threat analysis chatbot
- **CSV Upload** — Load your own network data

---

## 📚 Dataset

This project uses the **NSL-KDD Dataset**, an improved version of KDD'99:

- Training set: 125,973 records
- Test set: 22,544 records
- Features: 41 (38 used after selection)
- Classes: 5 (Normal + 4 attack types)

---

## 🙏 Acknowledgments

- NSL-KDD dataset creators
- PyTorch Geometric team
- Graph Attention Networks — Veličković et al., 2018
- Anthropic Claude API
