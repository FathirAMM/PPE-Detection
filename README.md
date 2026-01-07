# 🦺 PPE Detection System
This is a Streamlit-based web application for detecting Personal Protective Equipment (PPE) in videos using a custom YOLOE segmentation model. The system identifies whether people in a video are wearing safety gear such as helmets, gloves, vests, goggles, and shoes, and displays real-time analysis with annotated results.
---
## 🚀 Features
- Real-time detection of:
  - 🦺 Safety Vest
  - 🪖 Helmet
  - 🧤 Gloves
  - 🥽 Goggles
  - 🥾 Shoes
- Supports:
  - 📥 Video upload
  - 🎥 Sample video selection
- 🔍 Frame-by-frame PPE analysis
- 📊 Side-by-side comparison: Original vs. Detection
- ✅ Easy-to-use web interface powered by Streamlit
---
## 📦 Requirements
- Docker (recommended)  
  OR  
- Python 3.10+ with `pip` for manual setup
---
## 🐳 Run with Docker
### Step 1: Build the Docker Image
```bash
docker build -t ppe-detection .
```
### Step 2: Run the Container
```bash
docker run -p 8501:8501 ppe-detection
```
Then open your browser and go to: [http://localhost:8501](http://localhost:8501)
---
## 🧪 Manual Setup (Without Docker)
### 1. Clone the repository
```bash
git clone https://github.com/your-username/ppe-detection-system.git
cd ppe-detection-system
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the app
```bash
streamlit run app.py
```
---
## 📁 Project Structure
```
.
├── app.py               # Main Streamlit app
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
---
