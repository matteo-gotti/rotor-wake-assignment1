# BEM Analysis Project

## 📌 Project Overview
This project implements a **Blade Element Momentum (BEM) Method** for aerodynamic analysis. It includes functions for reading and processing airfoil data and calculating aerodynamic performance.

## 📂 Project Structure
```
BEM_Project/
│── src/                 # Source code folder
│   │── BEM_original.py  # Main script
│   │── BEM_functions.py # Helper functions
│
│── data/                # Data folder
│   │── DU95W180.csv     # Airfoil data
│
│── .venv/                # Virtual environment (not tracked by Git)
│── requirements.txt     # List of dependencies
│── README.md            # Project documentation
│── .gitignore           # Files to exclude from Git
```

## 🛠 Setup Instructions
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/BEM_Project.git
cd BEM_Project
```

### **2️⃣ Set Up the Virtual Environment**
#### **For Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

## 📊 How to Run the Code
To execute the main script:
```bash
python src/BEM_original.py
```

## 📄 Data Handling
The script reads airfoil data from `data/DU95W180.csv` using `pandas`. The path is managed dynamically to ensure cross-platform compatibility:
```python
import os
import pandas as pd

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "DU95W180.csv")
df = pd.read_csv(data_path)
```

## 👥 Collaboration Guidelines
- **Branching:** Use feature branches (`feature-branch-name`) for changes.
- **Commits:** Write clear commit messages (e.g., `fix: corrected BEM coefficient calculation`).
- **Pull Requests:** Submit PRs for review before merging.
- **Issues:** Report bugs or request features via GitHub Issues.
