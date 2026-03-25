# EduPredict: Early Academic Forecasting

EduPredict is a machine learning project designed to **predict student academic performance** using regression-based models. It provides an end-to-end pipeline including data preprocessing, exploratory data analysis (EDA), model training, and deployment via a Flask web application. The project is containerized with Docker for seamless deployment.

---

## 🚀 Features
- Multiple regression models for academic forecasting
- Flask web interface for real-time predictions
- Dockerized deployment for portability
- CI/CD workflows for automation
- Modular codebase with notebooks, source files, and templates

---

## 📂 Project Structure

```bash
Student-Score-Prediction/
│
├── app.py                # Flask application entry point
├── src/                  # Core ML source code
├── notebook/             # Jupyter notebooks for EDA & training
├── templates/            # HTML templates for Flask
├── static/
│   └── css/              # Styling for web app
├── frontend/             # Frontend components
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup file
├── Dockerfile            # Containerization setup
└── README.md             # Documentation
```

## ⚙️ Installation

### Clone the repository
```bash
git clone https://github.com/mukuldhattarwal/EduPredict--Early-Academic-Forecasting.git
cd EduPredict--Early-Academic-Forecasting
```
Install dependencies
pip install -r requirements.txt
Run locally
python app.py

## 🐳 Docker Deployment
```bashdocker build -t edupredict:latest .
docker run -p 5000:5000 edupredict:latest
```
## 📊 Models
- Linear Regression
- Ridge/Lasso Regression
- CatBoost
- Random Forest Regressor
Performance metrics are logged and compared during training.
## 📜 License
This project is licensed under the MIT License.

---

## ⚙️ setup.py (Draft)

```python
from setuptools import setup, find_packages

setup(
    name="edupredict",
    version="0.1.0",
    description="Early Academic Forecasting using ML models",
    author="mukuldhattarwal",
    author_email="mukuldhattarwal7@gmail.com",
    url="https://github.com/mukuldhattarwal/EduPredict--Early-Academic-Forecasting",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "flask",
        "numpy",
        "pandas",
        "scikit-learn",
        "catboost",
        "matplotlib",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

