# React Frontend for Student Performance Predictor

## How to Run

### 1. Start Flask Backend API
```bash
python app.py
```
The Flask API will run on `http://localhost:5000`

### 2. Open React Frontend
Simply open `frontend/index.html` in your web browser:
- Double-click the file, OR
- Right-click → Open with → Your browser

## Features

✅ **Modern React UI** - Built with React 18 (CDN-based, no build tools needed)
✅ **Real-time Predictions** - Calls Flask API for ML predictions
✅ **All Models Comparison** - Shows predictions from all 7 ML algorithms
✅ **Responsive Design** - Works on desktop, tablet, and mobile
✅ **Beautiful Animations** - Smooth transitions and effects

## API Endpoint

**POST** `/api/predict`

**Request Body:**
```json
{
    "gender": "male",
    "ethnicity": "group B",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 75,
    "writing_score": 80
}
```

**Response:**
```json
{
    "success": true,
    "prediction": 78.45,
    "best_model": "Linear Regression",
    "all_predictions": [
        {
            "name": "Linear Regression",
            "prediction": 78.45,
            "r2_score": 0.8804,
            "accuracy_percentage": 88.04
        },
        ...
    ]
}
```

## Tech Stack

- **Frontend**: React 18, Vanilla CSS
- **Backend**: Flask, Python ML Models
- **ML Models**: Linear Regression, Gradient Boosting, CatBoost, Random Forest, AdaBoost, XGBoost, Decision Tree
