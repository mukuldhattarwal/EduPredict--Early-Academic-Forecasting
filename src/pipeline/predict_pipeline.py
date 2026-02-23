import sys
import os
import pandas as pd
import json
import sklearn, numpy as np, traceback

try:
    print(f"DEBUG: scikit-learn={sklearn.__version__}, numpy={np.__version__}")
except Exception as e:
    print(f"DEBUG: scikit-learn={sklearn.__version__}, numpy={np.__version__}\n{e}")

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_all_models(self, features: pd.DataFrame):
        """Predict using all trained models and return their predictions"""
        try:
            preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            
            # Load all models from artifacts
            models_dir = "artifacts"
            predictions = []
            
            # Import all model types
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.linear_model import LinearRegression
            from xgboost import XGBRegressor
            from catboost import CatBoostRegressor
            
            # Try to load individual model files or use the best model
            model_files = {
                "Linear Regression": "linear_regression.pkl",
                "Decision Tree": "decision_tree.pkl",
                "Random Forest": "random_forest.pkl",
                "Gradient Boosting": "gradient_boosting.pkl",
                "XGBRegressor": "xgboost.pkl",
                "CatBoosting Regressor": "catboost.pkl",
                "AdaBoost Regressor": "adaboost.pkl"
            }
            
            for model_name, file_name in model_files.items():
                model_file_path = os.path.join(models_dir, file_name)
                if os.path.exists(model_file_path):
                    model = load_object(file_path=model_file_path)
                    pred = model.predict(data_scaled)
                    predictions.append({
                        "name": model_name,
                        "prediction": float(pred[0])
                    })
            
            return predictions

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)