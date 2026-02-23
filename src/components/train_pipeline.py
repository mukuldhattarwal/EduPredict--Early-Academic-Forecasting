import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import save_object

def run_training_pipeline():
    # Step 1: Ingest data
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2: Transform data
    transformer = DataTransformation()
    train_arr, test_arr, preprocessor = transformer.initiate_data_transformation(train_path, test_path)

    # Step 3: Train model
    trainer = ModelTrainer()
    model = trainer.initiate_model_training(train_arr, test_arr)

    # Step 4: Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    save_object("artifacts/preprocessor.pkl", preprocessor)
    save_object("artifacts/model.pkl", model)

if __name__ == "__main__":
    run_training_pipeline()