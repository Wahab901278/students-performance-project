import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_ingestion import DataIngestion
from data_transformation import DataTransformation
from model_trainer import ModelTrainer
obj=DataIngestion()
train_data,test_data=obj.initiate_data_ingestion()

data_transformation=DataTransformation()
train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

modelTrainer=ModelTrainer()
r2score,best_model_name=modelTrainer.initiate_model_trainer(train_arr,test_arr) 

print(r2score)
print(best_model_name)