import sys
import os
from dataclasses import dataclass

import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
class DataTransformationConfig:
    preprocessor_file_path=os.path.join('artifacts',"preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",                      
                "race_ethnicity",               
                "parental_level_of_education",
                "lunch",                        
                "test_preparation_course"      
            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encode",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f'Categorical Columns:{categorical_columns}')
            logging.info(f'Numerical Columns: {numerical_columns}')

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
        
            logging.info("Read Train and test data both")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_columns=['writing_score','reading_score']


            X_train=train_df.drop(columns=[target_column_name])
            y_train=train_df[target_column_name]

            X_test=test_df.drop(columns=[target_column_name])
            y_test=test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training data and testing data"
            )
            X_train=preprocessor_obj.fit_transform(X_train)
            X_test=preprocessor_obj.transform(X_test)
            print('All ok')
            train_arr=np.c_[X_train,np.array(y_train)]
            test_arr=np.c_[X_test,np.array(y_test)]
            
            logging.info('Saved preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_file_path,
            )
        
        
        
        
        except Exception as e:
           raise CustomException(e,sys)
        

