import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print('After loading')

            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)