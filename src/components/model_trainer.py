import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and testing input and output data")
            X_train=train_array[:,:-1]
            X_test=test_array[:,:-1]
            y_train=train_array[:,-1]
            y_test=test_array[:,-1]

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                # "Adaboost Regressor": AdaBoostRegressor(),
            }

            params={
                "Decision Tree":{
                    'criterion':['squared_error', 'poisson', 'friedman_mse', 'absolute_error'],
                    'splitter':["best","random"],
                    'max_features':['sqrt','log2']
                },
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]
                },
                # "AdaBoost Regressor":{
                #     "learning_rate":[0.1,0.01,0.5,0.1],
                #     "iterations":[30,50,100]
                # },
                "KNeighborsRegressor":{
                    "n_neighbors":[3,5,7,9,11,13,15,17,19,21,25]

                }
            }

            model_report=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing datset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model

            )
            y_pred=best_model.predict(X_test)

            r2score=r2_score(y_test,y_pred)
            return r2score,best_model_name
        
        except Exception as e:
            raise CustomException(e,sys)