import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import mySolution as Data_Correlation
import math
import sklearn
from sklearn.linear_model import LinearRegression
class Feature_Selection_Baseline2:
    '''
    This is another comparison approach. The idea is that using one sensor to predict others, and ranking them according to the incurred MSE.
    This is not currectly used.
    '''
    def __init__(self):
        Dir_plots = "plots/"
        # self.DG = DG.Data_generator()
        # self.adap_train = atrain.Adaptive_Training()
        # self.plotter = plotter.plotter(Dir_plots)
        # self.preprocess = preprocess.Data_Preprocess()
        # self.heu = heu.Heuristics()
        self.DC = Data_Correlation.Data_Correlation()

    def calculate_inference_error(self, X_train, Y_train, X_test, Y_test):
        total_mse=0
        for key in Y_train.columns.tolist():
            # Create a gradient boosting regressor
            # gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            gb_model = GradientBoostingRegressor()
            # Fit the model on the dataset
            gb_model.fit(X_train, Y_train[key])
            # gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            # Make predictions on the test set
            Y_pred = gb_model.predict(X_test)
            # check how the previoss
            score = sklearn.metrics.mean_squared_error(Y_pred, Y_test[key])
            mse = score * len(Y_train.columns.tolist())
            total_mse += mse
        return total_mse


    def rank_feature(self, Dataset):
        num_var=len(Dataset[0])
        train_data = pd.DataFrame(data=Dataset[0:1000, :])
        test_data= pd.DataFrame(data=Dataset[1000:1500, :])
        # Initialize a dictionary for the importance stores of each var
        total_mse_error_for_all_vars_dict={}
        for var in np.arange(num_var):
            #print(f"calculating for var {var}:")
            # Create a regression dataset - column i is the prediction, the rest columns are the features
            X_train=train_data.iloc[:, var]
            X_train=np.array([X_train]).transpose()
            Y_train=train_data.drop(train_data.columns[var], axis=1)
            X_test= test_data.iloc[:, var]
            X_test=np.array([X_test]).transpose()
            Y_test= test_data.drop(train_data.columns[var], axis=1)
            result= self.calculate_inference_error(X_train, Y_train, X_test, Y_test)
            total_mse_error_for_all_vars_dict[var]= result
        # calculate the average of the importance score and rank them in desending order
        sorted_vars = sorted(total_mse_error_for_all_vars_dict, key=lambda k: total_mse_error_for_all_vars_dict[k], reverse=True)
        return sorted_vars

    def train_model(self, drone_capability, sensors_map, Dataset, vars_rank):
        selected_vars, optimal_tour, optimal_distance = self.DC.binary_search(vars_rank, drone_capability, sensors_map)
        total_mse = 0
        if len(selected_vars) != 0:
            train_data = pd.DataFrame(data=Dataset[0:1000, :])
            X_train = train_data.iloc[:, selected_vars]
            Y_train = train_data.drop(selected_vars, axis=1)
            test_data = pd.DataFrame(data=Dataset[1000:1500, :])
            X_test = test_data.iloc[:, selected_vars]
            Y_test = test_data.drop(selected_vars, axis=1)
            for key in Y_train.columns.tolist():
                # Create a gradient boosting regressor
                #gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                #gb_model = GradientBoostingRegressor()
                gb_model=LinearRegression()
                # Fit the model on the dataset
                gb_model.fit(X_train, Y_train[key])
                #gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                # Make predictions on the test set
                Y_pred = gb_model.predict(X_test)
                # check how the previoss
                score = sklearn.metrics.mean_squared_error(Y_pred, Y_test[key])
                mse = score * len(Y_train.columns.tolist())
                print("MSE Score for %s:" % (key))
                # averaged_score=sum(scores)/len(scores)
                print("MSE: %f" % (mse))
                total_mse += mse
        else:
            total_mse=math.inf
        return total_mse, optimal_tour, optimal_distance