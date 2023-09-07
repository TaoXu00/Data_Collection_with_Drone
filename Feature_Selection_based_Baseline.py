import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import tsp
import math
import sklearn
class Feature_selection_based_baseline:
    '''
    Comparison approach 1:  Machine Learning based feature selection (permutation feature importance) + TSP + Machine Learning for inference
    '''
    def __init__(self, dir, drone, sensor_map, dataset, size_data_collection, number_of_training_data, num_of_estimation_data):
        self.dir=dir
        self.drone=drone
        self.sensor_map =  sensor_map
        self.dataset=dataset
        self.size_data_collection= size_data_collection
        self.number_of_training_data=number_of_training_data
        self.num_of_estimation_data = num_of_estimation_data

    def calculate_permutation_importance(self, X, y):
        # Create a GradientBoostingRegressor model
        gb_model = GradientBoostingRegressor()

        # Fit the model on the dataset
        gb_model.fit(X, y)

        # Calculate feature importance using permutation feature importance
        result = permutation_importance(gb_model, X, y, n_repeats=10, random_state=42)

        # # Print the feature importances
        # for i in result.importances_mean.argsort()[::-1]:
        #     print(f"{X.columns[i]:<8}"
        #           f"{result.importances_mean[i]:.3f}"
        #           f" +/- {result.importances_std[i]:.3f}")
        return result

    def calculate_feature_importance(self, Dataset):
        num_var=len(Dataset[0])
        train_data = pd.DataFrame(data=Dataset[0:1000, :])
        # Initialize a dictionary for the importance stores of each var
        importance_dict={}
        for var in np.arange(num_var):
            #print(f"calculating for var {var}:")
            # Create a regression dataset - column i is the prediction, the rest columns are the features
            Y_train=train_data.iloc[:, var]
            X_train=train_data.drop(train_data.columns[var], axis=1)
            result= self.calculate_permutation_importance(X_train, Y_train)
            # Print the feature importances
            for i in result.importances_mean.argsort()[::-1]:
                if X_train.columns[i] not in importance_dict:
                    importance_score_list=[result.importances_mean[i]]
                    importance_dict[X_train.columns[i]]=importance_score_list
                else:
                    importance_dict[X_train.columns[i]].append(result.importances_mean[i])
        # calculate the average of the importance score and rank them in desending order
        averages_importance_score = {key: sum(values) / len(values) for key, values in importance_dict.items()}
        sorted_vars = sorted(averages_importance_score, key=lambda k: averages_importance_score[k], reverse=True)
        return sorted_vars

    def greedy_search(self,ranked_sensors, drone, sensors_json, size_of_data_collection):

        """
                Select the sensors within the drone capability
                @param ranked_sensors: the raning of the sensor wrt their weights
                @param drone_capability: the drone capability (i.e., maximum travelling distance, energy capacity)
                @param sensors_json:  the 2D coordinates of the sensor maps
                @return: the selected sensors, trajectory and the optimal travelling distance
                """

        ## add the Depot point
        selected = []
        # loop until the bounds cross each other
        optimal_tour = []
        optimal_distance = 0
        optimal_energy_cost = 0
        add='True'
        while add == 'True':
            available_sensors = [sensor for sensor in ranked_sensors if sensor not in selected]
            add='False'
            for sensor in available_sensors:
                candidate = selected + [sensor]
                location_ids = list(map(str, candidate))
                location_ids.append('Depot')
                # take the coordinate from the json file of the candidate
                if len(location_ids) == 2:  # only one sensor is selected, so the drone will go from the depot to the sensor and then come back
                    n1 = (float(sensors_json[location_ids[0]]['Easting']), float(sensors_json[location_ids[0]]['Northing']))
                    n2 = (float(sensors_json[location_ids[1]]['Easting']), float(sensors_json[location_ids[1]]['Northing']))
                    tour = location_ids
                    dis = math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2) * 2
                    total_energy_cost = dis * drone.flying_energy_per_unit + (size_of_data_collection / drone.comm_rate)
                else:
                    coordinates = {}
                    coordinates['Depot'] = (
                    float(sensors_json['Depot']['Easting']), float(sensors_json['Depot']['Northing']))
                    for sensor in location_ids:
                        coordinates[sensor] = (
                        float(sensors_json[sensor]['Easting']), float(sensors_json[sensor]['Northing']))
                    my_tsp = tsp.tsp_solver(location_ids, coordinates)
                    tour, dis = my_tsp.solve()
                    # compute the energy cost, sum of the hovering energy and the flying energy
                    hovering_energy_cost = drone.hovering_energy_per_unit * (size_of_data_collection / drone.comm_rate) * (
                                len(candidate) - 1)
                    flying_energy_cost = drone.flying_energy_per_unit * dis
                    total_energy_cost = hovering_energy_cost + flying_energy_cost
                if total_energy_cost == drone.capacity:
                    optimal_energy_cost = total_energy_cost
                    return selected, tour, dis, total_energy_cost
                elif total_energy_cost < drone.capacity:
                    selected = candidate
                    add = 'True'
                    optimal_tour = tour
                    optimal_distance = dis
                    optimal_energy_cost = total_energy_cost
        return selected, optimal_tour, optimal_distance, optimal_energy_cost



    def binary_search(self, ranked_sensors, drone, sensors_json, size_of_data_collection):
        """
        Select the sensors within the drone capability
        @param ranked_sensors: the raning of the sensor wrt their weights
        @param drone_capability: the drone capability (i.e., maximum travelling distance, energy capacity)
        @param sensors_json:  the 2D coordinates of the sensor maps
        @return: the selected sensors, trajectory and the optimal travelling distance
        """
        # initialize the lower and upper bounds
        left = 0
        right = len(ranked_sensors)-1
        ## add the Depot point
        selected=[]
        # loop until the bounds cross each other
        optimal_tour=[]
        optimal_distance=0
        optimal_energy_cost=0
        while left <= right and len(selected)<len(ranked_sensors):
            mid= (left+right)//2
            bin= ranked_sensors[left:mid+1]
            candidate=selected+bin
            # calculate the middle index
            # check whether the tour for nodes in the left half > drone capability or not
            location_ids=list(map(str,candidate))
            location_ids.append('Depot')
            #take the coordinate from the json file of the candidate
            if len(location_ids)==2:  #only one sensor is selected, so the drone will go from the depot to the sensor and then come back
                n1=(float(sensors_json[location_ids[0]]['Easting']), float(sensors_json[location_ids[0]]['Northing']))
                n2=(float(sensors_json[location_ids[1]]['Easting']), float(sensors_json[location_ids[1]]['Northing']))
                tour = location_ids
                dis = math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)*2
                total_energy_cost= dis * drone.flying_energy_per_unit + (size_of_data_collection/drone.comm_rate)
            else:
                coordinates={}
                coordinates['Depot']=(float(sensors_json['Depot']['Easting']), float(sensors_json['Depot']['Northing']))
                for sensor in location_ids:
                    coordinates[sensor] = (float(sensors_json[sensor]['Easting']), float(sensors_json[sensor]['Northing']))
                my_tsp= tsp.tsp_solver(location_ids, coordinates)
                tour, dis= my_tsp.solve()
                #compute the energy cost, sum of the hovering energy and the flying energy
                hovering_energy_cost= drone.hovering_energy_per_unit * (size_of_data_collection/drone.comm_rate) * (len(candidate)-1)
                flying_energy_cost= drone.flying_energy_per_unit*dis
                total_energy_cost=hovering_energy_cost + flying_energy_cost
            if total_energy_cost == drone.capabiltiy:
                optimal_energy_cost= total_energy_cost
                return selected, tour, dis, total_energy_cost
            elif total_energy_cost < drone.capabiltiy:
                selected=candidate
                left=mid+1
                optimal_tour = tour
                optimal_distance = dis
                optimal_energy_cost=total_energy_cost
            else:
                right=mid-1
        return selected, optimal_tour, optimal_distance, optimal_energy_cost

    def train_model(self, vars_rank, drone):
        #selected_vars, optimal_tour, optimal_distance, energy_cost = self.binary_search(vars_rank, drone, self.sensor_map, self.size_data_collection)
        selected_vars, optimal_tour, optimal_distance, energy_cost = self.greedy_search(vars_rank, drone,
                                                                                        self.sensor_map,
                                                                                        self.size_data_collection)
        total_mse = 0
        if len(selected_vars) != 0:
            train_data = pd.DataFrame(data=self.dataset[0:self.number_of_training_data, :])
            X_train = train_data.iloc[:, selected_vars]
            Y_train = train_data.drop(selected_vars, axis=1)
            test_data = pd.DataFrame(data=self.dataset[self.number_of_training_data:(self.number_of_training_data+self.num_of_estimation_data), :])
            X_test = test_data.iloc[:, selected_vars]
            Y_test = test_data.drop(selected_vars, axis=1)
            for key in Y_train.columns.tolist():
                # Create a gradient boosting regressor
                #gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                gb_model = GradientBoostingRegressor()
                # Fit the model on the dataset
                gb_model.fit(X_train, Y_train[key])
                #gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                # Make predictions on the test set
                Y_pred = gb_model.predict(X_test)
                # check how the previoss
                mse = sklearn.metrics.mean_squared_error(Y_pred, Y_test[key])
                #mse = score * len(Y_train.columns.tolist())
                print("MSE Score for %s:" % (key))
                # averaged_score=sum(scores)/len(scores)
                print("MSE: %f" % (mse))
                total_mse += mse
        else:
            total_mse=math.inf
        return total_mse, selected_vars, optimal_tour, optimal_distance, energy_cost, vars_rank



