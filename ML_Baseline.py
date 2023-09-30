import math
import Data_Generator_and_Loader as DG
import Adaptive_Training as atrain
import Data_Preprocess as preprocess
import plotter
import Heuristics as heu
import mySolution
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import tsp
class ML_Baseline:
    '''
    This class is to test various machine learning baseline models, and select a good one to use for the comparison approach.
    '''
    def __init__(self, dir_results, drone, sensor_map, dataset, size_of_data_collection, num_of_data_for_training, num_of_data_for_inference):
        self.dir_results=dir_results
        self.adap_train=atrain.Adaptive_Training()
        self.preprocess=preprocess.Data_Preprocess()
        self.heu=heu.Heuristics()
        self.drone= drone
        self.sensor_map = sensor_map
        self.dataset= dataset
        self.size_of_data_collection= size_of_data_collection
        self.num_of_data_for_training= num_of_data_for_training
        self.num_of_data_for_inference= num_of_data_for_inference

    def calculate_weight_acoording_to_energy(self,  selected_vars, availbale_vars,sensor_json, drone, size_of_collect_data):
        weights_dict = {}
        for j in range(len(availbale_vars)):  # observed var
            # compute the distance cost
            weight_energy = math.inf
            for var in selected_vars:
                coor_x = sensor_json[str(var)]['Easting']
                coor_y = sensor_json[str(var)]['Northing']
                current_var = availbale_vars[j]
                dist = np.sqrt((sensor_json[str(current_var)]['Easting'] - coor_x) ** 2 + (
                            sensor_json[str(current_var)]['Northing'] - coor_y) ** 2)
                energy_cost=dist* drone.flying_energy_per_unit +  (size_of_collect_data/drone.comm_rate)*drone.hovering_energy_per_unit
                if weight_energy > energy_cost:
                         weight_energy =energy_cost
            weights_dict[j] = 1 / weight_energy
        return weights_dict

    def rank_sensor_with_energy(self, sensor_map, drone, size_of_collect_data) -> object:
        selected_vars = ['Depot']
        num_sensors= len(self.sensor_map)-1
        available_vars = np.arange(num_sensors)
        w = num_sensors  ##select all the variables
        while w != 0:
            #todo use tsp to calculate the incurred additional total distance

            weights_dict = self.calculate_weight_acoording_to_energy(selected_vars, available_vars,sensor_map, drone, size_of_collect_data)
            sorted_d = dict(sorted(weights_dict.items(), key=lambda x: x[1], reverse=True))
            top_one = list(sorted_d.keys())[0]
            top_one_var = available_vars[top_one]
            selected_vars.append(top_one_var)
            available_vars = np.delete(available_vars, np.where(available_vars == top_one_var))
            # update covariance matrix
            w = w - 1
        return selected_vars

    def calculate_weight_according_to_distance(self, selected_vars, availbale_vars,sensor_json):
        weights_dict = {}
        for j in range(len(availbale_vars)):  # observed var
            # compute the distance cost
            weight_dis = math.inf
            for var in selected_vars:
                coor_x = sensor_json[str(var)]['Easting']
                coor_y = sensor_json[str(var)]['Northing']
                current_var = availbale_vars[j]
                dist = np.sqrt((sensor_json[str(current_var)]['Easting'] - coor_x) ** 2 + (
                            sensor_json[str(current_var)]['Northing'] - coor_y) ** 2)
                if weight_dis > dist:
                         weight_dis = dist
            weights_dict[j] = 1 / weight_dis
        return weights_dict

    def rank_sensor_with_distance(self, num_sensors, sensor_map):
        selected_vars = ['Depot']
        available_vars = np.arange(num_sensors)
        w=num_sensors ##select all the variables
        while w != 0:
            weights_dict = self.calculate_weight_according_to_distance(selected_vars, available_vars, sensor_map) #the parameter here is different
            sorted_d = dict(sorted(weights_dict.items(), key=lambda x: x[1], reverse=True))
            top_one = list(sorted_d.keys())[0]
            top_one_var = available_vars[top_one]
            selected_vars.append(top_one_var)
            available_vars = np.delete(available_vars, np.where(available_vars == top_one_var))
            # update covariance matrix
            w = w - 1
        return selected_vars

    def rank_sensors_with_energy_cost_updated(self, sensor_map, drone, size_of_collect_data):
        selected_vars = ['Depot']
        last_tour_distance=0
        num_sensors = len(self.sensor_map) - 1
        available_vars = list(sensor_map.keys())[:num_sensors]
        w = num_sensors  ##select all the variables
        weights_dict={}
        while w != 0:
            # todo use tsp to calculate the incurred additional total distance
            var_dis={}
            for var in available_vars:
                location_ids = selected_vars + [var]
                if len(location_ids) == 2:  # only one sensor is selected, so the drone will go from the depot to the sensor and then come back
                    n1 = (
                    float(sensor_map[location_ids[0]]['Easting']), float(sensor_map[location_ids[0]]['Northing']))
                    n2 = (
                    float(sensor_map[location_ids[1]]['Easting']), float(sensor_map[location_ids[1]]['Northing']))
                    dis = math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2) * 2
                    var_dis[var] = dis

                else:
                    coordinates = {}
                    # coordinates['Depot'] = (
                    # float(sensor_map['Depot']['Easting']), float(sensor_map['Depot']['Northing']))
                    for sensor in location_ids:
                        coordinates[sensor] = (
                        float(sensor_map[sensor]['Easting']), float(sensor_map[sensor]['Northing']))
                    my_tsp = tsp.tsp_solver(location_ids, coordinates)
                    tour, dis = my_tsp.solve()
                    var_dis[var] = dis
            sorted_vars = dict(sorted(var_dis.items(), key=lambda x: x[1]))
            min_dis_var=list(sorted_vars.keys())[0]
            selected_vars.append(min_dis_var)
            min_dis= var_dis[min_dis_var]
            incured_dis = min_dis - last_tour_distance
            energy_cost = incured_dis * drone.flying_energy_per_unit + (
                            size_of_collect_data / drone.comm_rate) * drone.hovering_energy_per_unit
            weights_dict[min_dis_var] = 1 / energy_cost
            available_vars = [var for var in available_vars if var != min_dis_var]
            # update covariance matrix
            w = w - 1
        return selected_vars
    def model_selection(self,models, X_train, Y_train, X_test, Y_test):
        overall_results = []
        total_mse_list_for_all_models=[]
        for model in models:
            print(model)
            total_mse=0
            for key in Y_train.columns.tolist():
                steps = [('scaler', StandardScaler()), ('model', model)]
                pipeline = Pipeline(steps)
                wrapped_model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
                wrapped_model.fit(X_train, Y_train[key])
                #wrapped_model.feature_importances_
                prediction = wrapped_model.predict(X_test)
                #check how the previoss
                mse = sklearn.metrics.mean_squared_error(prediction, Y_test[key])
                #todo: check why here  the score needs to multiply the Y_train.columns
                #mse=score*len(Y_train.columns.tolist())
                print("Y_train.columns.tolist: %s" %(Y_train.columns.tolist()))
                print("MSE Score for %s:" % (key))
                # averaged_score=sum(scores)/len(scores)
                print("MSE: %f" %(mse))
                total_mse+=mse
            total_mse_list_for_all_models.append(total_mse)
            print('--------------------------------------------------------------')
        return total_mse_list_for_all_models
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
            if total_energy_cost == drone.capacity:
                optimal_energy_cost= total_energy_cost
                return selected, tour, dis, total_energy_cost
            elif total_energy_cost < drone.capacity:
                selected=candidate
                left=mid+1
                optimal_tour = tour
                optimal_distance = dis
                optimal_energy_cost=total_energy_cost
            else:
                right=mid-1
        return selected, optimal_tour, optimal_distance, optimal_energy_cost

    def train_model(self):
        # use the distance as the weight for feature selection
        #vars_rank= self.rank_sensor_with_distance(num_sensors, sensor_map)
        vars_rank = self.rank_sensor_with_energy(self.sensor_map, self.drone, self.size_of_data_collection)
        #vars_rank = self.rank_sensors_with_energy_cost_updated(self.sensor_map, self.drone, self.size_of_data_collection)
        vars_rank = vars_rank[1:]
        print(vars_rank)
        selected_vars, optimal_tour, optimal_distance, optimal_energy_cost=self.binary_search(vars_rank, self.drone, self.sensor_map, self.size_of_data_collection)
        #Then use the observation of the selected sensors to train the machine learning model, and the inference data for testing
        #prepare the train and test Dataset
        #models = [GradientBoostingRegressor(), LinearSVR(), LinearRegression(), RandomForestRegressor()]
        models= [GradientBoostingRegressor()]
        mse_list=[]
        selected_vars=[int(var) for var in selected_vars]
        if len(selected_vars) !=0:
            train_data=pd.DataFrame(data=self.dataset[0:self.num_of_data_for_training,:])
            X_train=train_data.iloc[:,selected_vars]
            Y_train=train_data.drop(selected_vars, axis=1)
            test_data=pd.DataFrame(data=self.dataset[self.num_of_data_for_training:(self.num_of_data_for_inference+self.num_of_data_for_training),:])
            X_test=test_data.iloc[:,selected_vars]
            Y_test=test_data.drop(selected_vars, axis=1)
            total_mse_list_for_all_models=self.model_selection(models, X_train, Y_train, X_test, Y_test)
            #mse_list=total_mse_list_for_all_models[0]
        else:
            total_mse_list_for_all_models=[math.inf]*len(models)
        return selected_vars, total_mse_list_for_all_models, optimal_tour, optimal_distance, optimal_energy_cost,vars_rank
