import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inference
import Adaptive_Training as atrain
import plotter
import Heuristics as heu
import tsp
import geopandas as gpd
import json
class mySolution:
    def __init__(self, plotter, dataset, drone, sensor_map, number_of_training_data, number_of_estimation_data,size_of_data_collection):
        self.drone = drone
        self.dataset = dataset
        self.sensor_map =sensor_map
        self.adap_train = atrain.Adaptive_Training()
        self.heu = heu.Heuristics()
        self.plotter = plotter
        self.num_of_training_data=number_of_training_data
        self.num_of_estimation_data=number_of_estimation_data
        # self.DG=DGL.Data_Generator_and_Loader()
        self.size_of_data_collection = size_of_data_collection


    def single_variable_learning(self):
        print("loading dataset")
        Dataset = self.DG.single_normal_variable_data_loader()
        print("Adaptive Learning")
        abs_error_mean,abs_error_std=self.adap_train.Adaptive_setting_of_the_interval(Dataset)
        self.plotter.plot_abs_error_mean(abs_error_mean)
        self.plotter.plot_abs_error_std(abs_error_std)



    def compute_conditional_trace(self, cov, obs_var, unknown_var):
        '''
        This function computes the expected mse based on the true covariance matrix
        :param cov: the true covariance matrix
        :param obs_var: variables that will be observed
        :param unknown_var: variables that will be inferred
        :return: the expected mse of inference
        '''
        cov_yy = cov[np.ix_(unknown_var, unknown_var)]
        cov_ys=cov[np.ix_(unknown_var,obs_var)]
        cov_ss=cov[np.ix_(obs_var,obs_var)]
        cov_sy=cov[np.ix_(obs_var,unknown_var)]
        inverse_cov_ss = np.linalg.pinv(cov_ss)
        M=cov_yy - np.dot(np.dot(cov_ys,inverse_cov_ss),cov_sy)
        #print("expected MSE matrix M %s" %M)
        expect_mse=sum(np.diag(M))
        #print("expected mse %f" %expect_mse)
        return expect_mse


    def solve_tsp_with_energy_cap(self, sensors_json, location_ids, drone, size_of_data_collection, candidate):
        coordinates = {}
        coordinates['Depot'] = (float(sensors_json['Depot']['Easting']), float(sensors_json['Depot']['Northing']))
        for sensor in location_ids:
            coordinates[sensor] = (float(sensors_json[sensor]['Easting']), float(sensors_json[sensor]['Northing']))
        my_tsp = tsp.tsp_solver(location_ids, coordinates)
        print(location_ids)
        print(coordinates)
        tour, dis = my_tsp.solve()
        # compute the energy cost, sum of the hovering energy and the flying energy
        hovering_energy_cost = drone.hovering_energy_per_unit * (size_of_data_collection / drone.comm_rate) * (
                    len(candidate) - 1)
        flying_energy_cost = drone.flying_energy_per_unit * dis
        total_energy_cost = hovering_energy_cost + flying_energy_cost
        return total_energy_cost, tour, dis

    def run(self):
        """
        This function runs the data collection application with specific sensor selection methods

        @param dataset: numpy array of the sensing data from all sensors
        @param gamma:  the parameter to control the training accuracy
        @param sensors_json: the coordinates of the sensor map
        @param weight_method: the method for sensor selection
        @param heu: the heuristic for assigning the weight of each sensor
        @param total_dist_list: the array of the drone budget varying with a pre-difined step size
        @return: the array of averaged mse, number of selected node and optimal distance varying the drone capability
        """
        print("start learning ...")
        mean_theta, cov_theta, k, mean_2d, var_2d = self.adap_train.Adaptive_learning_of_the_interval_multi_var( self.dataset, self.num_of_training_data)   #traning process for lerning the mean and covariance of all the sensors
        #np.savetxt("Dataset/VW_2015_Jan_April_32_sensors_hourly_synthetic/mean_vec.txt", mean_theta)
        #np.savetxt("Dataset/VW_2015_Jan_April_32_sensors_hourly_synthetic/cov.txt", cov_theta)
        #sample = np.random.multivariate_normal(mean_theta, cov_theta, 10000)
        #np.savetxt('Dataset/VW_2015_Jan_April_32_sensors_hourly_synthetic/water_content_hourly.txt', sample)
        # plot the changes along the learning time of mean and var
        '''
        for i in range(0,5):
            x=np.arange(len(var_2d[0]))
            plt.figure()
            plt.plot(x, var_2d[i])
            plt.savefig("sensor %d var learning" %(i), bbox_inches='tight')
        '''
        print("--------Learning is finished---------")
        #check the leanrning error of mean_vec
        learn_error=[]
        # for i in range(len(self.real_mean_of_sensors)):
        #     learn_error.append(abs(self.real_mean_of_sensors[i]-mean_theta[i]))
        # plt.figure()
        # plt.plot(np.arange(len(self.real_mean_of_sensors)),learn_error)
        # plt.savefig('learning error of the mean')
        mse_along_time, expect_mse, obs_var, tour, optimal_dis, optimal_energy_cost, vars_rank = self.Inference_with_drone_capabilities(self.dataset, mean_theta, cov_theta, k, self.drone, self.sensor_map, self.num_of_estimation_data, self.size_of_data_collection )

        #self.Inference_with_increasing_sensors(dataset,mean_theta, cov_theta,k,heu,sensors_json)
        #self.plotter.plot_inference_mse_multi_variable_with_different_heuristics(mse_along_time_total_list_heu_dict, n_obs_var)
        #self.plotter.plot_expect_mse_with_different_heuristics(expected_mse_heu_dict,n_obs_var)
        return mse_along_time, expect_mse, obs_var, tour, optimal_dis, optimal_energy_cost, vars_rank




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
               total_energy_cost, tour, dis =  self.solve_tsp_with_energy_cap(sensors_json, location_ids, drone, size_of_data_collection, candidate)
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
        #after selecting the sensors within the energy budget, use the remaining one to select the closed ones within the budget.
        #Iterate the reminding ones and find the ones which can fit the reminding energy capacity
        # remaining_energy=drone.capacity - optimal_energy_cost
        # if remaining_energy>0:
        #     available_sensors=[s for s in ranked_sensors if s not in selected]
        #     for sensor in available_sensors:
        #         selected.append(sensor)
        #         location_ids = list(map(str, selected))
        #         location_ids.append('Depot')
        #         if len(location_ids) == 2:  # only one sensor is selected, so the drone will go from the depot to the sensor and then come back
        #             n1 = (
        #             float(sensors_json[location_ids[0]]['Easting']), float(sensors_json[location_ids[0]]['Northing']))
        #             n2 = (
        #             float(sensors_json[location_ids[1]]['Easting']), float(sensors_json[location_ids[1]]['Northing']))
        #             tour = location_ids
        #             dis = math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2) * 2
        #             total_energy_cost = dis * drone.flying_energy_per_unit + (size_of_data_collection / drone.comm_rate)
        #         else:
        #             total_energy_cost, tour, dis = self.solve_tsp_with_energy_cap(sensors_json, location_ids, drone, size_of_data_collection, selected)
        #         if total_energy_cost == drone.capacity:
        #             optimal_energy_cost= total_energy_cost
        #             return selected, tour, dis, total_energy_cost
        #         elif total_energy_cost < drone.capacity:
        #             optimal_tour = tour
        #             optimal_distance = dis
        #             optimal_energy_cost=total_energy_cost
        #         else:
        #             selected.remove(sensor)
        return selected, optimal_tour, optimal_distance, optimal_energy_cost

    def Inference_with_drone_capabilities(self, dataset,mean_theta, cov_theta,k, drone, sensors_json, number_of_estimation_data, size_of_data_collection):
        """ The function selects the observation sensors and infer the data from the unselected ones
        @param dataset:  the dataset of sensor readings
        @param mean_theta: estimated mean vector
        @param cov_theta: estimated covariance vector
        @param k: the number of the data in dataset used for training
        @param heuristic: the method for ranking the features
        @param drone: the drone object, including energy capacity, hovering_energy_per_unit, flying_energy_per_unit
        @param sensors_json: the sensor map contains the coordinates of the sensors
        @param feature_selection_method: the methods to solve the problem, including ours and the comparison
        @param num_inference: the amount of sensing data of the unselected sensors to be estimated
        @return:
        """
        print("start inference under drone capabilities ...")
        n_obs_var = []
        x = np.arange(len(mean_theta))
        mse_along_time_total_list = []
        expected_mse_list = []
        # if heuristic == "topw":
        #     vars_rank = self.heu.topw(cov_theta, sensors_json)  #note: here is no feature_selection_method incoorperate wrt drone
        # elif heuristic == "topw_update":
        vars_rank = self.heu.topw_update_ranking_list(cov_theta, sensors_json,  drone, size_of_data_collection)  #the current adopted approach

        vars_rank = vars_rank[1:]
        print("sensor_weight_rank: %s" %(vars_rank) )
        #apply binary search to select the subset of sensors within drone capabilities
        obs_var, tour, optimal_dis, optimal_energy_cost =self.binary_search(vars_rank, drone, sensors_json, size_of_data_collection)
        #print("selected %d vars %s:" %(len(obs_var), obs_var))
        #todo handle the case when there is no sensor selected. So there is no inference. put a mark in the plot.
        if len(obs_var)!=0:
            unknown_vars = np.delete(x, obs_var)
            self.infer = inference.Inference(dataset, mean_theta, cov_theta)
            expect_mse=[]
            expect_mse = self.compute_conditional_trace(cov_theta, obs_var, unknown_vars)
            #mse_along_time is the a list for mse for all the sensors
            mse_along_time, inferred_all = self.infer.infer_unobs(obs_var, unknown_vars, k, number_of_estimation_data)
            #mse_along_time_total_list.append(mse_along_time)
            expected_mse_list.append(expect_mse)
            #averaged_mse_along_time = np.average(mse_along_time_total_list, axis=1)
            #self.plotter.plot_inference_mse_multi_variable(mse_along_time_total_list, expected_mse_list, n_obs_var)
            print("-------- Inference finished --------" )
        else: #the capacity can not cover the short round trip
            mse_along_time=[math.inf] * number_of_estimation_data
            expect_mse=math.inf
            obs_var=[]
            tour=[]
            optimal_dis=0
        return mse_along_time, expect_mse, obs_var, tour, optimal_dis, optimal_energy_cost, vars_rank

    def Inference_with_increasing_sensors(self, dataset,mean_theta, cov_theta,k,heuristic, sensors_json):
        print("start inference varying the number of sensors with %s selection ..." %(heuristic) )
        n_obs_var = []
        x = np.arange(len(mean_theta))
        mse_along_time_total_list = []
        expected_mse_list = []
        random_selected=[]
        opt_dists=[]
        for n in range(3, len(mean_theta)):  # select the observation variable from 1 to 9
            obs_var=[]
            if heuristic == 'random':
                candidate= [i for i in x if i not in random_selected]
                select = np.random.choice(candidate)
                random_selected.append(select)
                obs_var=random_selected
            elif heuristic=='topw':
               obs_var = self.heu.topw(cov_theta, n)
            elif heuristic=='topw update':
                obs_var = self.heu.topw_update(cov_theta,n)
            print("selected %d vars %s:" %(n, obs_var))
            selected_sensors=list(map(str, obs_var))
            coordinates = {}
            for sensor in selected_sensors:
                coordinates[sensor] = (float(sensors_json[sensor]['Easting']), float(sensors_json[sensor]['Northing']))
            mytsp=tsp.tsp_solver(selected_sensors,coordinates)
            tour, optimal_distance=mytsp.solve()
            opt_dists.append(optimal_distance)
            unknown_vars = np.delete(x, obs_var)
            n_obs_var.append(n)
            self.infer = inference.Inference(dataset, mean_theta, cov_theta)
            expect_mse = self.compute_conditional_trace(cov_theta, obs_var, unknown_vars)
            mse_along_time, inferred_all = self.infer.infer_unobs(obs_var, unknown_vars, k)
            mse_along_time_total_list.append(mse_along_time)
            expected_mse_list.append(expect_mse)
        averaged_mse_along_time = np.average(mse_along_time_total_list, axis=1)
        self.plotter.plot_distance_varying_sensors(range(3,len(mean_theta)),opt_dists)
        #self.plotter.plot_inference_mse_multi_variable(mse_along_time_total_list, expected_mse_list, n_obs_var)
        print("-------- Inference finished --------" )
        return averaged_mse_along_time, expected_mse_list, n_obs_var

    def data_statistics(self, filedict, filetype):
        Dataframe = self.preprocess.load_data(filedict, filetype)
        features = ['Location', 'Date','Time','VW_30cm']
        sliced_Dataframe = self.preprocess.slice_dataframe_daily(Dataframe, features)
        self.preprocess.statistics(sliced_Dataframe)

    def create_data_frame_VW_2015_Jan_April(self, filedict, filetype):
        dataframe = self.preprocess.load_data(filedict, filetype)
        features = ['Location', 'Date', 'Time', 'VW_30cm']
        year = 2015
        time_period = [1, 2, 3, 4]  # study the water content measurements from January-April
        locations = ['CAF095', 'CAF308', 'CAF135', 'CAF119', 'CAF125', 'CAF245', 'CAF133', 'CAF237', 'CAF035', 'CAF141',
                     'CAF357', 'CAF197', 'CAF397', 'CAF231', 'CAF033', 'CAF351', 'CAF019', 'CAF031', 'CAF377', 'CAF201',
                     'CAF215', 'CAF003', 'CAF163', 'CAF007', 'CAF401', 'CAF314', 'CAF075', 'CAF061', 'CAF139', 'CAF312',
                     'CAF310', 'CAF067']
        sliced_Dataframe = self.preprocess.slice_data_frame_hourly(dataframe, features, year, time_period, locations)
        # self.preprocess.statistics(sliced_Dataframe)

        self.preprocess.create_learning_dataframe(sliced_Dataframe)

    def  load_VW_data_2015_Jan_April(self):
        df = pd.read_csv("Dataset/water_content_2015_Jan_April_32_sensors")
        sensors=df.columns.values.tolist()
        VW_matrix = df.iloc[:, 2:].to_numpy()
        mean_vector = np.mean(VW_matrix, axis=0)
        std_vector = np.std(VW_matrix, axis=0)
        print("mean: %s" % (mean_vector))
        print("std: %s" % (std_vector))
        #load the location of sensors
        # np.random.shuffle(VW_matrix)
        # corr_vector=np.corrcoef(np.transpose(VW_matrix))
        # corr=np.triu(corr_vector).flatten()
        # corr_array=[cor for cor in corr if cor!=0]
        # self.plotter.plot_cummulative_probability(corr_array,'VW_cummulative_probablibity')
        #np.random.shuffle(VW_matrix)
        return VW_matrix

    def generate_sensors_location(self):
        df = pd.read_csv("Dataset/water_content_2015_Jan_April_32_sensors")
        sensors = df.columns.values.tolist()
        used_sensors = sensors[2:]
        gdf = gpd.read_file("Dataset/sensors/CAF_sensors.shp")
        sensors = {}
        index=0
        most_est=math.inf
        most_north=math.inf
        for s in used_sensors:
            row_df = gdf[gdf['Location'] == s]
            coor = list(row_df.iloc[0]['geometry'].coords)[0]
            if coor[0] < most_est:
                most_est=coor[0]
            if coor[1] < most_north:
                most_north=coor[1]
        for s in used_sensors:
            sensor={}
            row_df=gdf[gdf['Location']==s]
            sensor['Location'] = row_df.iloc[0]['Location']
            coor = list(row_df.iloc[0]['geometry'].coords)[0]
            sensor['Easting'] = coor[0]-most_est
            sensor['Northing'] = coor[1]-most_north
            sensors[index] = sensor
            index +=1
        sensor={}
        sensor['Location'] = "Depot"
        sensor['Easting']=0
        sensor['Northing']=500
        sensors['Depot'] = sensor
        sensors_loc = open("Dataset/sensors/CAF_sensors.json", 'w')
        json.dump(sensors, sensors_loc, indent=6)

    def loadmap(self, map_path):
        # Read capital names and coordinates from json file
        nodes=[]
        coordinates={}
        sensors_json = json.load(open(map_path))
        for sensor in sensors_json:
            # sensor_loc = sensors_json[sensor]['Location']
            nodes.append(sensor)
            coordinates[sensor] = (float(sensors_json[sensor]['Easting']), float(sensors_json[sensor]['Northing']))
        coordinates=coordinates.values()
        self.plotter.plot_sensor_map(coordinates, nodes)
        return sensors_json
