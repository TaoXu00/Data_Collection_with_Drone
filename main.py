import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inference
import Data_Generation as DG
import Adaptive_Training as atrain
import Data_Preprocess as preprocess
import plotter
import Heuristics as heu
import tsp
import geopandas as gpd
import json
class Data_Correlation:
    def __init__(self):
        Dir_plots="plots/"
        self.DG=DG.Data_generator()
        self.adap_train=atrain.Adaptive_Training()
        self.plotter=plotter.plotter(Dir_plots)
        self.preprocess=preprocess.Data_Preprocess()
        self.heu=heu.Heuristics()

    def single_variable_learning(self):
        print("loading dataset")
        Dataset = self.DG.single_normal_variable_data_loader()
        print("Adaptive Learning")
        abs_error_mean,abs_error_std=self.adap_train.Adaptive_setting_of_the_interval(Dataset)
        self.plotter.plot_abs_error_mean(abs_error_mean)
        self.plotter.plot_abs_error_std(abs_error_std)

    def generate_dataset(self):
        #self.DG.single_normal_variable_genrator(650,100,100000)
        mean_vec, cov=self.DG.multi_variables_normal_distribution_generator(650, 800, 3, 8, 10)
        np.savetxt("Dataset/synthetic/mean_vec.txt",mean_vec)
        np.savetxt("Dataset/synthetic/cov.txt",cov)


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
        inverse_cov_ss = np.linalg.inv(cov_ss)
        M=cov_yy - np.dot(np.dot(cov_ys,inverse_cov_ss),cov_sy)
        #print("expected MSE matrix M %s" %M)
        expect_mse=sum(np.diag(M))
        #print("expected mse %f" %expect_mse)
        return expect_mse

    def run(self, dataset, gamma, sensors_json, total_dist):
        print("start learning ...")
        mean_theta, cov_theta, k, mean_2d, var_2d = self.adap_train.Adaptive_learning_of_the_interval_multi_var(dataset, gamma)
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
        real_mean=np.loadtxt('Dataset/VW_2015_Jan_April_32_sensors_hourly_synthetic/mean_vec.txt')
        learn_error=[]
        for i in range(len(real_mean)):
            learn_error.append(abs(real_mean[i]-mean_theta[i]))
        plt.figure()
        plt.plot(np.arange(len(real_mean)),learn_error)
        plt.savefig('learning error of the mean')
        #print(real_mean)
        heu='topw update'
        mse_along_time_total_list_heu_dict={}
        expected_mse_heu_dict={}
        expected_mse_list=[]
        #vary the drone distance capability
        mse_along_time_total=[]
        total_dist_list=np.arange(600,4200,200)
        num_of_selected_nodes=[]
        optimal_dis_list=[]
        tour_coordinates={}
        with open('path_plan.txt', 'w') as f:
            f.write('Path Plan with different drone capabilities\n')
            tour_id=0
            for total_dist in total_dist_list:
                mse_along_time, expect_mse, obs_var, tour, optimal_dis = self.Inference_with_drone_capabilities(dataset,mean_theta, cov_theta,k, heu, total_dist, sensors_json)
                mse_along_time_total.append(mse_along_time)
                f.write('***************************************************************\n')
                f.write('drone_constraint: %d\n' %(total_dist))
                f.write('%d sensors are selected \n' %(len(obs_var)))
                f.write('optimal tour: %s\n' %(tour))
                f.write('length of the tour %f\n' %(optimal_dis))
                num_of_selected_nodes.append(len(obs_var))
                expected_mse_list.append(expect_mse)
                optimal_dis_list.append(optimal_dis)
                self.plotter.plot_tour_map(sensors_json,tour, tour_id, optimal_dis)
                tour_id+=1
        averaged_mse_varying_drone_capabilities=np.average(mse_along_time_total, axis=1)
        self.plotter.plot_averaged_mse_vary_drone_capability(total_dist_list,averaged_mse_varying_drone_capabilities, expected_mse_list)
        self.plotter.plot_selected_sensors_vary_drone_capability(total_dist_list, num_of_selected_nodes)
        self.plotter.plot_tour_length_vary_drone_capability(total_dist_list, optimal_dis_list)

        #self.Inference_with_increasing_sensors(dataset,mean_theta, cov_theta,k,heu,sensors_json)
        #self.plotter.plot_inference_mse_multi_variable_with_different_heuristics(mse_along_time_total_list_heu_dict, n_obs_var)
        #self.plotter.plot_expect_mse_with_different_heuristics(expected_mse_heu_dict,n_obs_var)

    def binary_search(self, ranked_sensors, target,sensors_json):
        # initialize the lower and upper bounds
        left = 0
        right = len(ranked_sensors)-1
        ## add the Depot point
        selected=[]
        # loop until the bounds cross each other
        optimal_tour=[]
        optimal_distance=0
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
                dis = math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)
            else:
                coordinates={}
                coordinates['Depot']=(float(sensors_json['Depot']['Easting']), float(sensors_json['Depot']['Northing']))
                for sensor in location_ids:
                    coordinates[sensor] = (float(sensors_json[sensor]['Easting']), float(sensors_json[sensor]['Northing']))
                my_tsp=tsp.tsp_solver(location_ids, coordinates)
                tour, dis= my_tsp.solve()
            if dis == target:
                return selected,optimal_tour, optimal_distance
            elif dis < target:
                selected=candidate
                left=mid+1
                optimal_tour = tour
                optimal_distance = dis
            else:
                right=mid-1
        return selected, optimal_tour, optimal_distance

    def Inference_with_drone_capabilities(self, dataset,mean_theta, cov_theta,k, heuristic, total_distance, sensors_json):
        print("start inference under drone capabilities with %s selection ..." %(heuristic) )
        n_obs_var = []
        x = np.arange(len(mean_theta))
        mse_along_time_total_list = []
        expected_mse_list = []
        vars_rank=self.heu.topw_update_rank_list(cov_theta)
        #print("sensor_weight_rank: %s" %(vars_rank) )
        #apply binary search to select the subset of sensors within drone capabilities
        obs_var, tour, optimal_dis=self.binary_search(vars_rank, total_distance, sensors_json)
        #print("selected %d vars %s:" %(len(obs_var), obs_var))
        #todo handle the case when there is no sensor selected. So there is no inference. put a mark in the plot.
        if len(obs_var)!=0:
            unknown_vars = np.delete(x, obs_var)
            self.infer = inference.Inference(dataset, mean_theta, cov_theta)
            expect_mse = self.compute_conditional_trace(cov_theta, obs_var, unknown_vars)
            mse_along_time, inferred_all = self.infer.infer_unobs(obs_var, unknown_vars, k)
            #mse_along_time_total_list.append(mse_along_time)
            expected_mse_list.append(expect_mse)
            #averaged_mse_along_time = np.average(mse_along_time_total_list, axis=1)
            #self.plotter.plot_inference_mse_multi_variable(mse_along_time_total_list, expected_mse_list, n_obs_var)
            print("-------- Inference finished --------" )
        return mse_along_time, expect_mse, obs_var, tour, optimal_dis

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

    def loadmap(self):
        # Read capital names and coordinates from json file
        nodes=[]
        coordinates={}
        sensors_json = json.load(open('Dataset/sensors/CAF_sensors.json'))
        for sensor in sensors_json:
            # sensor_loc = sensors_json[sensor]['Location']
            nodes.append(sensor)
            coordinates[sensor] = (float(sensors_json[sensor]['Easting']), float(sensors_json[sensor]['Northing']))
        coordinates=coordinates.values()
        self.plotter.plot_sensor_map(coordinates, nodes)
        return sensors_json
def main():
    system = Data_Correlation()
    #system.generate_dataset()
    #main_df=system.preprocess.load_data("Dataset/Daily", 'txt')
    #system.data_statistics("Dataset/Daily", 'txt')
    #system.data_statistics("Dataset/Hourly", 'txt')
    #this is the file for creating the learning dataframe
    #system.create_data_frame_VW_2015_Jan_April("Dataset/Hourly", 'txt') # the CSV file is stored in Dataset/water_content_2015_Jan_April_32_sensors
    #Dataset, mean_vec, cov = system.DG.multi_normal_variable_data_loader()
    #real dataset
    #Dataset=system.load_VW_data_2015_Jan_April()
    #system.generate_sensors_location()
    #synthetic dataset
    Dataset=np.loadtxt('Dataset/VW_2015_Jan_April_32_sensors_hourly_synthetic/water_content_hourly.txt')
    sensors_json=system.loadmap()
    total_dist=3000
    system.run(Dataset,0.01, sensors_json, total_dist)
    #tour, optimal_dis=system.tsp.solve()
    #print(tour, optimal_dis)
    #todo run with synthetic data with expected error



if __name__ == "__main__":
    main()