import math
import random

import numpy as np

import ML_Baseline
import ML_Baseline as ML_BS
import plotter as plotter
import Feature_Selection_based_Baseline as FS
import mySolution as mySolution
import json
import Drone as Drone
import os
import Data_Generator_and_Loader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
class Exp1:
    def __init__(self, config):
        self.config = config
        self.DGL = Data_Generator_and_Loader.Data_Generator_and_Loader()

    def plot_sensor_graph(self, sensor_map, plotter, name):
        nodes = []
        coordinates = {}
        for sensor in sensor_map:
            # sensor_loc = sensors_json[sensor]['Location']
            nodes.append(sensor)
            print(sensor)
            print(sensor_map[sensor]['Easting'],sensor_map[sensor]['Northing'])
            coordinates[sensor] = (float(sensor_map[sensor]['Easting']), float(sensor_map[sensor]['Northing']))
        coordinates = coordinates.values()
        plotter.plot_sensor_map(coordinates, nodes, name)

    def compute_avg_std_for_each_solu_exp1(self, dirs_solu, sensor_maps, metric_file_name, metric):
        for dir_solu in dirs_solu:
            mse_list_2D=[]
            for sensor_map_file in sensor_maps:
                map_name, file_extention = os.path.splitext(sensor_map_file)
                mse_file = dir_solu+map_name+'/'+metric_file_name
                mse = np.loadtxt(mse_file)
                mse_list_2D.append(mse)
            combined_mse_2D=np.array(mse_list_2D)
            avg= np.mean(combined_mse_2D,axis=0)
            std= np.std(combined_mse_2D, axis=0)
            np.savetxt("%s/%s_avg.txt" %(dir_solu,metric),avg)
            np.savetxt("%s/%s_std.txt" %(dir_solu,metric),std)

    def generate_sensor_maps(self, map_x_scale, map_y_scale, dir_maps, dir_sensor_map, map_num, plotter):
        for i in range(1, map_num+1):
            with open(dir_sensor_map, 'r') as json_file:
                data = json_file.read()
                sensor_map= json.loads(data)
            for sensor in sensor_map:
                print(sensor)
                if sensor != "Depot":
                    sensor_map[sensor]["Easting"] = random.uniform(0, map_x_scale)
                    sensor_map[sensor]["Northing"] = random.uniform(0, map_y_scale)
            updated_sensor_map=json.dumps(sensor_map, indent=4)
            with open('%smap_%d.json' %(dir_maps,i), 'w') as json_file:
                    json_file.write(updated_sensor_map)
            with open('%smap_%d.json' %(dir_maps,i), 'r') as json_file:
                data = json_file.read()
                sensor_map= json.loads(data)
            path_dataset = '%sdataset_map_%d.txt' %(dir_maps,i)

            self.DGL.generate_synthetic_dataset_with_correlation_considered_with_distance(sensor_map,path_dataset)
            self.plot_sensor_graph(sensor_map, plotter, '%smap_%d' % (dir_maps,i))

    def generate_clustered_maps(self, map_x_scale, map_y_scale, dir_maps, map_num, plotter):
        for i in range(1, map_num+1):
            sensor_map={}
            num_clusters= 7
            points_per_cluster=5
            # Generate random cluster centers
            cluster_centers = np.random.rand(num_clusters, 2) * [map_x_scale, map_y_scale]

            # Generate data points using K-Means clustering
            X = []
            desired_std=40
            for j in range(num_clusters):
                cluster_points = cluster_centers[j] + np.random.randn(points_per_cluster, 2)* desired_std
                X.extend(cluster_points)

            id=0
            for point in X:
                location = {"Easting": point[0], "Northing": point[1]}
                sensor_map[str(id)]=location
                id+=1
            location = {}
            location['Easting'] = 0
            location['Northing'] = 0
            sensor_map['Depot'] = location
            new_sensor_map=json.dumps(sensor_map, indent=4)
            with open('%smap_%d.json' %(dir_maps,i), 'w') as json_file:
                    json_file.write(new_sensor_map)
            with open('%smap_%d.json' %(dir_maps,i), 'r') as json_file:
                data = json_file.read()
                sensor_map= json.loads(data)
            path_dataset = '%sdataset_map_%d.txt' %(dir_maps,i)
            self.DGL.generate_synthetic_dataset_with_correlation_considered_with_distance(sensor_map,path_dataset)
            self.plot_sensor_graph(sensor_map, plotter, '%smap_%d' % (dir_maps,i))

    def generate_synthetic_data(self):
        #7 cluster, 5 points per cluster
        dataset=np.loadtxt('solar_radiation_dataset/DataMatrix_317.txt')
        cluster1=dataset[:, :4]
        # Duplicate the column (add it back to the array)
        cluster1 = np.insert(cluster1, 3, dataset[:,0], axis=1)
        print(cluster1.shape)
        cluster2=dataset[:,4:8]
        cluster2=np.insert(cluster2, 3, dataset[:,3], axis=1)
        cluster3 = dataset[:, 8:12]
        cluster3 = np.insert(cluster3, 3, dataset[:, 8], axis=1)
        cluster4 = dataset[:, 12:16]
        cluster4 = np.insert(cluster4, 3, dataset[:, 16], axis=1)
        cluster5 = dataset[:, 16:19]
        cluster5 = np.concatenate((cluster5, dataset[:, 16:18]), axis=1)
        cluster6 =cluster1
        cluster7=cluster2
        synthetic_dataset=np.concatenate((cluster1,cluster2,cluster3,cluster4,cluster5,cluster6,cluster7), axis=1)
        print(synthetic_dataset.shape)
        np.savetxt('Dataset/sensor_maps/dataset_map_2_DataMatrix_317.txt', synthetic_dataset)



    def exp_1_my_solution(self,dir_mysolu, maximum_drone_energy_capacity, step_size,  dataset, sensor_map,
                           hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data, real_mean_of_sensors,
                           num_of_estimation_data, exp1_my_solu_plotter,size_data_collection, drone_commu_rate,
                           mse_file_name,sensor_length_file_name):
        expected_mse_list = []
        mse_along_time_total = []
        num_of_selected_nodes = []
        optimal_dis_list = []
        optimal_energy_cost_list = []
        drone_capacity_list=range(0, maximum_drone_energy_capacity, step_size)
        with open('%spath_plan_mysolution.txt' %(dir_mysolu), 'w') as f:
            f.write('Path Plan with different drone capabilities\n') ## track the trajectory planned for each drone capacity
            tour_id=0
            for drone_capacity in drone_capacity_list:
                drone = Drone.Drone(drone_capacity, hovering_energy_per_unit, flying_energy_per_unit, drone_commu_rate)
                mySolu=mySolution.mySolution(plotter, dataset, drone, sensor_map, number_of_training_data, num_of_estimation_data, size_data_collection)
                mse_along_time, expect_mse, selected_sensors, tour, optimal_dis, optimal_energy_cost, vars_rank =mySolu.run()
                mse_along_time_total.append(mse_along_time)
                f.write('***************************************************************\n')
                f.write('vars_rank: %s \n' %(vars_rank))
                f.write('drone_constraint: %d\n' % (drone_capacity))
                f.write('%d sensors are selected \n' % (len(selected_sensors)))
                f.write('optimal tour: %s\n' % (tour))
                f.write('length of the tour %f\n' % (optimal_dis))
                f.write('optimal energy cost %f\n' %(optimal_energy_cost))
                num_of_selected_nodes.append(len(selected_sensors))
                expected_mse_list.append(expect_mse)
                optimal_dis_list.append(optimal_dis)
                optimal_energy_cost_list.append(optimal_energy_cost)
                if len(tour) != 0:
                    exp1_my_solu_plotter.plot_tour_map(sensor_map, tour, tour_id, optimal_dis, optimal_energy_cost)
                    tour_id += 1
            averaged_mse_varying_drone_capabilities = np.average(mse_along_time_total, axis=1)
            exp1_my_solu_plotter.plot_averaged_mse_vary_drone_capability(drone_capacity_list,
                                                                 averaged_mse_varying_drone_capabilities,
                                                                 expected_mse_list)
            exp1_my_solu_plotter.plot_selected_sensors_vary_drone_capability(drone_capacity_list, num_of_selected_nodes)
            exp1_my_solu_plotter.plot_tour_length_vary_drone_capability(drone_capacity_list, optimal_dis_list)
            exp1_my_solu_plotter.plot_energy_cost_vary_drone_capability(drone_capacity_list, optimal_energy_cost_list)
            np.savetxt("%s/%s" % (dir_mysolu, sensor_length_file_name), num_of_selected_nodes)
            np.savetxt("%s/%s" % (dir_mysolu, mse_file_name),
                       averaged_mse_varying_drone_capabilities)



    def exp1_ML_baseline(self, dir_baseline_ml, maximum_drone_energy_capacity, step_size,  dataset, sensor_map, hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data, num_of_estimation_data, exp1_baseline_plotter, size_data_collection, drone_commu_rate,
                          mse_file_name, sensor_length_file_name):
        mse_list_total = []
        num_of_selected_nodes = []
        optimal_dis_list = []
        optimal_energy_cost_list = []
        drone_capacity_list = range(0, maximum_drone_energy_capacity, step_size)
        with open('%spath_plan_baseline_ml.txt' % (dir_baseline_ml), 'w') as f:
            f.write(
                'Path Plan with different drone capabilities\n')  ## track the trajectory planned for each drone capacity
            tour_id = 0
            for drone_capacity in drone_capacity_list:
                drone = Drone.Drone(drone_capacity, hovering_energy_per_unit, flying_energy_per_unit, drone_commu_rate)
                ml_bs = ML_BS.ML_Baseline(dir_baseline_ml, drone, sensor_map, dataset, size_data_collection, number_of_training_data, num_of_estimation_data)
                selected_vars,total_mse_for_all_models , optimal_tour, optimal_distance, optimal_energy_cost, vars_rank = ml_bs.train_model()
                if len(optimal_tour) != 0:
                    exp1_baseline_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance, optimal_energy_cost)
                tour_id += 1
        # exp1_baseline_plotter.plotter.plot_mse_with_varying_drone_cap_for_different_ML_models(total_dist_list,
        #                                                              total_mse_list_for_all_ML_models_varying_drone_capability)


                f.write('***************************************************************\n')
                f.write('vars_rank: %s \n' %(vars_rank))
                f.write('drone_constraint: %d\n' % (drone_capacity))
                f.write('%d sensors are selected \n' % (len(selected_vars)))
                f.write('optimal tour: %s\n' % (optimal_tour))
                f.write('length of the tour %f\n' % (optimal_distance))
                f.write('optimal energy cost %f\n' % (optimal_energy_cost))
                num_of_selected_nodes.append(len(selected_vars))
                optimal_dis_list.append(optimal_distance)
                optimal_energy_cost_list.append(optimal_energy_cost)
                #here we only have one model GBR
                mse_list_total.append(total_mse_for_all_models[0])
                if len(optimal_tour) != 0:
                    exp1_baseline_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance, optimal_energy_cost)
                    tour_id += 1
            exp1_baseline_plotter.plot_mse_with_varying_drone_capabilities(drone_capacity_list, mse_list_total)
            exp1_baseline_plotter.plot_selected_sensors_vary_drone_capability(drone_capacity_list, num_of_selected_nodes)
            exp1_baseline_plotter.plot_tour_length_vary_drone_capability(drone_capacity_list, optimal_dis_list)
            exp1_baseline_plotter.plot_energy_cost_vary_drone_capability(drone_capacity_list, optimal_energy_cost_list)
            np.savetxt("%s/%s" %(dir_baseline_ml,sensor_length_file_name), num_of_selected_nodes)
            np.savetxt("%s/%s" %(dir_baseline_ml,mse_file_name), mse_list_total)

    def exp1_FS_baseline(self,dir_baseline_fs, maximum_drone_energy_capacity, step_size, dataset, sensor_map,
                              hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data,
                              num_of_estimation_data, exp1_baseline_fs_plotter, size_data_collection, drone_commu_rate,
                              mse_file_name, sensor_length_file_name):

        mse_list_total = []
        num_of_selected_nodes = []
        optimal_dis_list = []
        optimal_energy_cost_list = []
        drone_capacity_list = range(0, maximum_drone_energy_capacity, step_size)
        with open('%spath_plan_baseline_fs.txt' % (dir_baseline_fs), 'w') as f:
            f.write(
                'Path Plan with different drone capabilities\n')  ## track the trajectory planned for each drone capacity
            tour_id = 0
            drone = Drone.Drone(0, hovering_energy_per_unit, flying_energy_per_unit, drone_commu_rate)
            fs_bs = FS.Feature_selection_based_baseline(dir_baseline_fs, drone, sensor_map, dataset,
                                                        size_data_collection,
                                                        number_of_training_data, num_of_estimation_data)
            vars_rank = fs_bs.calculate_feature_importance(dataset)
            print("vars_rank %s:" %(vars_rank))
            for drone_capacity in drone_capacity_list:
                drone = Drone.Drone(drone_capacity, hovering_energy_per_unit, flying_energy_per_unit, drone_commu_rate)
                fs_bs = FS.Feature_selection_based_baseline(dir_baseline_fs, drone, sensor_map, dataset,
                                                            size_data_collection,
                                                            number_of_training_data, num_of_estimation_data)

                total_mse, selected_vars, optimal_tour, optimal_distance, optimal_energy_cost, vars_rank = fs_bs.train_model(vars_rank, drone)
                if len(optimal_tour) != 0:
                    exp1_baseline_fs_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance,
                                                        optimal_energy_cost)
                tour_id += 1
                # exp1_baseline_plotter.plotter.plot_mse_with_varying_drone_cap_for_different_ML_models(total_dist_list,
                #                                                              total_mse_list_for_all_ML_models_varying_drone_capability)

                f.write('***************************************************************\n')
                f.write('vars rank: %s\n' %(vars_rank))
                f.write('drone_constraint: %d\n' % (drone_capacity))
                f.write('%d sensors are selected \n' % (len(selected_vars)))
                f.write('optimal tour: %s\n' % (optimal_tour))
                f.write('length of the tour %f\n' % (optimal_distance))
                f.write('optimal energy cost %f\n' % (optimal_energy_cost))
                num_of_selected_nodes.append(len(selected_vars))
                optimal_dis_list.append(optimal_distance)
                optimal_energy_cost_list.append(optimal_energy_cost)
                # here we only have one model GBR
                mse_list_total.append(total_mse)
                if len(optimal_tour) != 0:
                    exp1_baseline_fs_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance,
                                                        optimal_energy_cost)
                    tour_id += 1
            exp1_baseline_fs_plotter.plot_mse_with_varying_drone_capabilities(drone_capacity_list, mse_list_total)
            exp1_baseline_fs_plotter.plot_selected_sensors_vary_drone_capability(drone_capacity_list, num_of_selected_nodes)
            exp1_baseline_fs_plotter.plot_tour_length_vary_drone_capability(drone_capacity_list, optimal_dis_list)
            exp1_baseline_fs_plotter.plot_energy_cost_vary_drone_capability(drone_capacity_list, optimal_energy_cost_list)
            np.savetxt("%s/%s" % (dir_baseline_fs, sensor_length_file_name), num_of_selected_nodes)

            np.savetxt("%s/%s" % (dir_baseline_fs, mse_file_name), mse_list_total)


    def exp_1(self, dir_exp1):
        # This set of experiments varying the drone capability
        # run the program with different maps, then plot the average values with error bar.
        dir_maps=self.config['Exp_1']['dir_sensor_maps']
        ## read the parameters of exp1
        dir_sensor_map = self.config['Exp_1']['dir_sensor_map']

        map_x_scale= int(self.config['Exp_1']['map_x_scale'])
        map_y_scale =int(self.config['Exp_1']['map_y_scale'])
        map_num = int(self.config['Exp_1']['map_num'])
        dir_dataset = self.config['Exp_1']['dir_dataset']
        dir_mysolu= self.config['Exp_1']['dir_mysolu']
        dir_baseline_ml = self.config['Exp_1']['dir_baseline_ml']
        dir_baseline_fs = self.config['Exp_1']['dir_baseline_fs']

        dir_real_mean_of_sensors = self.config['Exp_1']['dir_real_mean_of_sensors']
        maximum_drone_capacity = int(self.config['Exp_1']['maximum_drone_capacity'])
        maximum_drone_energy_capacity =int(self.config['Exp_1']['maximum_energy_capacity'])
        step_size = int(self.config['Exp_1']['step_size'])
        hovering_energy_per_unit = float(self.config['Drone']['unit_hovering_energy'])
        flying_energy_per_unit = float(self.config['Drone']['unit_flying_energy'])
        number_of_training_data= int(self.config['Exp_1']['num_of_training_data'])
        num_of_estimation_data= int(self.config['Exp_1']['num_of_estimation_data'])
        size_data_collection = int(self.config['Exp_1']['size_data_collection'])
        drone_commu_rate = int(self.config['Drone']['comm_rate'])
        mse_file_name = self.config['Exp_1']['mse_file_name']
        sensor_length_file_name=self.config['Exp_1']['sensor_length_file_name']
        if not os.path.exists(dir_maps):
            os.makedirs(dir_maps)
            os.chmod(dir_maps, 0o700)

        self.plotter = plotter.plotter(dir_exp1)
        #self.generate_sensor_maps(map_x_scale, map_y_scale, dir_maps, dir_sensor_map, map_num, self.plotter)
        #self.generate_clustered_maps(map_x_scale, map_y_scale, dir_maps, map_num, self.plotter)
        self.generate_synthetic_data()
        # # List all maps in the directory
        # files = os.listdir(dir_maps)
        # sensor_maps = []
        # for file in files:
        #     if file.endswith('.json'):
        #         sensor_maps.append(file)
        #
        # # plot the sensor maps
        # for sensor_map_file in sensor_maps:
        #     name, ext= os.path.splitext('%s%s'%(dir_maps, sensor_map_file))
        #     sensor_map = json.load(open(dir_maps+sensor_map_file))
        #     self.plot_sensor_graph(sensor_map, self.plotter, name)
        #
        # # load dataset
        # # print(dir_dataset)
        # # dataset=np.loadtxt(dir_dataset)
        #
        #
        # #load real mean of sensors
        # real_mean_of_sensors = np.loadtxt(dir_real_mean_of_sensors)
        #
        #
        # # Iterate through the files
        # for sensor_map_file in sensor_maps:
        #     # Check if the current item is a file
        #     if os.path.isfile(os.path.join(dir_maps, sensor_map_file)):
        #         print("File:", sensor_map_file)
        #
        #     # load and plot sensor map
        #     sensor_map = json.load(open(dir_maps+sensor_map_file))
        #
        #     #load dataset
        #     name, ext = os.path.splitext(sensor_map_file)
        #     dataset=np.loadtxt(dir_maps+'dataset_'+ name+'.txt')
        #
        #     # Create the directory if it doesn't exist
        #     map_name, file_extention= os.path.splitext(sensor_map_file)
        #
        #     #----------------------------Exp1 my solution ------------------------------------------------------------------
        #     if not os.path.exists(dir_mysolu+map_name):
        #         os.makedirs(dir_mysolu+map_name)
        #         os.chmod(dir_mysolu+map_name, 0o700)
        #     # create a plotter for exp1
        #     exp1_my_solu_plotter = plotter.plotter(dir_mysolu+map_name+'/')
        #
        #     self.exp_1_my_solution(dir_mysolu+map_name+'/', maximum_drone_energy_capacity, step_size,  dataset, sensor_map, hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data, real_mean_of_sensors, num_of_estimation_data, exp1_my_solu_plotter,size_data_collection,
        #                             drone_commu_rate,mse_file_name, sensor_length_file_name)
        #
        #     # ----------------------------Exp1 Baseline_ML ------------------------------------------------------------------
        #
        #
        #     if not os.path.exists(dir_baseline_ml+map_name):
        #         os.makedirs(dir_baseline_ml+map_name)
        #         os.chmod(dir_baseline_ml+map_name, 0o700)
        #     exp1_baseline_ml_plotter = plotter.plotter(dir_baseline_ml+map_name+'/')
        #     self.exp1_ML_baseline(dir_baseline_ml+map_name+'/', maximum_drone_energy_capacity, step_size,  dataset, sensor_map,
        #                           hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data, num_of_estimation_data,
        #                           exp1_baseline_ml_plotter, size_data_collection, drone_commu_rate, mse_file_name, sensor_length_file_name)
        #
        #     # ----------------------------Exp1 Baseline_fs ------------------------------------------------------------------
        #
        #     if not os.path.exists(dir_baseline_fs+map_name):
        #         os.makedirs(dir_baseline_fs+map_name)
        #         os.chmod(dir_baseline_fs+map_name, 0o700)
        #     exp1_baseline_fs_plotter = plotter.plotter(dir_baseline_fs+map_name+'/')
        #     self.exp1_FS_baseline(dir_baseline_fs+map_name+'/', maximum_drone_energy_capacity, step_size, dataset, sensor_map,
        #                           hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data,
        #                           num_of_estimation_data, exp1_baseline_fs_plotter, size_data_collection, drone_commu_rate,
        #                           mse_file_name, sensor_length_file_name)
        #
        # # after the all the maps are finished, calculate the average mse and std for the final plots
        # # average sensors being selected, and average trip length
        # dirs_solu=[dir_mysolu,dir_baseline_fs, dir_baseline_ml]
        # mse_metric= self.config['Exp_1']['mse_matric_name']
        # selected_sensor_metric=self.config['Exp_1']['selected_sensor_matric_name']
        # self.compute_avg_std_for_each_solu_exp1(dirs_solu, sensor_maps, mse_file_name, mse_metric)
        # self.compute_avg_std_for_each_solu_exp1(dirs_solu, sensor_maps, sensor_length_file_name, selected_sensor_metric)
        # #make the final plot of the metric with three solutions
        # matrics = [mse_metric, selected_sensor_metric]
        # drone_energy_capacity_list=np.arange(0, maximum_drone_energy_capacity,step_size)
        # for metric in matrics:
        #     avgs={}
        #     stds={}
        #     for dir_solu in dirs_solu:
        #         solu_name= dir_solu.split('/')[1]
        #         avg=np.loadtxt("%s/%s_avg.txt" %(dir_solu,metric))
        #         std=np.loadtxt("%s/%s_std.txt" %(dir_solu,metric))
        #         avgs[solu_name]=avg
        #         stds[solu_name] = std
        #
        #     self.plotter.plot_metrics_with_all_solutions(metric, drone_energy_capacity_list, avgs, stds)

























