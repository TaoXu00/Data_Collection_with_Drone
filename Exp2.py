import math
import random
import numpy as np
import ML_Baseline as ML_BS
import plotter as plotter
import Feature_Selection_based_Baseline as FS
import mySolution as mySolution
import json
import Drone as Drone
import os
import Data_Generator_and_Loader as DGL
class Exp2:
    def __init__(self, config):
        self.config = config
        self.DGL= DGL.Data_Generator_and_Loader()

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

    def compute_avg_std_for_each_solu_exp2(self, dirs_solu, sensor_maps, metric_file_name, metric):
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
            self.plot_sensor_graph(sensor_map, plotter, '%smap_%d' % (dir_maps, i))


    def exp_2_my_solution(self,dir_mysolu, drone_energy_capacity, step_size,  dataset, sensor_map, hovering_energy_per_unit,
                          flying_energy_per_unit, maximum_num_of_training_data, real_mean_of_sensors, num_of_estimation_data,
                          exp2_my_solu_plotter,size_data_collection, drone_commu_rate,mse_file_name, sensor_length_file_name):
        expected_mse_list = []
        mse_along_time_total = []
        num_of_selected_nodes = []
        optimal_dis_list = []
        optimal_energy_cost_list = []
        num_of_training_dataset_size_list=range(200, maximum_num_of_training_data, step_size)
        with open('%spath_plan_mysolution.txt' %(dir_mysolu), 'w') as f:
            f.write('Path Plan with different drone capabilities\n') ## track the trajectory planned for each drone capacity
            tour_id=0
            #todo
            #check this part and see whether it is correct for calculate the avg and std of the mse and sensor length
            for num_of_training_data in num_of_training_dataset_size_list:
                drone = Drone.Drone(drone_energy_capacity, hovering_energy_per_unit, flying_energy_per_unit, drone_commu_rate)
                mySolu=mySolution.mySolution(plotter, dataset, drone, sensor_map, num_of_training_data, num_of_estimation_data, size_data_collection)
                mse_along_time, expect_mse, selected_sensors, tour, optimal_dis, optimal_energy_cost, vars_rank =mySolu.run()
                mse_along_time_total.append(mse_along_time)
                f.write('***************************************************************\n')
                f.write('vars rank : %s\n' %(vars_rank))
                f.write('drone_constraint: %d\n' % (drone_energy_capacity))
                f.write('%d sensors are selected \n' % (len(selected_sensors)))
                f.write('optimal tour: %s\n' % (tour))
                f.write('length of the tour %f\n' % (optimal_dis))
                f.write('optimal energy cost %f\n' %(optimal_energy_cost))
                num_of_selected_nodes.append(len(selected_sensors))
                expected_mse_list.append(expect_mse)
                optimal_dis_list.append(optimal_dis)
                optimal_energy_cost_list.append(optimal_energy_cost)
                if len(tour) != 0:
                    exp2_my_solu_plotter.plot_tour_map(sensor_map, tour, tour_id, optimal_dis, optimal_energy_cost)
                    tour_id += 1
            averaged_mse_varying_training_dataset_size= np.average(mse_along_time_total, axis=1)
            exp2_my_solu_plotter.plot_averaged_mse_vary_training_dataset_size_with_expected_value(num_of_training_dataset_size_list,
                                                                 averaged_mse_varying_training_dataset_size,
                                                                 expected_mse_list)
            exp2_my_solu_plotter.plot_selected_sensors_varying_training_dataset_size(num_of_training_dataset_size_list, num_of_selected_nodes)
            exp2_my_solu_plotter.plot_tour_length_varying_training_dataset_size(num_of_training_dataset_size_list, optimal_dis_list)
            exp2_my_solu_plotter.plot_energy_cost_varying_training_dataset_size(num_of_training_dataset_size_list, optimal_energy_cost_list)
            np.savetxt("%s/%s" % (dir_mysolu, sensor_length_file_name), num_of_selected_nodes)
            np.savetxt("%s/%s" % (dir_mysolu, mse_file_name),
                       averaged_mse_varying_training_dataset_size)



    def exp2_ML_baseline(self, dir_baseline_ml, drone_energy_capacity, step_size,  dataset, sensor_map,
                                  hovering_energy_per_unit, flying_energy_per_unit, maximum_num_of_training_data, num_of_estimation_data,
                                  exp2_baseline_ml_plotter, size_data_collection, drone_commu_rate, mse_file_name, sensor_length_file_name):
        mse_list_total = []
        num_of_selected_nodes = []
        optimal_dis_list = []
        optimal_energy_cost_list = []
        training_dataset_size_list = range(200, maximum_num_of_training_data, step_size)
        with open('%spath_plan_baseline_ml.txt' % (dir_baseline_ml), 'w') as f:
            f.write(
                'Path Plan with different drone capabilities\n')  ## track the trajectory planned for each drone capacity
            tour_id = 0
            for training_dataset_size in training_dataset_size_list:
                drone = Drone.Drone(drone_energy_capacity, hovering_energy_per_unit, flying_energy_per_unit, drone_commu_rate)
                ml_bs = ML_BS.ML_Baseline(dir_baseline_ml, drone, sensor_map, dataset, size_data_collection, training_dataset_size,
                                          num_of_estimation_data)
                selected_vars,total_mse_for_all_models , optimal_tour, optimal_distance, optimal_energy_cost, vars_rank = ml_bs.train_model()
                if len(optimal_tour) != 0:
                    exp2_baseline_ml_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance, optimal_energy_cost)
                tour_id += 1

                f.write('***************************************************************\n')
                f.write('vars_rank : %s\n' %(vars_rank))
                f.write('drone_constraint: %d\n' % (drone_energy_capacity))
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
                    exp2_baseline_ml_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance, optimal_energy_cost)
                    tour_id += 1

            exp2_baseline_ml_plotter.plot_averaged_mse_vary_training_dataset_size(training_dataset_size_list,
                                                                              mse_list_total)
            exp2_baseline_ml_plotter.plot_selected_sensors_varying_training_dataset_size(training_dataset_size_list,
                                                                                     mse_list_total)
            exp2_baseline_ml_plotter.plot_tour_length_varying_training_dataset_size(training_dataset_size_list,
                                                                                mse_list_total)
            exp2_baseline_ml_plotter.plot_energy_cost_varying_training_dataset_size(training_dataset_size_list,
                                                                                mse_list_total)
            np.savetxt("%s/%s" %(dir_baseline_ml,sensor_length_file_name), num_of_selected_nodes)
            np.savetxt("%s/%s" %(dir_baseline_ml,mse_file_name), mse_list_total)


    def exp2_FS_baseline(self,dir_baseline_fs,drone_energy_capacity, step_size, dataset, sensor_map,
                              hovering_energy_per_unit, flying_energy_per_unit, maximum_number_of_training_data,
                              num_of_estimation_data, exp2_baseline_fs_plotter, size_data_collection, drone_commu_rate,
                              mse_file_name, sensor_length_file_name):

        mse_list_total = []
        num_of_selected_nodes = []
        optimal_dis_list = []
        optimal_energy_cost_list = []
        training_dataset_size_list = range(200, maximum_number_of_training_data, step_size)
        with open('%spath_plan_baseline_fs.txt' % (dir_baseline_fs), 'w') as f:
            f.write(
                'Path Plan with different drone capabilities\n')  ## track the trajectory planned for each drone capacity
            tour_id = 0
            drone = Drone.Drone(0, hovering_energy_per_unit, flying_energy_per_unit, drone_commu_rate)
            fs_bs = FS.Feature_selection_based_baseline(dir_baseline_fs, drone, sensor_map, dataset,
                                                        size_data_collection,
                                                        maximum_number_of_training_data, num_of_estimation_data)
            vars_rank = fs_bs.calculate_feature_importance(dataset)

            for training_dataseet_size in training_dataset_size_list:
                drone = Drone.Drone(drone_energy_capacity, hovering_energy_per_unit, flying_energy_per_unit, drone_commu_rate)
                fs_bs = FS.Feature_selection_based_baseline(dir_baseline_fs, drone, sensor_map, dataset,
                                                            size_data_collection,
                                                             training_dataseet_size, num_of_estimation_data)

                total_mse, selected_vars, optimal_tour, optimal_distance, optimal_energy_cost, vars_rank = fs_bs.train_model(vars_rank, drone)
                if len(optimal_tour) != 0:
                    exp2_baseline_fs_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance,
                                                        optimal_energy_cost)
                tour_id += 1
                # exp1_baseline_plotter.plotter.plot_mse_with_varying_drone_cap_for_different_ML_models(total_dist_list,
                #                                                              total_mse_list_for_all_ML_models_varying_drone_capability)

                f.write('***************************************************************\n')
                f.write('vars_rank: %s \n' %(vars_rank))
                f.write('drone_constraint: %d\n' % (drone_energy_capacity))
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
                    exp2_baseline_fs_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance,
                                                        optimal_energy_cost)
                    tour_id += 1
            exp2_baseline_fs_plotter.plot_mse_with_varying_drone_capabilities(training_dataset_size_list, mse_list_total)
            exp2_baseline_fs_plotter.plot_selected_sensors_vary_drone_capability(training_dataset_size_list, num_of_selected_nodes)
            exp2_baseline_fs_plotter.plot_tour_length_vary_drone_capability(training_dataset_size_list, optimal_dis_list)
            exp2_baseline_fs_plotter.plot_energy_cost_vary_drone_capability(training_dataset_size_list, optimal_energy_cost_list)
            np.savetxt("%s/%s" % (dir_baseline_fs, sensor_length_file_name), num_of_selected_nodes)
            np.savetxt("%s/%s" % (dir_baseline_fs, mse_file_name), mse_list_total)




    def exp_2(self, dir_exp2):
        # This set of experiments varying the drone capability
        # run the program with different maps, then plot the average values with error bar.
        dir_maps = self.config['Exp_2']['dir_sensor_maps']
        ## read the parameters of exp1
        dir_sensor_map = self.config['Exp_2']['dir_sensor_map']

        map_x_scale = int(self.config['Exp_2']['map_x_scale'])
        map_y_scale = int(self.config['Exp_2']['map_y_scale'])
        map_num = int(self.config['Exp_2']['map_num'])
        dir_dataset = self.config['Exp_2']['dir_dataset']
        dir_mysolu = self.config['Exp_2']['dir_mysolu']
        dir_baseline_ml = self.config['Exp_2']['dir_baseline_ml']
        dir_baseline_fs = self.config['Exp_2']['dir_baseline_fs']

        dir_real_mean_of_sensors = self.config['Exp_2']['dir_real_mean_of_sensors']
        maximum_drone_capacity = int(self.config['Exp_2']['maximum_drone_capacity'])
        drone_energy_capacity= int(self.config['Exp_2']['drone_energy_capacity'])
        step_size = int(self.config['Exp_2']['step_size'])
        hovering_energy_per_unit = float(self.config['Drone']['unit_hovering_energy'])
        flying_energy_per_unit = float(self.config['Drone']['unit_flying_energy'])
        maximum_num_of_training_data = int(self.config['Exp_2']['maximum_num_of_training_data'])
        num_of_estimation_data = int(self.config['Exp_2']['num_of_estimation_data'])
        size_data_collection = int(self.config['Exp_2']['size_data_collection'])
        drone_commu_rate = int(self.config['Drone']['comm_rate'])
        mse_file_name = self.config['Exp_2']['mse_file_name']
        sensor_length_file_name = self.config['Exp_2']['sensor_length_file_name']
        num_clusters = int(self.config['Exp_2']['num_clusters'])
        points_per_cluster= int(self.config['Exp_2']['points_per_cluster'])
        template_dataset_file_path = self.config['Exp_2']['template_dataset_file_path']
        if not os.path.exists(dir_maps):
            os.makedirs(dir_maps)
            os.chmod(dir_maps, 0o700)

        self.plotter = plotter.plotter(dir_exp2)
        #self.DGL.generate_clustered_maps(map_x_scale, map_y_scale, dir_maps, map_num, self.plotter, num_clusters, points_per_cluster,template_dataset_file_path)
        #self.generate_sensor_maps(map_x_scale, map_y_scale, dir_maps, dir_sensor_map, map_num, self.plotter)
        # List all maps in the directory
        files = os.listdir(dir_maps)
        sensor_maps = []
        for file in files:
            if file.endswith('.json'):
                sensor_maps.append(file)



        # load real mean of sensors
        real_mean_of_sensors = np.loadtxt(dir_real_mean_of_sensors)

        # Iterate through the files
        for sensor_map_file in sensor_maps:
            # Check if the current item is a file
            if os.path.isfile(os.path.join(dir_maps, sensor_map_file)):
                print("File:", sensor_map_file)

            # load and plot sensor map
            sensor_map = json.load(open(dir_maps+sensor_map_file))
            # load dataset
            name, ext = os.path.splitext(sensor_map_file)
            dataset = np.loadtxt(dir_maps + 'dataset_' + name + '.txt')

            # Create the directory if it doesn't exist
            map_name, file_extention = os.path.splitext(sensor_map_file)

            # #----------------------------Exp2 my solution ------------------------------------------------------------------
            # if not os.path.exists(dir_mysolu+map_name):
            #     os.makedirs(dir_mysolu+map_name)
            #     os.chmod(dir_mysolu+map_name, 0o700)
            # # create a plotter for exp1
            # exp2_my_solu_plotter = plotter.plotter(dir_mysolu+map_name+'/')
            #
            # self.exp_2_my_solution(dir_mysolu+map_name+'/', drone_energy_capacity, step_size,  dataset, sensor_map, hovering_energy_per_unit, flying_energy_per_unit, maximum_num_of_training_data, real_mean_of_sensors, num_of_estimation_data, exp2_my_solu_plotter,size_data_collection,
            #                         drone_commu_rate,mse_file_name, sensor_length_file_name)
            #
            # # # ----------------------------Exp2 Baseline_ML ------------------------------------------------------------------
            #
            # if not os.path.exists(dir_baseline_ml+map_name):
            #     os.makedirs(dir_baseline_ml+map_name)
            #     os.chmod(dir_baseline_ml+map_name, 0o700)
            # exp2_baseline_ml_plotter = plotter.plotter(dir_baseline_ml+map_name+'/')
            # self.exp2_ML_baseline(dir_baseline_ml+map_name+'/', drone_energy_capacity, step_size,  dataset, sensor_map,
            #                       hovering_energy_per_unit, flying_energy_per_unit, maximum_num_of_training_data, num_of_estimation_data,
            #                       exp2_baseline_ml_plotter, size_data_collection, drone_commu_rate, mse_file_name, sensor_length_file_name)
            #
            # # # ----------------------------Exp2 Baseline_fs ------------------------------------------------------------------
            #
            # if not os.path.exists(dir_baseline_fs+map_name):
            #     os.makedirs(dir_baseline_fs+map_name)
            #     os.chmod(dir_baseline_fs+map_name, 0o700)
            # exp1_baseline_fs_plotter = plotter.plotter(dir_baseline_fs+map_name+'/')
            # self.exp2_FS_baseline(dir_baseline_fs+map_name+'/', drone_energy_capacity, step_size, dataset, sensor_map,
            #                       hovering_energy_per_unit, flying_energy_per_unit, maximum_num_of_training_data,
            #                       num_of_estimation_data, exp1_baseline_fs_plotter, size_data_collection, drone_commu_rate,
            #                       mse_file_name, sensor_length_file_name)

        # after the all the maps are finished, calculate the average mse and std for the final plots
        # average sensors being selected, and average trip length
        dirs_solu = [dir_mysolu, dir_baseline_fs, dir_baseline_ml]
        sensor_maps = []
        for file in files:
            if file.endswith('.json'):
                sensor_maps.append(file)
        mse_metric = self.config['Exp_2']['mse_matric_name']
        selected_sensor_metric = self.config['Exp_2']['selected_sensor_matric_name']

        self.compute_avg_std_for_each_solu_exp2(dirs_solu, sensor_maps, mse_file_name, mse_metric)
        self.compute_avg_std_for_each_solu_exp2(dirs_solu, sensor_maps, sensor_length_file_name, selected_sensor_metric)
        # make the final plot of the metric with three solutions
        matrics = [mse_metric, selected_sensor_metric]
        training_dataset_size_list = np.arange(200, maximum_num_of_training_data, step_size)
        for metric in matrics:
            avgs = {}
            stds = {}
            for dir_solu in dirs_solu:
                solu_name = dir_solu.split('/')[1]
                avg = np.loadtxt("%s/%s_avg.txt" % (dir_solu, metric))
                std = np.loadtxt("%s/%s_std.txt" % (dir_solu, metric))
                avgs[solu_name] = avg
                stds[solu_name] = std
            self.plotter.plot_metrics_with_all_solutions_exp2(metric,training_dataset_size_list, avgs, stds)

