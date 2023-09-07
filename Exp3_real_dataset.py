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
class Exp3_real_dataset:
    def __init__(self, config):
        self.config = config

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


    def exp_3_my_solution(self,dir_mysolu, maximum_drone_energy_capacity, step_size,  dataset, sensor_map,
                           hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data,
                           num_of_estimation_data, exp3_my_solu_plotter,size_data_collection, drone_commu_rate,
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
                    exp3_my_solu_plotter.plot_tour_map(sensor_map, tour, tour_id, optimal_dis, optimal_energy_cost)
                    tour_id += 1
            averaged_mse_varying_drone_capabilities = np.average(mse_along_time_total, axis=1)
            exp3_my_solu_plotter.plot_averaged_mse_vary_drone_capability(drone_capacity_list,
                                                                 averaged_mse_varying_drone_capabilities,
                                                                 expected_mse_list)
            exp3_my_solu_plotter.plot_selected_sensors_vary_drone_capability(drone_capacity_list, num_of_selected_nodes)
            exp3_my_solu_plotter.plot_tour_length_vary_drone_capability(drone_capacity_list, optimal_dis_list)
            exp3_my_solu_plotter.plot_energy_cost_vary_drone_capability(drone_capacity_list, optimal_energy_cost_list)
            np.savetxt("%s/%s" % (dir_mysolu, sensor_length_file_name), num_of_selected_nodes)
            np.savetxt("%s/%s" % (dir_mysolu, mse_file_name),
                       averaged_mse_varying_drone_capabilities)



    def exp3_ML_baseline(self, dir_baseline_ml, maximum_drone_energy_capacity, step_size,  dataset, sensor_map, hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data, num_of_estimation_data, exp3_baseline_plotter, size_data_collection, drone_commu_rate,
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
                    exp3_baseline_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance, optimal_energy_cost)
                tour_id += 1
        # exp3_baseline_plotter.plotter.plot_mse_with_varying_drone_cap_for_different_ML_models(total_dist_list,
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
                    exp3_baseline_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance, optimal_energy_cost)
                    tour_id += 1
            exp3_baseline_plotter.plot_mse_with_varying_drone_capabilities(drone_capacity_list, mse_list_total)
            exp3_baseline_plotter.plot_selected_sensors_vary_drone_capability(drone_capacity_list, num_of_selected_nodes)
            exp3_baseline_plotter.plot_tour_length_vary_drone_capability(drone_capacity_list, optimal_dis_list)
            exp3_baseline_plotter.plot_energy_cost_vary_drone_capability(drone_capacity_list, optimal_energy_cost_list)
            np.savetxt("%s/%s" %(dir_baseline_ml,sensor_length_file_name), num_of_selected_nodes)
            np.savetxt("%s/%s" %(dir_baseline_ml,mse_file_name), mse_list_total)

    def exp3_FS_baseline(self,dir_baseline_fs, maximum_drone_energy_capacity, step_size, dataset, sensor_map,
                              hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data,
                              num_of_estimation_data, exp3_baseline_fs_plotter, size_data_collection, drone_commu_rate,
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
                    exp3_baseline_fs_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance,
                                                        optimal_energy_cost)
                tour_id += 1
                # exp3_baseline_plotter.plotter.plot_mse_with_varying_drone_cap_for_different_ML_models(total_dist_list,
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
                # here we only have one model GBR
                mse_list_total.append(total_mse)
                if len(optimal_tour) != 0:
                    exp3_baseline_fs_plotter.plot_tour_map(sensor_map, optimal_tour, tour_id, optimal_distance,
                                                        optimal_energy_cost)
                    tour_id += 1
            exp3_baseline_fs_plotter.plot_mse_with_varying_drone_capabilities(drone_capacity_list, mse_list_total)
            exp3_baseline_fs_plotter.plot_selected_sensors_vary_drone_capability(drone_capacity_list, num_of_selected_nodes)
            exp3_baseline_fs_plotter.plot_tour_length_vary_drone_capability(drone_capacity_list, optimal_dis_list)
            exp3_baseline_fs_plotter.plot_energy_cost_vary_drone_capability(drone_capacity_list, optimal_energy_cost_list)
            np.savetxt("%s/%s" % (dir_baseline_fs, sensor_length_file_name), num_of_selected_nodes)

            np.savetxt("%s/%s" % (dir_baseline_fs, mse_file_name), mse_list_total)


    def exp_3(self, dir_exp3):
        #todo run_exp3 and check the results
        # This set of experiments varying the drone capability
        ## read the parameters of exp3
        path_sensor_map = self.config['Exp_3']['dir_sensor_map']
        dir_dataset = self.config['Exp_3']['dir_dataset']
        dir_mysolu= self.config['Exp_3']['dir_mysolu']
        dir_baseline_ml = self.config['Exp_3']['dir_baseline_ml']
        dir_baseline_fs = self.config['Exp_3']['dir_baseline_fs']
        maximum_drone_energy_capacity =int(self.config['Exp_3']['maximum_energy_capacity'])
        step_size = int(self.config['Exp_3']['step_size'])
        hovering_energy_per_unit = float(self.config['Drone']['unit_hovering_energy'])
        flying_energy_per_unit = float(self.config['Drone']['unit_flying_energy'])
        number_of_training_data= int(self.config['Exp_3']['num_of_training_data'])
        num_of_estimation_data= int(self.config['Exp_3']['num_of_estimation_data'])
        size_data_collection = int(self.config['Exp_3']['size_data_collection'])
        drone_commu_rate = int(self.config['Drone']['comm_rate'])
        mse_file_name = self.config['Exp_3']['mse_file_name']
        sensor_length_file_name=self.config['Exp_3']['sensor_length_file_name']

        self.plotter = plotter.plotter(dir_exp3)

        # load dataset
        print(dir_dataset)
        dataset=np.loadtxt(dir_dataset)
        # # load and plot sensor map
        with open(path_sensor_map, 'r') as json_file:

            sensor_map = json.load(json_file)

        #----------------------------Exp3 my solution ------------------------------------------------------------------
        if not os.path.exists(dir_mysolu):
            os.makedirs(dir_mysolu)
            os.chmod(dir_mysolu, 0o700)
        # create a plotter for exp3
        exp3_my_solu_plotter = plotter.plotter(dir_mysolu+'/')

        self.exp_3_my_solution(dir_mysolu+'/', maximum_drone_energy_capacity, step_size,  dataset, sensor_map, hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data, num_of_estimation_data, exp3_my_solu_plotter,size_data_collection,
                                drone_commu_rate,mse_file_name, sensor_length_file_name)

        # # ----------------------------Exp3 Baseline_ML ------------------------------------------------------------------


        if not os.path.exists(dir_baseline_ml):
            os.makedirs(dir_baseline_ml)
            os.chmod(dir_baseline_ml, 0o700)
        exp3_baseline_ml_plotter = plotter.plotter(dir_baseline_ml+'/')
        self.exp3_ML_baseline(dir_baseline_ml+'/', maximum_drone_energy_capacity, step_size,  dataset, sensor_map,
                              hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data, num_of_estimation_data,
                              exp3_baseline_ml_plotter, size_data_collection, drone_commu_rate, mse_file_name, sensor_length_file_name)

        #----------------------------Exp3 Baseline_fs ------------------------------------------------------------------

        if not os.path.exists(dir_baseline_fs):
            os.makedirs(dir_baseline_fs)
            os.chmod(dir_baseline_fs, 0o700)
        exp3_baseline_fs_plotter = plotter.plotter(dir_baseline_fs+'/')
        self.exp3_FS_baseline(dir_baseline_fs+'/', maximum_drone_energy_capacity, step_size, dataset, sensor_map,
                              hovering_energy_per_unit, flying_energy_per_unit, number_of_training_data,
                              num_of_estimation_data, exp3_baseline_fs_plotter, size_data_collection, drone_commu_rate,
                              mse_file_name, sensor_length_file_name)

        # after the all the maps are finished, calculate the average mse and std for the final plots
        # average sensors being selected, and average trip length
        dirs_solu=[dir_mysolu,dir_baseline_fs, dir_baseline_ml]
        mse_metric= self.config['Exp_3']['mse_matric_name']
        selected_sensor_metric=self.config['Exp_3']['selected_sensor_matric_name']

        #make the final plot of the metric with three solutions
        matrics = [mse_metric, selected_sensor_metric]
        drone_energy_capacity_list=np.arange(0, maximum_drone_energy_capacity,step_size)
        for metric in matrics:
            avgs={}
            for dir_solu in dirs_solu:
                solu_name= dir_solu.split('/')[1]
                if metric==mse_metric:
                    data=np.loadtxt("%s/mse_varying_drone_capabilities.txt" %(dir_solu))
                elif metric==selected_sensor_metric:
                    data=np.loadtxt(("%s/sensor_length.txt" %(dir_solu)))
                avgs[solu_name]=data

            self.plotter.plot_metrics_with_all_solutions_real_dataset(metric, drone_energy_capacity_list, avgs)
        self.plot_sensor_graph(sensor_map, self.plotter, "sensor_map")
























