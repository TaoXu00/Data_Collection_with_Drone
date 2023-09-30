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
class Exp4:
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

    def compute_avg_std_for_each_solu_exp4(self, dirs_solu, sensor_maps, metric_file_name, metric):
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


    def exp_4_my_solution(self, dir_solu, num_sensor_scalability_dir_list, drone_energy_capacity, hovering_energy_per_unit,
                          flying_energy_per_unit, num_of_training_data,  num_of_estimation_data,
                          size_data_collection, drone_commu_rate,mse_file_name, sensor_length_file_name):
        sensor_num_list = []
        expected_mse_list_varying_sensor_numbers = []
        avg_mse_total_varying_sensor_numbers = []
        std_mse_total_varying_sensor_numbers = []

        avg_num_of_selected_nodes_varying_sensor_numbers = []
        std_num_of_selected_nodes_varying_sensor_numbers =[]

        avg_optimal_dis_list_varying_sensor_numbers = []
        std_optimal_dis_list_varying_sensor_numbers =[]

        avg_optimal_energy_cost_list_varying_sensor_numbers = []
        std_optimal_energy_cost_list_varying_sensor_numbers = []
        #iterate each sensor_num
        for dir_sensor_num in num_sensor_scalability_dir_list:
            # split the dir_sensor_num with '/' and take the third part
            sensor_num_name = dir_sensor_num.split('/')[2]
            sensor_num = int(sensor_num_name.split('_')[0])
            sensor_num_list.append(sensor_num)
            # make the dir for each sensor number in the solution directory
            dir_sensor_num_within_solu_dir = dir_solu + str(sensor_num) + '/'
            if not os.path.exists(dir_sensor_num_within_solu_dir):
                os.makedirs(dir_sensor_num_within_solu_dir)
                os.chmod(dir_sensor_num_within_solu_dir, 0o700)
            # perform solution for each map
            sensor_map_files = [f for f in os.listdir(num_sensor_scalability_dir_list[0]) if f.endswith('.json')]
            expected_mse_list = []
            mse_total = []
            num_of_selected_nodes = []
            optimal_dis_list = []
            optimal_energy_cost_list = []
            for sensor_map_file in sensor_map_files:
                # Create the directory if it doesn't exist
                map_name, file_extention = os.path.splitext(sensor_map_file)
                map_id_dir = dir_sensor_num_within_solu_dir + map_name + '/'
                if not os.path.exists(map_id_dir):
                    os.makedirs(map_id_dir)
                    os.chmod(map_id_dir, 0o700)
                # load and plot sensor map
                sensor_map = json.load(open(dir_sensor_num+ sensor_map_file))
                # load dataset
                dataset = np.loadtxt(dir_sensor_num + 'dataset_' + map_name + '.txt')
                # create a plotter for exp1
                exp4_solu_plotter_num_sensor = plotter.plotter(map_id_dir + '/')
                with open('%spath_plan_mysolu.txt' % (map_id_dir), 'w') as f:
                    f.write(
                        'Path Plan with different drone capabilities\n')  ## track the trajectory planned for each drone capacity
                    tour_id = 0
                    drone = Drone.Drone(drone_energy_capacity, hovering_energy_per_unit, flying_energy_per_unit, drone_commu_rate)

                    solu = mySolution.mySolution(exp4_solu_plotter_num_sensor, dataset, drone, sensor_map, num_of_training_data,
                                                   num_of_estimation_data, size_data_collection)
                    mse_along_time, expect_mse, selected_sensors, tour, optimal_dis, optimal_energy_cost, vars_rank = solu.run()

                    mse = np.average(mse_along_time)
                    f.write('***************************************************************\n')
                    f.write('vars rank : %s\n' % (vars_rank))
                    f.write('drone_constraint: %d\n' % (drone_energy_capacity))
                    f.write('%d sensors are selected \n' % (len(selected_sensors)))
                    f.write('optimal tour: %s\n' % (tour))
                    f.write('length of the tour %f\n' % (optimal_dis))
                    f.write('optimal energy cost %f\n' % (optimal_energy_cost))
                    f.write('mse %f \n' %(mse))
                    f.close()
                if len(tour) != 0:
                    exp4_solu_plotter_num_sensor.plot_tour_map(sensor_map, tour, tour_id, optimal_dis, optimal_energy_cost)
                    tour_id += 1
                mse_total.append(mse)
                num_of_selected_nodes.append(len(selected_sensors))
                expected_mse_list.append(expect_mse)
                optimal_dis_list.append(optimal_dis)
                optimal_energy_cost_list.append(optimal_energy_cost)
                f.close()


            # mse
            avg_mse = np.average(mse_total)
            std_mse = np.std(mse_total)
            avg_mse_total_varying_sensor_numbers.append(avg_mse)
            std_mse_total_varying_sensor_numbers.append(std_mse)

            # num of selected sensors
            avg_num_of_selected_nodes = np.average(num_of_selected_nodes)
            std_num_of_selected_nodes = np.std(num_of_selected_nodes)
            avg_num_of_selected_nodes_varying_sensor_numbers.append(avg_num_of_selected_nodes)
            std_num_of_selected_nodes_varying_sensor_numbers.append(std_num_of_selected_nodes)


            # optimal distance
            avg_optimal_dis= np.average(optimal_dis_list)
            std_optimal_dis= np.std(optimal_dis_list)
            avg_optimal_dis_list_varying_sensor_numbers.append(avg_optimal_dis)
            std_optimal_dis_list_varying_sensor_numbers.append(std_optimal_dis)

            # optimal energy cost
            avg_optimal_energy_cost = np.average(optimal_energy_cost_list)
            std_optimal_energy_cost = np.average(optimal_energy_cost_list)
            avg_optimal_energy_cost_list_varying_sensor_numbers.append(avg_optimal_energy_cost)
            std_optimal_energy_cost_list_varying_sensor_numbers.append(std_optimal_energy_cost)

            with open('%sstatistics.txt' % (dir_sensor_num_within_solu_dir), 'w') as f:
                f.write('avg_mse: %f\n' %(avg_mse))  ## track the trajectory planned for each drone capacity
                f.write('std_mse: %f\n' %(std_mse))
                f.write('avg_num_of_selected_nodes: %f\n' % (avg_num_of_selected_nodes))
                f.write('std_num_of_selected_nodes: %f\n' %(std_num_of_selected_nodes))
                f.write('avg_optimal_dis: %f \n' %(avg_optimal_dis))
                f.write('std_optimal_dis: %f \n' %(std_optimal_dis))
                f.write('avg_optimal_energy_cost: %f \n' % (avg_optimal_dis))
                f.write('std_optimal_energy_cost: %f \n' % (std_optimal_dis))
                f.close()

        np.savetxt("%s%s" % (dir_solu, 'avg_' + sensor_length_file_name),
                   avg_num_of_selected_nodes_varying_sensor_numbers)
        np.savetxt("%s%s" % (dir_solu, 'std_' + sensor_length_file_name),
                   std_num_of_selected_nodes_varying_sensor_numbers)

        np.savetxt("%s%s" % (dir_solu, 'avg_' + mse_file_name), avg_mse_total_varying_sensor_numbers)
        np.savetxt("%s%s" % (dir_solu, 'std_' + mse_file_name), std_mse_total_varying_sensor_numbers)
        exp4_solu_plotter= plotter.plotter(dir_solu + '/')
        exp4_solu_plotter.plot_averaged_mse_varying_sensor_num(
                    sensor_num_list, avg_mse_total_varying_sensor_numbers,
                    std_mse_total_varying_sensor_numbers)

        exp4_solu_plotter.plot_selected_sensors_varying_sensor_num(
                    sensor_num_list, avg_num_of_selected_nodes_varying_sensor_numbers,
                    std_num_of_selected_nodes_varying_sensor_numbers)


    def exp4_ML_baseline(self, dir_solu, num_sensor_scalability_dir_list, drone_energy_capacity, hovering_energy_per_unit,
                          flying_energy_per_unit, num_of_training_data,  num_of_estimation_data,
                          size_data_collection, drone_commu_rate,mse_file_name, sensor_length_file_name):
        sensor_num_list = []
        expected_mse_list_varying_sensor_numbers = []
        avg_mse_total_varying_sensor_numbers = []
        std_mse_total_varying_sensor_numbers = []

        avg_num_of_selected_nodes_varying_sensor_numbers = []
        std_num_of_selected_nodes_varying_sensor_numbers = []

        avg_optimal_dis_list_varying_sensor_numbers = []
        std_optimal_dis_list_varying_sensor_numbers = []

        avg_optimal_energy_cost_list_varying_sensor_numbers = []
        std_optimal_energy_cost_list_varying_sensor_numbers = []
        # iterate each sensor_num
        for dir_sensor_num in num_sensor_scalability_dir_list:
            # split the dir_sensor_num with '/' and take the third part
            sensor_num_name = dir_sensor_num.split('/')[2]
            sensor_num = int(sensor_num_name.split('_')[0])
            sensor_num_list.append(sensor_num)
            # make the dir for each sensor number in the solution directory
            dir_sensor_num_within_solu_dir = dir_solu + str(sensor_num) + '/'
            if not os.path.exists(dir_sensor_num_within_solu_dir):
                os.makedirs(dir_sensor_num_within_solu_dir)
                os.chmod(dir_sensor_num_within_solu_dir, 0o700)
            # perform solution for each map
            sensor_map_files = [f for f in os.listdir(num_sensor_scalability_dir_list[0]) if f.endswith('.json')]
            expected_mse_list = []
            mse_total = []
            num_of_selected_nodes = []
            optimal_dis_list = []
            optimal_energy_cost_list = []
            for sensor_map_file in sensor_map_files:
                # Create the directory if it doesn't exist
                map_name, file_extention = os.path.splitext(sensor_map_file)
                map_id_dir = dir_sensor_num_within_solu_dir + map_name + '/'
                if not os.path.exists(map_id_dir):
                    os.makedirs(map_id_dir)
                    os.chmod(map_id_dir, 0o700)
                # load and plot sensor map
                sensor_map = json.load(open(dir_sensor_num + sensor_map_file))
                # load dataset
                dataset = np.loadtxt(dir_sensor_num + 'dataset_' + map_name + '.txt')
                # create a plotter for exp1
                exp4_solu_plotter_num_sensor = plotter.plotter(map_id_dir + '/')
                with open('%spath_plan_bs_ml.txt' % (map_id_dir), 'w') as f:
                    f.write(
                        'Path Plan with different drone capabilities\n')  ## track the trajectory planned for each drone capacity
                    tour_id = 0
                    drone = Drone.Drone(drone_energy_capacity, hovering_energy_per_unit, flying_energy_per_unit,
                                        drone_commu_rate)

                    ml_bs = ML_BS.ML_Baseline(map_id_dir+'/', drone, sensor_map, dataset, size_data_collection,
                                              num_of_training_data,
                                              num_of_estimation_data)
                    selected_sensors, total_mse_for_all_models, tour, optimal_dis, optimal_energy_cost, vars_rank = ml_bs.train_model()
                    #mse_along_time, expect_mse, selected_sensors, tour, optimal_dis, optimal_energy_cost, vars_rank = solu.run()

                    mse = total_mse_for_all_models[0]
                    f.write('***************************************************************\n')
                    f.write('vars rank : %s\n' % (vars_rank))
                    f.write('drone_constraint: %d\n' % (drone_energy_capacity))
                    f.write('%d sensors are selected \n' % (len(selected_sensors)))
                    f.write('optimal tour: %s\n' % (tour))
                    f.write('length of the tour %f\n' % (optimal_dis))
                    f.write('optimal energy cost %f\n' % (optimal_energy_cost))
                    f.write('mse %f \n' % (mse))
                    f.close()
                if len(tour) != 0:
                    exp4_solu_plotter_num_sensor.plot_tour_map(sensor_map, tour, tour_id, optimal_dis,
                                                               optimal_energy_cost)
                    tour_id += 1
                mse_total.append(mse)
                num_of_selected_nodes.append(len(selected_sensors))
                optimal_dis_list.append(optimal_dis)
                optimal_energy_cost_list.append(optimal_energy_cost)
                f.close()

            # mse
            avg_mse = np.average(mse_total)
            std_mse = np.std(mse_total)
            avg_mse_total_varying_sensor_numbers.append(avg_mse)
            std_mse_total_varying_sensor_numbers.append(std_mse)

            # num of selected sensors
            avg_num_of_selected_nodes = np.average(num_of_selected_nodes)
            std_num_of_selected_nodes = np.std(num_of_selected_nodes)
            avg_num_of_selected_nodes_varying_sensor_numbers.append(avg_num_of_selected_nodes)
            std_num_of_selected_nodes_varying_sensor_numbers.append(std_num_of_selected_nodes)

            # optimal distance
            avg_optimal_dis = np.average(optimal_dis_list)
            std_optimal_dis = np.std(optimal_dis_list)
            avg_optimal_dis_list_varying_sensor_numbers.append(avg_optimal_dis)
            std_optimal_dis_list_varying_sensor_numbers.append(std_optimal_dis)

            # optimal energy cost
            avg_optimal_energy_cost = np.average(optimal_energy_cost_list)
            std_optimal_energy_cost = np.average(optimal_energy_cost_list)
            avg_optimal_energy_cost_list_varying_sensor_numbers.append(avg_optimal_energy_cost)
            std_optimal_energy_cost_list_varying_sensor_numbers.append(std_optimal_energy_cost)

            with open('%sstatistics.txt' % (dir_sensor_num_within_solu_dir), 'w') as f:
                f.write('avg_mse: %f\n' % (avg_mse))  ## track the trajectory planned for each drone capacity
                f.write('std_mse: %f\n' % (std_mse))
                f.write('avg_num_of_selected_nodes: %f\n' % (avg_num_of_selected_nodes))
                f.write('std_num_of_selected_nodes: %f\n' % (std_num_of_selected_nodes))
                f.write('avg_optimal_dis: %f \n' % (avg_optimal_dis))
                f.write('std_optimal_dis: %f \n' % (std_optimal_dis))
                f.write('avg_optimal_energy_cost: %f \n' % (avg_optimal_dis))
                f.write('std_optimal_energy_cost: %f \n' % (std_optimal_dis))
            f.close()

        np.savetxt("%s%s" % (dir_solu, 'avg_' + sensor_length_file_name),
                   avg_num_of_selected_nodes_varying_sensor_numbers)
        np.savetxt("%s%s" % (dir_solu, 'std_' + sensor_length_file_name),
                   std_num_of_selected_nodes_varying_sensor_numbers)

        np.savetxt("%s%s" % (dir_solu, 'avg_' + mse_file_name), avg_mse_total_varying_sensor_numbers)
        np.savetxt("%s%s" % (dir_solu, 'std_' + mse_file_name), std_mse_total_varying_sensor_numbers)
        exp4_solu_plotter = plotter.plotter(dir_solu + '/')
        exp4_solu_plotter.plot_averaged_mse_varying_sensor_num(
            sensor_num_list, avg_mse_total_varying_sensor_numbers,
            std_mse_total_varying_sensor_numbers)

        exp4_solu_plotter.plot_selected_sensors_varying_sensor_num(
            sensor_num_list, avg_num_of_selected_nodes_varying_sensor_numbers,
            std_num_of_selected_nodes_varying_sensor_numbers)

    def exp4_FS_baseline(self, dir_solu, num_sensor_scalability_dir_list, drone_energy_capacity,
                                      hovering_energy_per_unit,
                                      flying_energy_per_unit,
                                      maximum_num_of_training_data,
                                      num_of_estimation_data,
                                      size_data_collection,
                                      drone_commu_rate, mse_file_name, sensor_length_file_name):

        sensor_num_list = []
        avg_mse_total_varying_sensor_numbers = []
        std_mse_total_varying_sensor_numbers = []

        avg_num_of_selected_nodes_varying_sensor_numbers = []
        std_num_of_selected_nodes_varying_sensor_numbers = []

        avg_optimal_dis_list_varying_sensor_numbers = []
        std_optimal_dis_list_varying_sensor_numbers = []

        avg_optimal_energy_cost_list_varying_sensor_numbers = []
        std_optimal_energy_cost_list_varying_sensor_numbers = []
        # iterate each sensor_num
        for dir_sensor_num in num_sensor_scalability_dir_list:
            # split the dir_sensor_num with '/' and take the third part
            sensor_num_name = dir_sensor_num.split('/')[2]
            sensor_num = int(sensor_num_name.split('_')[0])
            sensor_num_list.append(sensor_num)
            # make the dir for each sensor number in the solution directory
            dir_sensor_num_within_solu_dir = dir_solu + str(sensor_num) + '/'
            if not os.path.exists(dir_sensor_num_within_solu_dir):
                os.makedirs(dir_sensor_num_within_solu_dir)
                os.chmod(dir_sensor_num_within_solu_dir, 0o700)
            # perform solution for each map
            sensor_map_files = [f for f in os.listdir(num_sensor_scalability_dir_list[0]) if f.endswith('.json')]
            expected_mse_list = []
            mse_total = []
            num_of_selected_nodes = []
            optimal_dis_list = []
            optimal_energy_cost_list = []
            for sensor_map_file in sensor_map_files:
                # Create the directory if it doesn't exist
                map_name, file_extention = os.path.splitext(sensor_map_file)
                map_id_dir = dir_sensor_num_within_solu_dir + map_name + '/'
                if not os.path.exists(map_id_dir):
                    os.makedirs(map_id_dir)
                    os.chmod(map_id_dir, 0o700)
                # load and plot sensor map
                sensor_map = json.load(open(dir_sensor_num + sensor_map_file))
                # load dataset
                dataset = np.loadtxt(dir_sensor_num + 'dataset_' + map_name + '.txt')
                # create a plotter for exp1
                exp4_solu_plotter_num_sensor = plotter.plotter(map_id_dir + '/')
                with open('%spath_plan_bs_ml.txt' % (map_id_dir), 'w') as f:
                    f.write(
                        'Path Plan with different drone capabilities\n')  ## track the trajectory planned for each drone capacity
                    tour_id = 0

                    drone = Drone.Drone(drone_energy_capacity, hovering_energy_per_unit, flying_energy_per_unit,
                                        drone_commu_rate)

                    fs_bs = FS.Feature_selection_based_baseline(dir_solu, drone, sensor_map, dataset,
                                                                size_data_collection,
                                                                maximum_num_of_training_data, num_of_estimation_data)

                    vars_rank = fs_bs.calculate_feature_importance(dataset)

                    total_mse, selected_sensors, tour, optimal_dis, optimal_energy_cost, vars_rank = fs_bs.train_model(
                        vars_rank, drone)

                    mse = total_mse
                    f.write('***************************************************************\n')
                    f.write('vars rank : %s\n' % (vars_rank))
                    f.write('drone_constraint: %d\n' % (drone_energy_capacity))
                    f.write('%d sensors are selected \n' % (len(selected_sensors)))
                    f.write('optimal tour: %s\n' % (tour))
                    f.write('length of the tour %f\n' % (optimal_dis))
                    f.write('optimal energy cost %f\n' % (optimal_energy_cost))
                    f.write('mse %f \n' % (mse))
                    f.close()
                if len(tour) != 0:
                    exp4_solu_plotter_num_sensor.plot_tour_map(sensor_map, tour, tour_id, optimal_dis,
                                                               optimal_energy_cost)
                    tour_id += 1
                mse_total.append(mse)
                num_of_selected_nodes.append(len(selected_sensors))
                optimal_dis_list.append(optimal_dis)
                optimal_energy_cost_list.append(optimal_energy_cost)
                f.close()

            # mse
            avg_mse = np.average(mse_total)
            std_mse = np.std(mse_total)
            avg_mse_total_varying_sensor_numbers.append(avg_mse)
            std_mse_total_varying_sensor_numbers.append(std_mse)

            # num of selected sensors
            avg_num_of_selected_nodes = np.average(num_of_selected_nodes)
            std_num_of_selected_nodes = np.std(num_of_selected_nodes)
            avg_num_of_selected_nodes_varying_sensor_numbers.append(avg_num_of_selected_nodes)
            std_num_of_selected_nodes_varying_sensor_numbers.append(std_num_of_selected_nodes)

            # optimal distance
            avg_optimal_dis = np.average(optimal_dis_list)
            std_optimal_dis = np.std(optimal_dis_list)
            avg_optimal_dis_list_varying_sensor_numbers.append(avg_optimal_dis)
            std_optimal_dis_list_varying_sensor_numbers.append(std_optimal_dis)

            # optimal energy cost
            avg_optimal_energy_cost = np.average(optimal_energy_cost_list)
            std_optimal_energy_cost = np.average(optimal_energy_cost_list)
            avg_optimal_energy_cost_list_varying_sensor_numbers.append(avg_optimal_energy_cost)
            std_optimal_energy_cost_list_varying_sensor_numbers.append(std_optimal_energy_cost)

            with open('%sstatistics.txt' % (dir_sensor_num_within_solu_dir), 'w') as f:
                f.write('avg_mse: %f\n' % (avg_mse))  ## track the trajectory planned for each drone capacity
                f.write('std_mse: %f\n' % (std_mse))
                f.write('avg_num_of_selected_nodes: %f\n' % (avg_num_of_selected_nodes))
                f.write('std_num_of_selected_nodes: %f\n' % (std_num_of_selected_nodes))
                f.write('avg_optimal_dis: %f \n' % (avg_optimal_dis))
                f.write('std_optimal_dis: %f \n' % (std_optimal_dis))
                f.write('avg_optimal_energy_cost: %f \n' % (avg_optimal_dis))
                f.write('std_optimal_energy_cost: %f \n' % (std_optimal_dis))
            f.close()

        np.savetxt("%s%s.txt" % (dir_solu, 'avg_' + sensor_length_file_name),
                   avg_num_of_selected_nodes_varying_sensor_numbers)
        np.savetxt("%s%s.txt" % (dir_solu, 'std_' + sensor_length_file_name),
                   std_num_of_selected_nodes_varying_sensor_numbers)

        np.savetxt("%s%s" % (dir_solu, 'avg_' + mse_file_name), avg_mse_total_varying_sensor_numbers)
        np.savetxt("%s%s" % (dir_solu, 'std_' + mse_file_name), std_mse_total_varying_sensor_numbers)
        exp4_solu_plotter = plotter.plotter(dir_solu + '/')
        exp4_solu_plotter.plot_averaged_mse_varying_sensor_num(
            sensor_num_list, avg_mse_total_varying_sensor_numbers,
            std_mse_total_varying_sensor_numbers)

        exp4_solu_plotter.plot_selected_sensors_varying_sensor_num(
            sensor_num_list, avg_num_of_selected_nodes_varying_sensor_numbers,
            std_num_of_selected_nodes_varying_sensor_numbers)




    def exp_4(self, dir_exp4):
        # This set of experiments varying the drone capability
        # run the program with different maps, then plot the average values with error bar.
        dir_maps = self.config['Exp_4']['dir_sensor_maps']  #exp4/sensor_maps/
        map_x_scale = int(self.config['Exp_2']['map_x_scale'])
        map_y_scale = int(self.config['Exp_2']['map_y_scale'])
        ## read the parameters of exp4
        map_num = int(self.config['Exp_4']['map_num'])
        dir_mysolu = self.config['Exp_4']['dir_mysolu']
        dir_baseline_ml = self.config['Exp_4']['dir_baseline_ml']
        dir_baseline_fs = self.config['Exp_4']['dir_baseline_fs']

        drone_energy_capacity= int(self.config['Exp_4']['drone_energy_capacity'])
        step_size = int(self.config['Exp_4']['step_size'])
        hovering_energy_per_unit = float(self.config['Drone']['unit_hovering_energy'])
        flying_energy_per_unit = float(self.config['Drone']['unit_flying_energy'])
        maximum_num_of_training_data = int(self.config['Exp_4']['maximum_num_of_training_data'])
        num_of_estimation_data = int(self.config['Exp_4']['num_of_estimation_data'])
        size_data_collection = int(self.config['Exp_4']['size_data_collection'])
        drone_commu_rate = int(self.config['Drone']['comm_rate'])
        mse_metric = self.config['Exp_4']['mse_matric_name']
        selected_sensor_metric = self.config['Exp_4']['selected_sensor_matric_name']

        #mse_file_name = self.config['Exp_4']['mse_file_name']
        #sensor_length_file_name = self.config['Exp_4']['sensor_length_file_name']

        num_clusters = int(self.config['Exp_4']['num_clusters'])

        template_dataset_file_path = self.config['Exp_4']['template_dataset_file_path']

        if not os.path.exists(dir_maps):
            os.makedirs(dir_maps)
            os.chmod(dir_maps, 0o700)
        self.plotter = plotter.plotter(dir_exp4)
        num_sensor_scalability_dir_list=[]
        dir_100_sensor= dir_exp4
        points_per_cluster=20
        map_num_100_sensors= 1
        self.DGL.generate_clustered_maps(map_x_scale, map_y_scale, dir_100_sensor, map_num_100_sensors, self.plotter, num_clusters,
                                         points_per_cluster, template_dataset_file_path)
        # generate the sensor maps with increasing sensors [20, 40, 60, 80, 100]
        #load the sensor_map_100 and the dataset
        with open('map_0.json', 'r') as json_file:
            map_json_100 = json.load(json_file)
        dataset= np.load("dataset_map_0.txt")
        for i in range(map_num):
        #for i in range(map_num):
            clusters={}
            for j in range(num_clusters):
                clusters[j]=np.arange(j*points_per_cluster, (j+1)*points_per_cluster)
            points_per_sensor_num={}
            for sensor_num in range(step_size, 101,step_size):
                points_per_sensor_num[sensor_num] =[]
                if sensor_num != step_size:
                    points_per_sensor_num[sensor_num]=points_per_sensor_num[sensor_num-step_size].copy()  #take the previous choices
                dir_sensor_num= dir_maps + str(sensor_num) + '_sensors/'
                num_sensor_scalability_dir_list.append(dir_sensor_num)
                if not os.path.exists(dir_sensor_num):
                    os.makedirs(dir_sensor_num)
                    os.chmod(dir_sensor_num, 0o700)
                #points_per_cluster= int(sensor_num/int(num_clusters))
                num_sensors_to_select_per_cluster= int(step_size/num_clusters)
                for cluster in clusters:
                    #samples= random.sample(clusters[cluster], points_per_cluster)
                    samples= np.random.choice(clusters[cluster], num_sensors_to_select_per_cluster, replace=False)
                    points_per_sensor_num[sensor_num].extend(samples)
                    new_cluster = [x for x in clusters[cluster] if x not in samples]
                    clusters[cluster] = new_cluster
                #create the map (json, png, dataset_map_0.txt)
                #sort the sensors
                sensor_ids= points_per_sensor_num[sensor_num].sort()
                str_sensor_ids= str(sensor_ids)
                # Generate data points using K-Means clustering
                sensors= {id: map_json_100[id] for id in str_sensor_ids if id in map_json_100}
                location = {}
                location['Easting'] = 0
                location['Northing'] = 0
                sensors['Depot'] = location
                new_sensor_map = json.dumps(sensor_map, indent=4)
                with open('%smap_%d.json' % (dir_sensor_num, i), 'w') as json_file:
                    json_file.write(new_sensor_map)
                with open('%smap_%d.json' % (dir_sensor_num, i), 'r') as json_file:
                    data = json_file.read()
                    sensor_map = json.loads(data)
                self.plot_sensor_graph(sensor_map, plotter, '%smap_%d' % (dir_sensor_num, i))
                path_dataset = '%sdataset_map_%d.txt' % (dir_sensor_num, i)
                dataset_map= dataset[:, sensor_ids]
                np.savetxt(path_dataset,dataset_map)
            print(points_per_sensor_num)
                #randomly select the points from
                #self.DGL.generate_clustered_maps(map_x_scale, map_y_scale, dir_sensor_num, map_num, self.plotter, num_clusters, points_per_cluster,template_dataset_file_path)
            # for each solution, perform the experiment from 20 to 100.
            dirs_solu = [dir_mysolu, dir_baseline_fs, dir_baseline_ml]
            #dirs_solu = [dir_baseline_fs]
            # for dir_solu in dirs_solu:
            #     #make the solution dir
            #     if not os.path.exists(dir_solu):
            #         os.makedirs(dir_solu)
            #         os.chmod(dir_solu, 0o700)
            #     if dir_solu == dir_mysolu:
            #         self.exp_4_my_solution(dir_solu, num_sensor_scalability_dir_list, drone_energy_capacity, hovering_energy_per_unit,
            #                   flying_energy_per_unit, maximum_num_of_training_data,  num_of_estimation_data,
            #                   size_data_collection, drone_commu_rate,mse_metric, selected_sensor_metric)
            #     elif dir_solu == dir_baseline_ml:
            #         self.exp4_ML_baseline(dir_solu, num_sensor_scalability_dir_list, drone_energy_capacity, hovering_energy_per_unit,
            #         flying_energy_per_unit, maximum_num_of_training_data, num_of_estimation_data,
            #         size_data_collection, drone_commu_rate, mse_metric, selected_sensor_metric)
            #
            #     elif dir_solu == dir_baseline_fs:
            #         self.exp4_FS_baseline(dir_solu, num_sensor_scalability_dir_list, drone_energy_capacity,
            #                               hovering_energy_per_unit,
            #                               flying_energy_per_unit, maximum_num_of_training_data, num_of_estimation_data,
            #                               size_data_collection, drone_commu_rate, mse_metric, selected_sensor_metric)



        # make the final plot of the metric with three solutions
        matrics = [mse_metric, selected_sensor_metric]
        sensor_num_list = np.arange(step_size, 101,step_size)
        for metric in matrics:
            avgs = {}
            stds = {}
            for dir_solu in dirs_solu:
                solu_name = dir_solu.split('/')[1]
                avg = np.loadtxt("%s/avg_%s" % (dir_solu, metric))
                std = np.loadtxt("%s/std_%s" % (dir_solu, metric))
                avgs[solu_name] = avg
                stds[solu_name] = std
            self.plotter.plot_metrics_with_all_solutions_exp4(metric,sensor_num_list, avgs, stds)

