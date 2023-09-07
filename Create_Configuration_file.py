import configparser

config =configparser.ConfigParser()
config['Dir'] = {
            'dir_dataset' :  'Dataset/VW_2015_Jan_April_32_sensors_hourly_synthetic/water_content_hourly.txt',
            'dir_sensor_map' : 'Dataset/sensors/CAF_sensors.json',
            'dir_exp1' : 'exp1/',
            'dir_exp2': 'exp2/',
            'dir_exp3': 'exp3/'
    }
config['Run'] = {
    'experiments': 'Exp2'
}
config['Drone'] = {
            'unit_hovering_energy' : 35,  #joulesPerSecond
            'unit_flying_energy' : 10,   #joulesPerMeter
             'comm_rate': 25   #Mbps
    }
config['Exp_1'] = {
            'dir_dataset' :  'Dataset/VW_2015_Jan_April_32_sensors_hourly_synthetic/water_content_hourly.txt',
            'dir_sensor_maps' :  'exp1/sensor_maps/',
            'dir_sensor_map' : 'Dataset/sensors/CAF_sensors.json',
            'dir_mysolu' :  'exp1/mysolu/',
            'dir_baseline_ml' : 'exp1/baseline_ml/',
            'dir_baseline_fs' : 'exp1/baseline_fs/',
            'dir_real_mean_of_sensors' : 'Dataset/VW_2015_Jan_April_32_sensors_hourly_synthetic/mean_vec.txt',
            'mse_file_name' : 'mse_varying_drone_capabilities.txt',
            'sensor_length_file_name' : 'sensor_length.txt',
            'maximum_drone_capacity' : 4000,
            'maximum_energy_capacity': 43000,
            'step_size': 2000,
            'num_of_training_data' : 1000,
            'num_of_estimation_data': 500,
            'size_data_collection': 50,
            'map_x_scale': 850,
            'map_y_scale': 550,
            'map_num': 5,
            'mse_matric_name': 'mse',
            'selected_sensor_matric_name': 'num_sensor'
    }

config['Exp_2'] = {
            'dir_dataset' :  'Dataset/VW_2015_Jan_April_32_sensors_hourly_synthetic/water_content_hourly.txt',
            'dir_sensor_maps' :  'exp2/sensor_maps/',
            'dir_sensor_map' : 'Dataset/sensors/CAF_sensors.json',
            'dir_mysolu' :  'exp2/mysolu/',
            'dir_baseline_ml' : 'exp2/baseline_ml/',
            'dir_baseline_fs' : 'exp2/baseline_fs/',
            'dir_real_mean_of_sensors' : 'Dataset/VW_2015_Jan_April_32_sensors_hourly_synthetic/mean_vec.txt',
            'mse_file_name' : 'mse_varying_drone_capabilities.txt',
            'sensor_length_file_name' : 'sensor_length.txt',
            'maximum_drone_capacity' : 4000,
            'drone_energy_capacity': 25000,
            'maximum_num_of_training_data' : 1001,
            'step_size': 200,
            'num_of_estimation_data': 500,
            'size_data_collection': 50,
            'map_x_scale': 850,
            'map_y_scale': 550,
            'map_num': 5,
            'mse_matric_name': 'mse',
            'selected_sensor_matric_name': 'num_sensor'
    }

config['Exp_3'] = {
            'dir_dataset' :  'solar_radiation_dataset/DataMatrix_313.txt',
            'dir_sensor_map' : 'solar_radiation_dataset/sensor_map.json',
            'dir_mysolu' :  'exp3/mysolu/',
            'dir_baseline_ml' : 'exp3/baseline_ml/',
            'dir_baseline_fs' : 'exp3/baseline_fs/',
            'mse_file_name' : 'mse_varying_drone_capabilities.txt',
            'sensor_length_file_name' : 'sensor_length.txt',
            'maximum_energy_capacity': 40000,
            'step_size': 2000,
            'num_of_training_data' : 1000,
            'num_of_estimation_data': 500,
            'size_data_collection': 50,
            'mse_matric_name': 'mse',
            'selected_sensor_matric_name': 'num_sensor'
    }

with open('config.ini', 'w') as config_file:
            config.write(config_file)
