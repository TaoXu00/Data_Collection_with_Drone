import math
import os
import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
import json
import pyproj

def generate_map_of_dataset_online_and_cleanup_the_data():
    # Coordinates for two points
    #todo
    map={}
    # Define the Mercator projection
    mercator_proj = pyproj.Proj(proj='utm',zone='10', ellps='WGS84')
    min_esting=math.inf
    min_northing=math.inf
    with open('Dataset/solar_radiation_dataset/Solar_Radiation_sensor_map', 'r') as file:
        for line in file:
            # Assuming each line contains valid JSON
            values=line.split(',')
            dict_loc={}
            dict_loc['Location']=values[0]
            lat=float(values[1])
            lon=float(values[2])
            x,y=mercator_proj(lon,lat)
            dict_loc['Easting'] = x -  727554
            dict_loc['Northing'] = y -4138131
            # if x<min_esting:
            #     min_esting=x
            # if y<min_northing:
            #     min_northing=y
            map[values[0]]=dict_loc

        with open('%s/map.json' %('solar_radiation_dataset'), 'w') as json_file:
            json_file.write(json.dumps(map, indent=4))
        print("min_easting: %f" %(min_esting))
        print("min_northing: %f" %(min_northing))

    file_path = 'Dataset/solar_radiation_dataset/DataMatrix_319.txt'
    data_rf = pd.read_csv(file_path, sep='\t')
    cleaned_df= data_rf.dropna(how='any')
    cleaned_df = cleaned_df.iloc[:, :-1].reset_index(drop=True)
    cleaned_df.to_csv(file_path,sep='\t', index=False)
    print(cleaned_df.shape)

def generate_sensor_map_of_collected_solar_reidation_sensor(file,output_file):
    map = {}
    # Define the Mercator projection
    mercator_proj = pyproj.Proj(proj='utm', zone='10', ellps='WGS84')
    min_esting = math.inf
    min_northing = math.inf
    min_esting=3916420
    min_northing=4990244
    with open(file, 'r') as file:
        for line in file:
            # Assuming each line contains valid JSON
            values = line.split(',')
            dict_loc = {}
            dict_loc['Location'] = values[0]
            lat = float(values[1])
            lon = float(values[2])
            x, y = mercator_proj(lon, lat)
            dict_loc['Easting'] = x - min_esting
            dict_loc['Northing'] = y - min_northing
            # if x<min_esting:
            #     min_esting=x
            # if y<min_northing:
            #     min_northing=y
            map[values[0]] = dict_loc
        with open(output_file, 'w') as json_file:
            json_file.write(json.dumps(map, indent=4))

filepath='Dataset/CPS_Solar_Radiation_Dataset/21_sensors/CPS_solar_radiation_sensor_map_21_sensors.txt'
output_filepath='Dataset/CPS_Solar_Radiation_Dataset/21_sensors/CPS_solar_radiation_sensor_map_21_sensors.json'
generate_sensor_map_of_collected_solar_reidation_sensor(filepath,output_filepath)









