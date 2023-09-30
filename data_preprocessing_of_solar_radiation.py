import math
import os
import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
import json
import pyproj
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

# # Create a 2D graph
# plt.figure(figsize=(8, 6))
# plt.plot([x1, x2], [y1, y2], marker='o')
# plt.xlabel('Meters Easting')
# plt.ylabel('Meters Northing')
# plt.title('2D Graph of Google Coordinates in Meters')
# plt.grid()
# plt.show()

file_path = 'Dataset/solar_radiation_dataset/DataMatrix_319.txt'
data_rf = pd.read_csv(file_path, sep='\t')
cleaned_df= data_rf.dropna(how='any')
cleaned_df = cleaned_df.iloc[:, :-1].reset_index(drop=True)
cleaned_df.to_csv(file_path,sep='\t', index=False)
print(cleaned_df.shape)






