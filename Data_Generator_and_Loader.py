import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
import json
import os
class Data_Generator_and_Loader:
    def single_variable_normal_distribution_genrator(self, mean, sd,size):
        np.random.seed(10)
        sample = np.random.normal(mean, sd, size)
        np.savetxt('Dataset/synthetic/single_normal_variable.txt', sample)
        print("Dataset saved in 'Dataset/synthetic/single_normal_variable.txt'")
        plt.hist(sample)
        plt.title("Standard Normal Distribution")
        plt.show()
    def multi_variables_normal_distribution_generator(self, min_mean, max_mean, min_cov, max_cov, num):
        np.random.seed(10)
        mean_vec=np.random.randint(min_mean,max_mean,num)
        A=np.random.randint(min_cov,max_cov,size=[num,num])
        cov=np.dot(A, A.transpose())
        sample=np.random.multivariate_normal(mean_vec, cov, 1500)
        np.savetxt('Dataset/synthetic/multi_normal_variables.txt', sample)
        return mean_vec, cov

    def compute_elucidation_distance(self, point1_x, point1_y, point2_x, point2_y):
        return math.sqrt((point1_x-point2_x)**2 +(point1_y - point2_y)**2)

    def generate_synthetic_dataset_with_correlation_considered_with_distance(self, sensor_map,path_dataset):
        '''
        @param sensor_map: the json object of the sensor_map
        @return: a 2D matrix of the sensor readings
        '''
        n= len(sensor_map)-1
        distance_matrix=[[0 for _ in range(n)] for _ in range(n)]
        max_distance=0
        for i in range(n):
            for j in range(i,n):
                    point1_x= sensor_map[str(i)]['Easting']
                    point1_y= sensor_map[str(i)]['Northing']
                    point2_x = sensor_map[str(j)]['Easting']
                    point2_y = sensor_map[str(j)]['Northing']
                    distance_matrix[i][j] =self.compute_elucidation_distance(point1_x,point1_y, point2_x, point2_y)
                    distance_matrix[j][i] = distance_matrix[i][j]
                    if distance_matrix[i][j] > max_distance:
                        max_distance=distance_matrix[i][j]
        #divide the max_distance equally in 10 bins, and randomly sample the correlation
        bin_size = max_distance/10
        corr_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    corr_matrix[i][j]=1
                else:
                    num_bin=int(corr_matrix[i][j]/bin_size)
                    cov_va=np.random.uniform(num_bin/10, num_bin/10+1, 1)
                    corr_matrix[i][j] = cov_va
                    corr_matrix[j][i]=cov_va
        solar_radiation_dataset=np.loadtxt('solar_radiation_dataset/DataMatrix_313.txt')
        avg_vector = np.mean(solar_radiation_dataset, axis=0)
        std_vector= np.std(solar_radiation_dataset, axis=0)
        print(avg_vector)
        print(std_vector)
        mean_min= min(avg_vector)
        mean_max=max(avg_vector)
        std_min = min(std_vector)
        std_max= max(std_vector)
        mean_vector=[0]*n
        mean_vector[0]= np.random.uniform(mean_min,mean_max)
        for i in range(1, n):
            mean_vector[i]=mean_vector[0] + distance_matrix[0][i]/max_distance * (mean_max-mean_min)
        print("mean_vector:%s" %(mean_vector))
        std_vector=np.random.uniform(std_min, std_max, len(sensor_map) -1)
        cor_va_matrix = [[0 for _ in range(n)] for _ in range(n)]
        np.savetxt('covariance_matrix.txt', cor_va_matrix)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    cor_va_matrix[i][j] = std_vector[i]**2
                else:
                   cor_va_matrix[i][j] = corr_matrix[i][j] * std_vector[i] * std_vector[j]
                   cor_va_matrix[j][i] = cor_va_matrix[i][j]
        np.savetxt('covariance_matrix.txt', cor_va_matrix)
        sample_synthetic= np.random.multivariate_normal(mean_vector, cor_va_matrix, 10000)
        sample_synthetic[sample_synthetic<0] =0
        sample_synthetic = sample_synthetic/10
        # Find rows that contain at least one zero
        rows_with_zeros = np.any(sample_synthetic == 0, axis=1)

        # Use boolean indexing to remove rows with zeros
        sample_synthetic = sample_synthetic[~rows_with_zeros]
        # # Get the indices that would sort the first column
        # sorted_indices = np.argsort(sample_synthetic[:, 0])
        #
        # # Use these indices to sort the entire array
        # sample_synthetic = sample_synthetic[sorted_indices]

        #sample_synthetic=sample_synthetic[129:, :]
        np.savetxt(path_dataset, sample_synthetic)
        return distance_matrix
    def single_normal_variable_data_loader(self):
        Dataset= np.loadtxt('Dataset/synthetic/single_normal_variable.txt')
        return Dataset

    def multi_normal_variable_data_loader(self):
        print("Loading Dataset...")
        Dataset= np.loadtxt('Dataset/synthetic/multi_normal_variables.txt')
        mean_vec = np.loadtxt("Dataset/synthetic/mean_vec.txt")
        cov = np.loadtxt("Dataset/synthetic/cov.txt")
        return Dataset, mean_vec, cov



# Data_Generator_and_Loader= Data_Generator_and_Loader()
# path_map_json= 'Dataset/sensors/CAF_sensors.json'
# #create an synthetic dataset for solar_map
# #path_map_json='solar_radiation_dataset/sensor_map.json'
# with open(path_map_json, "r") as json_file:
#     map_json = json.load(json_file)
#
# synthetic_data_file='solar_radiation_dataset/synthetic_dataset_solar_radiation.txt'
# distance_matrix= Data_Generator_and_Loader.generate_synthetic_dataset_with_correlation_considered_with_distance(map_json, synthetic_data_file)
# #print(distance_matrix)






