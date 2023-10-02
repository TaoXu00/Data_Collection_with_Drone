import math

import numpy as np
import matplotlib.pyplot as plt
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

    def plot_sensor_graph(self, sensor_map, plotter, name):
        nodes = []
        coordinates = {}
        for sensor in sensor_map:
            # sensor_loc = sensors_json[sensor]['Location']
            nodes.append(sensor)
            coordinates[sensor] = (float(sensor_map[sensor]['Easting']), float(sensor_map[sensor]['Northing']))
        coordinates = coordinates.values()
        plotter.plot_sensor_map(coordinates, nodes, name)

    def generate_clustered_maps(self, map_x_scale, map_y_scale, dir_maps, map_num, plotter, num_clusters, points_per_cluster,template_dataset_file_path):
        for i in range(0, map_num):
            sensor_map={}

            # Generate random cluster centers
            cluster_centers = np.random.rand(num_clusters, 2) * [map_x_scale, map_y_scale]

            # Generate data points using K-Means clustering
            X = []
            desired_std=200
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
            #self.DGL.generate_synthetic_dataset_with_correlation_considered_with_distance(sensor_map,path_dataset)
            self.generate_synthetic_dataset_with_mean_cov_learned_from_real_solar_radiation(template_dataset_file_path,
                                                                                       path_dataset, num_clusters,
                                                                                       points_per_cluster)
            self.plot_sensor_graph(sensor_map, plotter, '%smap_%d' % (dir_maps,i))
    def generate_synthetic_dataset_with_mean_cov_learned_from_real_solar_radiation(self, template_dataset_file_path, path_dataset, number_clusters, points_per_cluster):
        #the template file is the solar radiation file, each of them has 5 clusters, and the first 4 cluster has 4 points, the last cluster has 3 points.
        template_dataset= np.loadtxt(template_dataset_file_path)
        list_of_clusters= []
        # 5 cluster, first 4 cluster has 4 points, the last one has 3 points in the template
        cluster1 = template_dataset[:, :4]
        cluster2 = template_dataset[:, 4:8]
        cluster3 = template_dataset[:, 8:12]
        cluster4 = template_dataset[:, 12:16]
        cluster5 = template_dataset[:, 16:19]
        list_of_clusters.append(cluster1)
        list_of_clusters.append(cluster2)
        list_of_clusters.append(cluster3)
        list_of_clusters.append(cluster4)
        list_of_clusters.append(cluster5)
        list_of_new_clusters=[]
        for cluster in list_of_clusters:
            if cluster.shape[1] < points_per_cluster:
                num_addional_points= points_per_cluster - cluster.shape[1]
                new_cluster=cluster
                for j in range(num_addional_points):
                    # Calculate the row-wise means
                    row_means = np.mean(cluster, axis=1, keepdims=True)
                    # Define the noise level
                    noise_level = 1  # You can adjust this value to control the amount of noise
                    # Generate random noise with the same shape as row_means
                    noise = np.random.normal(0, noise_level, size=row_means.shape)
                    # Add the noise to the row_means
                    row_means_with_noise = row_means + noise
                    # Create a new column with row means and noise
                    new_column = row_means_with_noise
                    # Concatenate the new column to the original array
                    new_cluster = np.concatenate((new_cluster, new_column), axis=1)
                list_of_new_clusters.append(new_cluster)
            else:
                list_of_new_clusters.append(cluster)
        # Concatenate the arrays along the last column (axis=1)
        new_dataset = np.concatenate([arr for arr in list_of_new_clusters], axis=1)
        # Calculate the mean (average) of the entire array
        mean = np.mean(new_dataset, axis=0)
        # Calculate the covariance matrix of the array
        cov_matrix = np.cov(new_dataset, rowvar=False)  # Set rowvar to False for column-wise covariance
        # Specify the number of samples you want to generate
        num_samples = 5000
        # Generate random samples from the multivariate Gaussian distribution
        synthetic_dataset= np.random.multivariate_normal(mean, cov_matrix, num_samples)
        print(path_dataset)
        print(synthetic_dataset.shape)
        np.savetxt(path_dataset,synthetic_dataset)


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
        solar_radiation_dataset=np.loadtxt('Dataset/solar_radiation_dataset/DataMatrix_313.txt')
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


#
# Data_Generator_and_Loader= Data_Generator_and_Loader()
# path_map_json= 'Dataset/sensor_maps/map_2.json'
# #create an synthetic dataset for solar_map
# #path_map_json='solar_radiation_dataset/sensor_map.json'
# with open(path_map_json, "r") as json_file:
#     map_json = json.load(json_file)
#
# synthetic_data_file='Dataset/sensor_maps/dataset_map_2_generated_based_on_synthetic_covariance.txt'
# distance_matrix= Data_Generator_and_Loader.generate_synthetic_dataset_with_correlation_considered_with_distance(map_json, synthetic_data_file)
# #print(distance_matrix)

# dataset= np.loadtxt('Dataset/solar_radiation_dataset/DataMatrix_313_synthetic.txt')
# dataset= np.insert(dataset, 17, dataset[:, 16], axis=1)
# print(dataset.shape)
# np.savetxt('Dataset/solar_radiation_dataset/DataMatrix_313_synthetic.txt', dataset)

# dataset= np.loadtxt('Dataset/solar_radiation_dataset/DataMatrix_313.txt')
# cov_matrix = np.cov(dataset, rowvar=False)
# mean_vector = np.mean(dataset, axis=0)
# # print(mean_vector)
# # print(cov_matrix)
# # sample= np.random.multivariate_normal(mean_vector, cov_matrix, 5000)
# # np.savetxt("Dataset/solar_radiation_dataset/DataMatrix_313_synthetic.txt", sample)








