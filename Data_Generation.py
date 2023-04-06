import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
import os
class Data_generator:
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

    def single_normal_variable_data_loader(self):
        Dataset= np.loadtxt('Dataset/synthetic/single_normal_variable.txt')
        return Dataset

    def multi_normal_variable_data_loader(self):
        print("Loading Dataset...")
        Dataset= np.loadtxt('Dataset/synthetic/multi_normal_variables.txt')
        mean_vec = np.loadtxt("Dataset/synthetic/mean_vec.txt")
        cov = np.loadtxt("Dataset/synthetic/cov.txt")
        return Dataset, mean_vec, cov

