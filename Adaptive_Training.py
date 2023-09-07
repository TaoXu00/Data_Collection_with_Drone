import math

import Data_Generator_and_Loader as DG
import numpy as np
import scipy.stats
class Adaptive_Training:
    def calculate_mean_var(self, var_old, mean_old, k, realization):
        '''
        this function is to update the mean and variance of the unknown random variable when a new realization is observed.
        mean is updated with (mean_old * t)/(t+1);
        variance if updated with var(x) = E(x^2)-(E(x))^2
        :param var_old: the variance of current time t
        :param mean_old: the mean of current time t
        :param t: the current time t
        :param realization: a new observation at time t+1
        :return: updated mean and variance
        '''
        mean=(mean_old*(k-1)+realization)/k
        var=((var_old+(mean_old)**2)*(k-1)+realization**2)/(k) - mean**2
        return mean, var

    def Adaptive_setting_of_the_interval(self, Dataset):
        k_min= np.infty
        mean_theta=Dataset[0]
        var_theta=0
        Dataset = np.delete(Dataset, 0)
        k=1
        print("k=%s mean=%s var=%s" % (k, mean_theta, var_theta))
        abs_error_mean = [abs(mean_theta-650)/650]
        abs_error_std=[abs(0-100)/100]
        while k_min > k:
            k=k+1
            realization=Dataset[0]
            print("----------------------------------")
            print("k=%s mean=%s var=%s" % (k, mean_theta, var_theta))
            print("new observation: %f" %(realization))
            Dataset=np.delete(Dataset, 0)
            mean_theta,var_theta=self.calculate_mean_var(var_theta, mean_theta, k, realization)
            std_theta=math.sqrt(var_theta)
            print("calculated mean: %f, std %f" % (mean_theta, std_theta))
            abs_error_mean.append(abs(mean_theta-650)/650)
            abs_error_std.append(abs(std_theta-100)/650)
            gamma=0.004
            print("gamma = %f" %(gamma))
            error=gamma*mean_theta
            t_value=scipy.stats.t.ppf(q=1-0.05/2, df=k-1)
            print("t_value=%f" %(t_value))
            print("error= %f" %(error))
            print("t_value/error = %f" %(t_value/error))
            k_min=(t_value/error)**2 * var_theta
            print("k_min = %f" %(k_min))
        return abs_error_mean, abs_error_std

    def Adaptive_learning_of_the_interval_multi_var(self, Dataset, num_traing_data):
        '''
        This function is for the training phase, where it learns the mean and covariance matrix of the variables.
        :param Dataset: the target Dataset, every column represents one variable, every row represents one realization of all variables.
        :return: it returns the learned mean vector and the covariance matrix of the Dataset,
                 2D array of mean and variance to track the values in the learning process for each sensor,
                 and k, the index and data has been used for training
        '''
        num_rows, n_var=Dataset.shape
        k_expect=np.ones(n_var)*np.inf
        k=0
        mean_theta=np.zeros(n_var)
        var_theta = np.zeros(n_var)
        cov_theta=np.zeros((n_var,n_var))
        error_mean=np.zeros(n_var)
        #error_var=np.zeros(n_var)
        total_error_mean=[]
        #total_error_std=[]
        k_min=np.amax(k_expect)
        mean_2d=[]
        var_2d=[]
        #gamma is not using in the current version
        # gammas=[0.002, 0.004, 0.006, 0.008, 0.01]
        # gamma=gammas[0]
        #while k_min > k:
        while k <= num_traing_data:
            #if (k%200) ==0:
            #    gamma=gammas[int(k/200)]
            k=k+1
            error_mean = np.zeros(n_var)
            error_std = np.zeros(n_var)
            x=Dataset[k-1]
            for n in range(n_var):
                realization=x[n]  # take a new realization
                mean_est,var_est=self.calculate_mean_var(var_theta[n], mean_theta[n], k, realization)
                mean_theta[n]= mean_est
                var_theta[n] =var_est
                if k ==1:
                    mean_2d.append([mean_est])
                    var_2d.append([var_est])
                else:
                    mean_2d[n].append(mean_est)
                    var_2d[n].append(var_est)
                #  error=gamma*mean_theta[n]
                # if(k>1):
                #     t_value=scipy.stats.t.ppf(q=0.975, df=k-1)
                #     k_min=(t_value/error)**2 * var_theta[n]
                #     k_expect[n]=k_min
            total_error_mean.append(error_mean)
            #total_error_std.append(error_std)
            # calculate the covariance matrix of the learned mean and variance
            for i in range(n_var):
                for j in range(n_var):
                    cov_theta[i][j] = (cov_theta[i][j]*(k-1)+(x[i] - mean_theta[i])*(x[j] - mean_theta[j]))/k
            k_min=np.amax(k_expect)
        #print("%d iterations to achieve gamma %s" % (k, gamma))
        return  mean_theta, cov_theta, k, mean_2d, var_2d







