import math
import os
import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
import json
import pyproj
from scipy.stats import gaussian_kde
import scipy.stats as stats

def kl_divergence_multivariate(data_p, data_q):
    """
    Compute the Kullback-Leibler (KL) divergence between two multivariate datasets
    data_p and data_q. Assumes each row represents a sample.

    Parameters:
    data_p (numpy array): Array of samples from dataset P (each row is a sample).
    data_q (numpy array): Array of samples from dataset Q (each row is a sample).

    Returns:
    float: KL divergence D(P || Q).
    """
    # Estimate marginal densities for each variable

    kdes_p = [gaussian_kde(data_p[:, i]) for i in range(data_p.shape[1])]
    kdes_q = [gaussian_kde(data_q[:, i]) for i in range(data_q.shape[1])]

    # Define a function to compute KL divergence for two KDEs
    def kl_divergence(p, q):
        return np.sum(p* np.log(p / q))

    # Evaluate KDEs at the same points
    min_data = np.min(np.concatenate((data_p, data_q), axis=0), axis=0)
    max_data = np.max(np.concatenate((data_p, data_q), axis=0), axis=0)
    points = [np.linspace(min_data[i], max_data[i], 100) for i in range(data_p.shape[1])]

    # Compute KL divergence D(P || Q) for each variable and sum
    kl_div = 0.0
    epsilon = 1e-8
    for i in range(data_p.shape[1]):
        pdf_p = kdes_p[i].evaluate(points[i])
        pdf_q = kdes_q[i].evaluate(points[i])
        pdf_p=pdf_p+epsilon
        pdf_q=pdf_q+epsilon
        kl_div += kl_divergence(pdf_p, pdf_q)

    return kl_div

def plot_KL_divergence(x, kls_JGD, kls_EXP, kls_uni):
    plt.figure()
    plt.rcParams.update(
        {'font.size': 25, 'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large', 'axes.titlesize': 'x-large'})
    colors = ['cornflowerblue', 'goldenrod', 'forestgreen', 'firebrick', 'purple']
    linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid', '--', 'dotted']
    markers = ["s", "^", "*", "p", "X"]
    plt.grid(True)
    x=[1, 2, 3, 4, 5]
    yticks=[2, 4, 6, 8, 10]
    kls_JGD=[0.408250201, 0.410761685, 0.442558813, 0.443480764, 0.463652796]
    kls_EXP=[2.037165347, 5.298643387, 7.010969475, 8.025435553, 9.049510114 ]
    kls_uni=[1.143445049, 3.092326738, 4.750767361, 5.971547847, 6.65799991]
    plt.plot(x, kls_JGD, label='JDG', color=colors[0], marker=markers[0], markersize=12, linewidth=2)
    plt.plot(x, kls_EXP, label='Exponential', color=colors[1], marker=markers[1], markersize=12, linewidth=2)
    plt.plot(x, kls_uni, label='Uniform', color=colors[2], marker=markers[2], markersize=12,linewidth=2)
    plt.xticks(x)
    plt.yticks(yticks)
    plt.xlabel("# of sensors")
    plt.ylabel(" KL Divergence")
    plt.legend(fontsize=20, loc='upper left')
    plt.savefig('KL Divergence of different distribution_solar_radiation.png', format="PNG",
                bbox_inches='tight')
    plt.close()


#take 500 samples from the dataset DataMatrix_313.txt
data_path='Dataset/solar_radiation_dataset/DataMatrix_316_real.txt'
#data_path='Dataset/solar_radiation_dataset/synthetic_dataset_solar_radiation.txt'
dataset=np.loadtxt(data_path)
data_selected_rows= dataset[:500, :] #first 500 rows
# [0, 4 , 8, 12, 17]
data=data_selected_rows[:, [0, 4, 8, 9, 12]]
#data=data_selected_rows
num_samples=500
print("shape of the real dataset:", data.shape)
np.savetxt("Dataset/Dataset for distribution evaluation/empirical_data.txt", data, delimiter=' ')
mean_vector=np.mean(data, axis=0)
# print(mean_vector)
cov=np.cov(data.T)
lows=np.min(data,axis=0)
highs=np.max(data, axis=0) # Upper bounds of the uniform distributions for each variable
# print(lows)
# print(highs)
num_var=len(mean_vector)


#generate the theoretical distribution of JGD
corr_matrix=np.corrcoef(data.T)
theoretical_JGD=np.random.multivariate_normal(mean_vector, cov, num_samples)
print("Shape of the generated exponential array:", theoretical_JGD.shape)
np.savetxt("Dataset/Dataset for distribution evaluation/theoretical_JGD.txt", theoretical_JGD, delimiter=' ')

#generate the theoretical distribution of the exponential
theoretical_exponential=np.zeros((num_samples, num_var))
theoretical_exponential=np.random.exponential(scale=mean_vector, size=(num_samples, num_var))
np.savetxt("Dataset/Dataset for distribution evaluation/theoretical_exp.txt", theoretical_exponential, delimiter=' ')
print("Shape of the generated exponential array:", theoretical_exponential.shape)

#generate the theoretical distribution of the uniform
low = 0  # Lower bound of the uniform distribution
theoretical_uniform = np.zeros((num_samples, num_var))
for i in range(num_var):
    low=lows[i]
    high = highs[i]
    theoretical_uniform[:, i] = np.random.uniform(low, high, size=num_samples)
np.savetxt("Dataset/Dataset for distribution evaluation/theoretical_uniform.txt", theoretical_uniform, delimiter=' ')
# Print the shape of the generated array
print("Shape of the generated uniform array:", theoretical_uniform.shape)

#compute the KL divergence varying the number of sensors
dict_kls={}
kls_JGD = []
kls_EXP=[]
kls_uni=[]
for i in range(2, num_var+1, 1):
    empirical_data=data[:, :i]
    data_JGD=theoretical_JGD[:, :i]
    data_exp=theoretical_exponential[:, :i]
    data_uni=theoretical_uniform[:,:i]
    print("Shape of the KL Empirical array:", empirical_data.shape)
    print("Shape of the KL JGD array:", data_JGD.shape)
    print("Shape of the KL exp array:", data_exp.shape)
    print("Shape of the KL uniform array:", data_uni.shape)
    kl_JGD= kl_divergence_multivariate(empirical_data, data_JGD)
    kls_JGD.append(kl_JGD)
    kl_EXP = kl_divergence_multivariate(empirical_data,data_exp)
    kls_EXP.append(kl_EXP)
    kl_uni = kl_divergence_multivariate(empirical_data, data_uni)
    kls_uni.append(kl_uni)
    x=np.arange(2, num_var+1, 1)
plot_KL_divergence(x, kls_JGD,kls_EXP, kls_uni)






