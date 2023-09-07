import math
import numpy as np

import Exp3_real_dataset
import plotter as plotter
import Data_Preprocess as preprocess
import configparser
import os
import shutil
import Exp1
import Exp2
class main():
    def __init__(self, dir_plots, dir_dataset, dir_sensor_map):
        self.plotter=plotter.plotter(dir_plots)
        self.preprocess = preprocess.Data_Preprocess()
        self.dir_dataset=dir_dataset
        self.dir_sensor_map=dir_sensor_map

    def generate_dataset(self):
        #self.DG.single_normal_variable_genrator(650,100,100000)
        mean_vec, cov=self.DG.multi_variables_normal_distribution_generator(650, 800, 3, 8, 10)
        np.savetxt("Dataset/synthetic/mean_vec.txt",mean_vec)
        np.savetxt("Dataset/synthetic/cov.txt",cov)


    def plot_inference_error_of_both_ML_and_our_approach(self):
        total_dist_list = np.arange(0, self.drone_capability, self.step_size)
        averaged_mse_varying_drone_capabilities_inference_only = np.loadtxt(
            "results/averaged_mse_varying_drone_capabilities_inference_only.txt")
        averaged_mse_varying_drone_capabilities_distance_only = np.loadtxt(
            "results/averaged_mse_varying_drone_capabilities_distance_only.txt")
        averaged_mse_varying_drone_capabilities_with_both = np.loadtxt(
            "results/averaged_mse_varying_drone_capabilities_with_both.txt")
        mse_along_distance_for_variable_models = np.loadtxt("results/total_mse_list_for_all_ML_models_varying_drone_capability.txt")
        mse_feature_selection_for_GBR_model=np.loadtxt("results/total_mse_list_feature_selection_varying_drone_capability.txt")
        mse_feature_selection_of_GBR_model_baseline2=np.loadtxt("results/total_mse_list_feature_selection_varying_drone_capability_baseline2.txt")
        mse_feature_selection_baseline2_linear_regression = np.loadtxt(
            "results/total_mse_list_feature_selection_varying_drone_capability_baseline2_linear_regression.txt")
        log_averaged_mse_varying_drone_capabilities_inference_only = [-math.log(i) if i != 0 else 0  for i in
                                                                      averaged_mse_varying_drone_capabilities_inference_only]
        log_averaged_mse_varying_drone_capabilities_distance_only = [-math.log(i) if i != 0 else 0 for i in
                                                                     averaged_mse_varying_drone_capabilities_inference_only]
        log_averaged_mse_varying_drone_capabilities_with_both = [-math.log(i) if i != 0 else 0 for i in
                                                                 averaged_mse_varying_drone_capabilities_with_both]

        log_mse_along_distance_for_variable_models = []
        for i in np.arange(len(mse_along_distance_for_variable_models)):
            row = [-math.log(j) if j != 0 else 0 for j in mse_along_distance_for_variable_models[i]]
            log_mse_along_distance_for_variable_models.append(row)

        log_mse_feature_selection_for_GBR_model = [-math.log(i) if i != 0 else 0 for i in
                                                   mse_feature_selection_for_GBR_model]

        log_mse_feature_selection_of_GBR_model_baseline2 = [-math.log(i) if i != 0 else 0 for i in mse_feature_selection_of_GBR_model_baseline2]
        self.plotter.plot_mse_with_varying_drone_cap_for_different_ML_models_with_correlation_based(total_dist_list,
                                                                                                      mse_along_distance_for_variable_models,
                                                                                                      averaged_mse_varying_drone_capabilities_inference_only,
                                                                                                    averaged_mse_varying_drone_capabilities_distance_only,
                                                                                                      averaged_mse_varying_drone_capabilities_with_both,
                                                                                                    mse_feature_selection_for_GBR_model,
                                                                                                    mse_feature_selection_of_GBR_model_baseline2,
                                                                                                    mse_feature_selection_baseline2_linear_regression,
                                                                                                    "raw")

        # #self.plotter.plot_mse_with_varying_drone_cap_for_different_ML_models_with_correlation_based(total_dist_list,
        #                                                                                               log_mse_along_distance_for_variable_models,
        #                                                                                               log_averaged_mse_varying_drone_capabilities_inference_only,
        #                                                                                              log_averaged_mse_varying_drone_capabilities_distance_only,
        #                                                                                             log_averaged_mse_varying_drone_capabilities_with_both,
        #                                                                                             log_mse_feature_selection_for_GBR_model,
        #                                                                                             log_mse_feature_selection_of_GBR_model_baseline2,
        #                                                                                                 "log")



    def test_diferent_weight__calculation_solution(self):
        """
        This function is to test differebt weight calculation methods(distance only, inference error only, combined with distance and inferrence error  )
        @return:
        """
        mse_along_distance_for_variable_models = []  # one row corresponds to the mse of all the models
        heu="topw_update"
        total_dist_list=np.arange(0,self.drone_capability, self.step_size)
        averaged_mse_varying_drone_capabilities_inference_only, num_of_selected_nodes_inference_only, optimal_dis_list_inference_only = self.data_correlation.run(
            self.dataset, 0.01, self.sensors_map, 'inference only', heu, total_dist_list)
        averaged_mse_varying_drone_capabilities_distance_only, num_of_selected_nodes_distance_only, optimal_dis_list_distance_only = self.data_correlation.run(
           self.dataset, 0.01, self.sensors_map, 'distance only', heu, total_dist_list)
        averaged_mse_varying_drone_capabilities_with_both, num_of_selected_nodes_with_both, heu, optimal_dis_list_with_both = self.data_correlation.run(
           self.dataset, 0.01, self.sensors_map, 'both', total_dist_list)
        self.plotter.plot_comparison_vary_drone_capability(total_dist_list,
                                                            averaged_mse_varying_drone_capabilities_inference_only,
                                                            averaged_mse_varying_drone_capabilities_distance_only,
                                                            averaged_mse_varying_drone_capabilities_with_both,
                                                            "averaged MSE of inference (m\u00b3/m\u00b3)",
                                                            'comparison of inference error varying drone capabilities.png')
        self.plotter.plot_comparison_vary_drone_capability(total_dist_list, num_of_selected_nodes_inference_only,
                                                            num_of_selected_nodes_distance_only,
                                                            num_of_selected_nodes_with_both, "# of observation sensors",
                                                            'comparison of num sensor selection.png')
        self.plotter.plot_comparison_vary_drone_capability(total_dist_list, optimal_dis_list_inference_only,
                                                            optimal_dis_list_distance_only, optimal_dis_list_with_both,
                                                            'length of the tour (m)', 'comparison of tour length.png')
        np.savetxt("results/averaged_mse_varying_drone_capabilities_inference_only.txt",
                  averaged_mse_varying_drone_capabilities_inference_only)
        np.savetxt("results/averaged_mse_varying_drone_capabilities_distance_only.txt",
                  averaged_mse_varying_drone_capabilities_distance_only)
        np.savetxt("results/averaged_mse_varying_drone_capabilities_with_both.txt",
                  averaged_mse_varying_drone_capabilities_with_both)

    def cleanup(self,config):
        dir_exp1 = config['Dir']['dir_exp1']
        # Check if the directory exists
        if os.path.exists(dir_exp1) and os.path.isdir(dir_exp1):
            # Delete the directory and its contents
            shutil.rmtree(dir_exp1)
            print(f"Directory '{dir_exp1}' deleted.")
        else:
            print(f"Directory '{dir_exp1}' does not exist.")



config = configparser.ConfigParser()
config.read('config.ini')
dir_plots=config['Dir']['dir_dataset']

dir_sensor_map=config['Dir']['dir_sensor_map']
dir_dataset=config['Dir']['dir_dataset']
system=main(dir_plots, dir_dataset, dir_sensor_map)
exp1=Exp1.Exp1(config)
exp2=Exp2.Exp2(config)
exp3=Exp3_real_dataset.Exp3_real_dataset(config)
#system.cleanup(config)
dir_exp1=config['Dir']['dir_exp1']
dir_exp2=config['Dir']['dir_exp2']
dir_exp3=config['Dir']['dir_exp3']
exps=config['Run']['experiments'].split(',')
for exp in exps:
    if exp == 'Exp1':
        exp1.exp_1(dir_exp1)
    if exp == 'Exp2':
        exp2.exp_2(dir_exp2)
    if exp=='Exp3':
        exp3.exp_3(dir_exp3)



#todo
# 1.check the plot of sensor map for each tour, there are some problem for the depot
# 2. randomlize the sensor maps and run for 10 times for each capacity. check how Evan did that.
# 2. perform the experiment also for the machine learning and feature selection based approach
# 3  plot them and see.


#system.generate_dataset()
#main_df=system.preprocess.load_data("Dataset/Daily", 'txt')
#system.data_statistics("Dataset/Daily", 'txt')
#system.data_statistics("Dataset/Hourly", 'txt')
#this is the file for creating the learning dataframe
#system.create_data_frame_VW_2015_Jan_April("Dataset/Hourly", 'txt') # the CSV file is stored in Dataset/water_content_2015_Jan_April_32_sensors
#Dataset, mean_vec, cov = system.DG.multi_normal_variable_data_loader()
#real dataset
#Dataset=system.load_VW_data_2015_Jan_April()
#system.generate_sensors_location()
#synthetic dataset


#drone_capability=4000    #the Distance Budget, it needs to be changed to the Energy budget for hovering and travelling
#step_size=200   #it used in the experiments for varying the Energy Budget.
#system.initialization(data_path, sensor_map_path, drone_capability, step_size)
#system.run_myheuristic()
#system.run_baseline_ML()
#system.run_feature_selection_basline()
#system.plot_inference_error_of_both_ML_and_our_approach()


