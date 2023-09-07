import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

class plotter:
    def __init__(self, directory):
        self.directory= directory
        self.colors=['firebrick', 'cornflowerblue', 'goldenrod', 'forestgreen', 'darkmagenta', 'black', 'yellow' ]
        self.colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        self.linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid', '--', 'dotted']
        self.font_size =13

    def plot_metrics_with_all_solutions_real_dataset(self, metric, drone_energy_capacity_list, avgs):
        plt.figure()
        plt.rcParams.update({'font.size': self.font_size})
        x = drone_energy_capacity_list
        keys = list(avgs.keys())
        for i in range(len(keys)):
            #plt.plot(x, avgs[keys[i]], label=keys[i], color=self.colors[i], linestyle=self.linestyles[i])
            # if keys[i] == 'baseline_ml' or keys[i] == 'mysolu':
            plt.plot(x, avgs[keys[i]], label=keys[i], color=self.colors[i], linestyle=self.linestyles[i])

        plt.xlabel("Budget of drone capabilities (J)")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(self.directory + '%s.png' % (metric), format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_metrics_with_all_solutions_exp2(self, metric,training_dataset_size_list, avgs, stds):
        plt.figure()
        plt.rcParams.update({'font.size': self.font_size})
        x = training_dataset_size_list
        keys = list(avgs.keys())
        for i in range(len(keys)):
            #if keys[i]=='baseline_ml' or keys[i] =='mysolu':
            plt.errorbar(x, avgs[keys[i]], stds[keys[i]], label=keys[i], color=self.colors[i],
                             linestyle=self.linestyles[i], elinewidth=1, capsize=5, capthick=1)

        plt.xlabel("size of training dataset")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(self.directory + '%s.png' % (metric), format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_metrics_with_all_solutions(self, metric, drone_energy_capacity_list, avgs, stds):
        plt.figure()
        plt.rcParams.update({'font.size': self.font_size})
        x=drone_energy_capacity_list
        keys=list(avgs.keys())
        for i in range(len(keys)):
           plt.errorbar(x, avgs[keys[i]], stds[keys[i]], label=keys[i], color=self.colors[i], linestyle=self.linestyles[i], elinewidth = 1, capsize = 5, capthick = 1, errorevery=3)

        plt.xlabel("Budget of drone capabilities (J)")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(self.directory + '%s.png' %(metric), format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_averaged_mse_vary_training_dataset_size_with_expected_value(self, num_of_training_dataset_size_list,
                                                                     averaged_mse_varying_traing_dataset_size,
                                                                     expected_mse_list):
        plt.figure()
        plt.plot(num_of_training_dataset_size_list, averaged_mse_varying_traing_dataset_size)
        plt.xlabel("size of the training dataset")
        plt.ylabel("MSE")
        plt.savefig(self.directory + 'mse_varying_training_dataset_size.png', format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_averaged_mse_vary_training_dataset_size(self, num_of_training_dataset_size_list,
                                                 averaged_mse_varying_training_dataset_size):

        plt.figure()
        plt.plot(num_of_training_dataset_size_list, averaged_mse_varying_training_dataset_size)
        plt.xlabel("size of the training dataset")
        plt.ylabel("MSE")
        plt.savefig(self.directory + 'mse_varying_training_dataset_size.png', format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_selected_sensors_varying_training_dataset_size(self, num_of_training_dataset_size_list, num_of_selected_nodes):
        plt.figure()
        plt.plot(num_of_training_dataset_size_list, num_of_selected_nodes)
        plt.xlabel("size of the training dataset")
        plt.ylabel("# of observation sensors")
        plt.savefig(self.directory + 'number of observation sensors with varying training dataset size.png', format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_tour_length_varying_training_dataset_size(self, num_of_training_dataset_size_list, optimal_dis_list):
        plt.figure()
        plt.plot(num_of_training_dataset_size_list, optimal_dis_list)
        plt.xlabel("size of the training dataset")
        plt.ylabel("tour distance")
        plt.savefig(self.directory + 'tour distance varying training dataset size.png',
                    format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_energy_cost_varying_training_dataset_size(self, num_of_training_dataset_size_list, optimal_energy_cost_list):
        plt.figure()
        plt.plot(num_of_training_dataset_size_list, optimal_energy_cost_list)
        plt.xlabel("size of the training dataset")
        plt.ylabel("energy cost of the tour")
        plt.savefig(self.directory + 'energy cost varying the training dataset size.png',
                    format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_mse_with_varying_drone_capabilities(self, drone_capability_list, mse_list):
        plt.figure()
        plt.plot(drone_capability_list, mse_list)
        plt.xlabel("Budget of drone capabilities (J)")
        plt.ylabel("MSE ")
        plt.savefig(self.directory + 'mse_varying_drone_capabilities.png', format="PNG",
                    bbox_inches='tight')
        plt.close()



    def plot_mse_with_varying_drone_cap_for_different_ML_models_with_correlation_based(self, total_dist_list, ML_mes_along_distance_for_variable_models, results_inference_only,
                                            results_distance_only, results_both, mse_feature_selection_for_GBR_model, mse_feature_selection_of_GBR_model_baseline2, mse_feature_selection_baseline2_linear_regression, scale):
        ML_models=['GBR', 'SVM', 'LR', 'RFR']
        plt.figure()
        for i in range(len(ML_models)):
            plt.plot(total_dist_list, np.array(ML_mes_along_distance_for_variable_models)[:,i], label=ML_models[i], color=self.colors[i],
                     linestyle=self.linestyles[i])
        plt.plot(total_dist_list, results_inference_only, label='inference only', color=self.colors[i+1],
                 linestyle=self.linestyles[i+1])
        plt.plot(total_dist_list, results_distance_only, label='distance only', color=self.colors[i+2],
                 linestyle=self.linestyles[i+2])
        plt.plot(total_dist_list, results_both, label='both', color=self.colors[i+3],
                 linestyle=self.linestyles[i+3])
        plt.plot(total_dist_list, mse_feature_selection_for_GBR_model, label='FS_GBR_feature_importance', color=self.colors[i+4])
        plt.plot(total_dist_list, mse_feature_selection_of_GBR_model_baseline2, label='FS_GBR', color=self.colors[i+5])
        plt.plot(total_dist_list, mse_feature_selection_baseline2_linear_regression, label="FS_lR", color=self.colors[i+6])
        plt.xlabel("Budget of drone capabilities (m)")
        plt.ylabel("MSE")
        plt.legend()
        if scale =="raw":
            plt.savefig(self.directory + "mse_for_various_ML_model_and_data_correlation_raw.png", format="PNG",
                        bbox_inches='tight')
        elif scale=="log":
            plt.savefig(self.directory + "mse_for_various_ML_model_and_data_correlation_log.png", format="PNG",
                            bbox_inches='tight')
        plt.close()

    def plot_mse_with_varying_drone_cap_for_different_ML_models(self, total_dist_list, mes_along_distance_for_variable_models):
        ML_models=['GBR', 'SVM', 'LR', 'RFR']
        plt.figure()
        for i in range(len(ML_models)):
            plt.plot(total_dist_list, np.array(mes_along_distance_for_variable_models)[:,i], label=ML_models[i], color=self.colors[i],
                     linestyle=self.linestyles[i])
        plt.xlabel("Budget of drone capabilities (m)")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig(self.directory + "mse_for_various_ML_model.png", format="PNG",
                    bbox_inches='tight')
        plt.close()


    def plot_comparison_vary_drone_capability(self, total_dist_list, results_inference_only,
                                            results_distance_only,results_both, ylabel, figure_name):
        plt.figure()
        plt.plot(total_dist_list, results_inference_only, label='inference only', color=self.colors[0],
                 linestyle=self.linestyles[0])
        plt.plot(total_dist_list, results_distance_only, label='distance only', color=self.colors[2],
                 linestyle=self.linestyles[2])
        plt.plot(total_dist_list, results_both, label='both', color=self.colors[1],
                 linestyle=self.linestyles[1])
        plt.xlabel("Budget of drone capabilities (m)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(self.directory + figure_name, format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_tour_map(self, sensors_json, tour, num_tour, distance, energy_cost):
        plt.figure()
        x=[]
        y=[]
        for sensor in sensors_json:
            x_0=sensors_json[sensor]['Easting']
            y_0=sensors_json[sensor]['Northing']
            if sensor=='Depot':
                plt.scatter(x_0, y_0, color='red')
            else:
                x.append(x_0)
                y.append(y_0)
        plt.scatter(x, y)
        for sensor in sensors_json:
            plt.text(sensors_json[sensor]['Easting'], sensors_json[sensor]['Northing'], sensor, va='bottom')

        start_x=sensors_json[tour[0]]['Easting']
        start_y=sensors_json[tour[0]]['Northing']

        for i in range(len(tour)-1):
            x_i=sensors_json[tour[i]]['Easting']
            y_i=sensors_json[tour[i]]['Northing']
            x_next=sensors_json[tour[i+1]]['Easting']
            y_next=sensors_json[tour[i+1]]['Northing']
            plt.annotate("",
                           xy=(x_i, y_i), xycoords='data',
                           xytext=(x_next,y_next), textcoords='data',
                           arrowprops=dict(arrowstyle="<-",
                                           connectionstyle="arc3"))
            #plt.arrow(x_i,y_i ,(x_next-x_i),(y_next-y_i), color='g', length_includes_head = True)
        end_x=sensors_json[tour[i+1]]['Easting']
        end_y=sensors_json[tour[i + 1]]['Northing']

        plt.annotate("",
                     xy=(end_x, end_y), xycoords='data',
                     xytext=(start_x, start_y), textcoords='data',
                     arrowprops=dict(arrowstyle="<-",
                                     connectionstyle="arc3"))
        textstr = "Tour %d\nN nodes: %d\nLength: %.3f\nEnergy cost: %.3f" % (num_tour,len(tour), distance, energy_cost)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(650, 630, textstr, fontsize=10,  # Textbox
                   verticalalignment='top', bbox=props)
        plt.xlabel('Relative Easting (m)')
        plt.ylabel('Relative Easting (m)')
        plt.savefig(self.directory + 'tour_%s.png' %(num_tour), format="PNG", bbox_inches='tight')
        plt.close()

    def plot_tour_length_vary_drone_capability(self, total_dist_list, optimal_dis_list):
        plt.figure()
        plt.plot(total_dist_list, optimal_dis_list)
        plt.xlabel("Budget of drone capabilities (m)")
        plt.ylabel("length of the tour (m)")
        plt.savefig(self.directory + 'length of the tour with varying drone capabilities.png', format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_energy_cost_vary_drone_capability(self, drone_capacity_list, optimal_energy_cost_list):
        plt.figure()
        plt.plot(drone_capacity_list, optimal_energy_cost_list)
        plt.xlabel("Budget of drone energy capabilities (m)")
        plt.ylabel("energy cost of the tour (m)")
        plt.savefig(self.directory + 'energy cost of the tour with varying drone capabilities.png', format="PNG",
                    bbox_inches='tight')
        plt.close()


    def plot_selected_sensors_vary_drone_capability(self, total_dist_list, num_of_selected_nodes):
        plt.figure()
        plt.plot(total_dist_list, num_of_selected_nodes)
        plt.xlabel("Budget of drone capabilities (m)")
        plt.ylabel("# of observation sensors")
        plt.savefig(self.directory + 'number of observation sensors with varying drone capabilities.png', format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_sensor_map(self, coordinates,nodes, name):
        x, y = zip(*coordinates)
        plt.figure()
        plt.scatter(x, y)
        # plt.scatter(0, 0, color='red')
        for i in range(len(nodes)):
            plt.text(x[i], y[i], nodes[i], va='bottom')
            if nodes[i] == 'Depot':
                plt.scatter(x[i], y[i], color='red')
        plt.xlabel('Relative Easting (m)')
        plt.ylabel('Relative Easting (m)')
        plt.savefig('%s.png' %(name), format="PNG",
                    bbox_inches='tight')

    def plot_distance_varying_sensors(self,num_sensor_list, opt_dists):
        plt.figure()
        plt.plot(num_sensor_list,opt_dists)
        plt.xlabel("Budget of #sensors")
        plt.ylabel("optimal distance (m)")
        plt.savefig(self.directory + 'drone distance with varying sensors.png', format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_averaged_mse_vary_drone_capability(self, total_dist_list,averaged_mse_varying_drone_capabilities, expected_mse_list):
        plt.figure()
        colors = ['firebrick', 'cornflowerblue', 'goldenrod', 'forestgreen', 'darkmagenta']
        linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        plt.plot(total_dist_list, averaged_mse_varying_drone_capabilities, label='inference error', color=colors[0], linestyle=linestyles[0])
        plt.plot(total_dist_list, expected_mse_list, label='expected error', color=colors[1], linestyle=linestyles[1] )
        plt.xlabel("Budget of drone capabilities (m)")
        plt.ylabel("averaged MSE of inference (m\u00b3/m\u00b3)")
        plt.legend()
        plt.savefig(self.directory + 'mse_varying_drone_capabilities.png', format="PNG", bbox_inches='tight')
        plt.close()

    def plot_averaged_mse_vary_drone_capability_varying_method(self, total_dist_list,averaged_mse_varying_drone_capabilities, expected_mse_list, method):
        plt.figure()
        colors = ['firebrick', 'cornflowerblue', 'goldenrod', 'forestgreen', 'darkmagenta']
        linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        plt.plot(total_dist_list, averaged_mse_varying_drone_capabilities, label='inference error', color=colors[0], linestyle=linestyles[0])
        plt.plot(total_dist_list, expected_mse_list, label='expected error', color=colors[1], linestyle=linestyles[1] )
        plt.xlabel("Budget of drone capabilities (m)")
        plt.ylabel("averaged MSE of inference (m\u00b3/m\u00b3)")
        plt.legend()
        plt.savefig(self.directory + 'inference error with varying drone capabilities_%s.png' %(method), format="PNG", bbox_inches='tight')
        plt.close()

    def plot_cummulative_probability(self, array, plotname):
        plt.figure()
        plt.hist(array, weights=np.ones(len(array)) / len(array), cumulative=True)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        #plt.show()
        plt.xlabel("Correlation coefficient")
        plt.ylabel("Cumulative probability")
        plt.savefig('plots/%s.png' %(plotname), bbox_inches='tight')

    def plot_abs_error_mean(self,abs_error_mean):
        # plot the mse
        plt.figure()
        x = range(len(abs_error_mean))
        plt.plot(x,abs_error_mean)
        plt.xlabel("time")
        plt.ylabel("% absolute error of estimated mean")
        plt.savefig(self.directory + 'abs_error_mean.png', format="PNG")
        plt.close()

    def plot_abs_error_std(self,abs_error_std):
        # plot the mse
        plt.figure()
        x = range(len(abs_error_std)-1)
        plt.plot(x,abs_error_std[1:])
        plt.xlabel("time")
        plt.ylabel("% absolute error of estimated std")
        plt.savefig(self.directory + 'abs_error_std.png', format="PNG")
        plt.close()

    def plot_inference_mse(self, mse_list, expect_mse, n_obs):
        # plot the total rewards
        # print(f"total_rewards:{total_rewards}")
        plt.figure()
        mse_list=np.array(mse_list)
        averaged_array=np.average(mse_list.reshape(-1,10), axis=1)
        print("%d elements after average" %(len(averaged_array)))
        x = range(len(averaged_array))
        #plt.ylim(0,10)
        plt.plot(x, averaged_array)
        plt.hlines(expect_mse,0,len(averaged_array),color='red')
        plt.xlabel("time")
        plt.ylabel("MSE of Inference")
        # plt.show()
        plt.savefig(self.directory + 'mse of inference_inferred_cov_mean_%s_obs_var.png' %(n_obs), format="PNG")
        plt.close()

    def plot_inference_mse_multi_variable_with_different_heuristics(self, mse_along_time_total_list_heu_dict,n_obs_var):
        labels=list(mse_along_time_total_list_heu_dict.keys())
        # line_num=len(total_edge_mse_list_with_increasing_monitors)
        x = n_obs_var
        averaged_mse_list=[]
        std_mse_list=[]
        '''
        for key in mse_along_time_total_list_heu_dict:
            averaged_mse=np.average(mse_along_time_total_list_heu_dict[key],axis=1)
            std_mse=np.std(mse_along_time_total_list_heu_dict[key], axis=1)
            averaged_mse_list.append(averaged_mse)
            std_mse_list.append(std_mse)
        '''
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})
        colors = ['firebrick', 'cornflowerblue', 'goldenrod', 'forestgreen', 'darkmagenta']
        linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        #plt.plot(x, expected_mse_list, label=labels[0], color=colors[0], linestyle=linestyles[0])
        for i in range(len(list(mse_along_time_total_list_heu_dict.keys()))):
            #plt.errorbar(x,averaged_mse_list[i], std_mse_list[i], label=labels[i], color=colors[i], linestyle=linestyles[i], elinewidth = 1, capsize = 5, capthick = 1, errorevery=3)
             plt.plot(x,mse_along_time_total_list_heu_dict[list(mse_along_time_total_list_heu_dict.keys())[i]], label=labels[i], color=colors[i], linestyle=linestyles[i])
        plt.xlabel("Budget(# of observation variables)")
        plt.ylabel("averaged MSE of inference (m\u00b3/m\u00b3)")
        #plt.ylabel("averaged inference MSE")
        plt.legend(fontsize=13)
        # plt.grid(True)
        plt.savefig(self.directory + "MSE_of_VW_Inference_2015_Jan_April_with_sensor_num_as_budget_heu.png", bbox_inches='tight')
        #plt.savefig(self.directory + "MSE_of_synthetic_data_heu.png",bbox_inches='tight')
        plt.close()

    def plot_inference_mse_multi_variable(self, mse_along_time_total_list,expected_mse_list, n_obs_var):
        labels = ['expected_mse', 'time_averaged_mse']
        for n in n_obs_var:
            labels.append(str(n))
        # line_num=len(total_edge_mse_list_with_increasing_monitors)
        x = n_obs_var
        averaged_mse=np.average(mse_along_time_total_list,axis=1)
        std_mse=np.std(mse_along_time_total_list, axis=1)
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})
        colors = ['firebrick', 'cornflowerblue', 'goldenrod', 'forestgreen', 'darkmagenta']
        linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        #for i in range(len(n_obs_var)):
        #    plt.hlines(expected_mse_list[i],0,len(mse_along_time_total_list[0]), color=colors[i])
        #    plt.plot(x, mse_along_time_total_list[i], label=labels[i], color=colors[i], linestyle=linestyles[i])
        plt.plot(x, expected_mse_list, label=labels[0], color=colors[0], linestyle=linestyles[0])
        plt.errorbar(x,averaged_mse, std_mse, label=labels[1], color=colors[1], linestyle=linestyles[1], elinewidth = 1, capsize = 5, capthick = 1, errorevery=3)
        plt.xlabel("Budget(# of observation variables)")
        plt.ylabel("MSE of inference (m\u00b3/m\u00b3)")
        plt.legend(fontsize=13)
        # plt.grid(True)
        plt.savefig(self.directory + "MSE_of_VW_Inference_2015_Jan_April_with_sensor_num_as_budget.png", bbox_inches='tight')
        plt.close()

    def plot_expect_mse_with_different_heuristics(self, expected_mse_heu_dict, n_obs_var):
        labels = list(expected_mse_heu_dict.keys())
        x = n_obs_var

        '''
        for key in mse_along_time_total_list_heu_dict:
            averaged_mse=np.average(mse_along_time_total_list_heu_dict[key],axis=1)
            std_mse=np.std(mse_along_time_total_list_heu_dict[key], axis=1)
            averaged_mse_list.append(averaged_mse)
            std_mse_list.append(std_mse)
        '''
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})
        colors = ['firebrick', 'cornflowerblue', 'goldenrod', 'forestgreen', 'darkmagenta']
        linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        # plt.plot(x, expected_mse_list, label=labels[0], color=colors[0], linestyle=linestyles[0])
        for i in range(len(list(expected_mse_heu_dict.keys()))):
            # plt.errorbar(x,averaged_mse_list[i], std_mse_list[i], label=labels[i], color=colors[i], linestyle=linestyles[i], elinewidth = 1, capsize = 5, capthick = 1, errorevery=3)
            plt.plot(x, expected_mse_heu_dict[list(expected_mse_heu_dict.keys())[i]],
                     label=labels[i], color=colors[i], linestyle=linestyles[i])
        plt.xlabel("Budget(# of observation variables)")
        plt.ylabel("expected inference MSE (m\u00b3/m\u00b3)")
        # plt.ylabel("averaged inference MSE")
        plt.legend(fontsize=13)
        # plt.grid(True)
        plt.savefig(self.directory + "expected_MSE_of_VW_Inference_2015_Jan_April_with_sensor_num_as_budget_synthetic_heu.png",
                    bbox_inches='tight')
        plt.close()