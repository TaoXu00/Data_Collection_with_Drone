import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

class plotter:
    def __init__(self, directory):
        self.directory= directory

    def plot_tour_map(self, sensors_json, tour, num_tour, distance):
        plt.figure()
        x=[]
        y=[]
        for sensor in sensors_json:
            x_0=sensors_json[sensor]['Easting']
            x.append(x_0)
            y_0=sensors_json[sensor]['Northing']
            y.append(y_0)
        plt.scatter(x, y)
        plt.scatter(0, 500, color='red')
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
        textstr = "Tour %d\nN nodes: %d\nLength: %.3f" % (num_tour,len(tour), distance)
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

    def plot_selected_sensors_vary_drone_capability(self, total_dist_list, num_of_selected_nodes):
        plt.figure()
        plt.plot(total_dist_list, num_of_selected_nodes)
        plt.xlabel("Budget of drone capabilities (m)")
        plt.ylabel("# of observation sensors")
        plt.savefig(self.directory + 'number of observation sensors with varying drone capabilities.png', format="PNG",
                    bbox_inches='tight')
        plt.close()

    def plot_sensor_map(self,coordinates,nodes):
        x, y = zip(*coordinates)
        plt.figure()
        plt.scatter(x, y)
        # plt.scatter(0, 0, color='red')
        for i in range(len(nodes)):
            plt.text(x[i], y[i], nodes[i], va='bottom')
        plt.scatter(0, 500, color='red')
        plt.xlabel('Relative Easting (m)')
        plt.ylabel('Relative Easting (m)')
        plt.savefig(self.directory + 'sensor_map.png', format="PNG",
                    bbox_inches='tight')

    def plot_distance_varying_sensors(self,num_sensor_list, opt_dists):
        plt.figure()
        plt.plot(num_sensor_list,opt_dists)
        plt.xlabel("Budget of #sensors")
        plt.ylabel("optimal distance (m)")
        plt.savefig(self.directory + 'drone distance with varying sensors.png', format="PNG",
                    bbox_inches='tight')
        plt.close()
    def plot_averaged_mse_vary_drone_capability(self, total_dist_list,averaged_mse_varying_drone_capabilities, expected_mse_list ):
        plt.figure()
        colors = ['firebrick', 'cornflowerblue', 'goldenrod', 'forestgreen', 'darkmagenta']
        linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        plt.plot(total_dist_list, averaged_mse_varying_drone_capabilities, label='inference error', color=colors[0], linestyle=linestyles[0])
        plt.plot(total_dist_list, expected_mse_list, label='expected error', color=colors[1], linestyle=linestyles[1] )
        plt.xlabel("Budget of drone capabilities (m)")
        plt.ylabel("averaged MSE of inference (m\u00b3/m\u00b3)")
        plt.legend()
        plt.savefig(self.directory + 'inference error with varying drone capabilities.png', format="PNG", bbox_inches='tight')
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


    def plot_edge_delay_difference_alongtime(self, s,e, edge_delay_difference_list,link_range):
        # set width of bar
        barWidth = 0.25
        fig = plt.subplots()
        # set height of bar
        init= edge_delay_difference_list[0][s:e]
        t1000 = edge_delay_difference_list[1][s:e]
        t2000 = edge_delay_difference_list[2][s:e]
        t3000= edge_delay_difference_list[3][s:e]
        # Set position of bar on X axis
        br1 = np.arange(e-s)
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
        # Make the plot
        plt.bar(br1, init, color='r', width=barWidth,
                edgecolor='grey', label='init')
        plt.bar(br2, t1000, color='g', width=barWidth,
                edgecolor='grey', label='t1000')
        plt.bar(br3, t2000, color='b', width=barWidth,
                edgecolor='grey', label='t2000')
        plt.bar(br4, t3000, color='c', width=barWidth,
                edgecolor='grey', label='t3000')

        # Adding Xticks
        xlable=[x for x in range(s,e)]
        plt.xlabel('linkID', fontweight='bold', fontsize=15)
        plt.ylabel('delay difference from the mean', fontweight='bold', fontsize=15)
        plt.xticks([r for r in range(e-s)], range(s,e,1))
        plt.legend()
        plt.savefig(self.directory + 'delay difference from the mean link %s' %link_range, format="PNG")
        plt.close()



    def plot_total_edge_delay_mse_with_increasing_monitor_training_from_file(self, monitors_deployment_percentage, filename):
        labels = []
        for per in monitors_deployment_percentage:
            labels.append(str(per) + '%')
        #line_num=len(total_edge_mse_list_with_increasing_monitors)
        total_edge_mse_list_with_increasing_monitors=np.loadtxt(filename, dtype=float)
        x=range(len(total_edge_mse_list_with_increasing_monitors[0]))
        #x = range(len(total_edge_mse_list_with_increasing_monitors))
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})

        colors=['firebrick', 'cornflowerblue', 'goldenrod', 'forestgreen', 'darkmagenta']
        linestyles = ['dotted','dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)),'solid']
        for i in range (len(total_edge_mse_list_with_increasing_monitors)):
            plt.plot(x, total_edge_mse_list_with_increasing_monitors[i], label=labels[i], color=colors[i], linestyle= linestyles[i])
        plt.xlabel("learning time")
        plt.ylabel("MSE of link delay during learning")
        plt.legend(fontsize=13)
        #plt.grid(True)
        plt.savefig(self.directory + "MSE_of_total_links_delay_with_increasing_monitor_training")
        plt.close()

    def plot_NT_verification_edge_computed_rate_with_monitors_increasing(self, G, monitors_list, solved_edges_count ):
        plt.figure()
        x = [len(monitors) / len(G.nodes) for monitors in monitors_list]
        y = [edges_count / len(G.edges) for edges_count in solved_edges_count]
        #print(x, y)
        plt.plot(x, y)
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel("% of solved links")
        # plt.show()
        plt.savefig('plots/network_tomography_verification_node%s_with_link_weight=1.png'%(len(G.nodes)))
        plt.close()

    def plot_rewards_mse_along_with_different_monitors(self,monitors_deployment_percentage,total_rewards_mse_list):
        labels=[]
        for per in monitors_deployment_percentage:
            labels.append(str(per)+'%')
        line_num = len(total_rewards_mse_list)
        x = range(len(total_rewards_mse_list[0]))
        fig = plt.figure(figsize=(10, 7))
        for i in range(line_num):
            plt.plot(x, total_rewards_mse_list[i], label=labels[i])
        plt.xlabel("time")
        plt.ylabel("mse of time averaged rewards of the selected optimal paths during training")
        plt.legend()
        plt.savefig(self.directory + "rewards mse with different #minitors")
        plt.close()

    def plot_edge_exporation_times_with_differrent_monitor_size(self, G, total_edge_exploration_during_training_list):
            edges_num = len(G.edges)
            index = range(0, edges_num)
            # index.sort()
            selected_edges_list = []
            for i in range(len(total_edge_exploration_during_training_list)):
                list = total_edge_exploration_during_training_list[i]
                selected_edges_list.append([list[index[j]] for j in range(len(index))])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            # Multiple bar chart
            x = ['0.1', '0.2', '0.3', '0.4','0.5']
            index = [str(index[i]) for i in range(len(index))]
            for i in range(len(total_edge_exploration_during_training_list)):
                ax.bar(index, selected_edges_list[i], width=0.55, align='center', label=x[i])
            # Define x-ticks
            # ticks=[str(index[i]) for i in range(len(index))]
            # plt.xticks(index, ticks)
            # Layout and Display
            plt.xlabel("LinkID")
            plt.ylabel("total explored time during MAB training ")
            plt.tight_layout()
            plt.legend()
            plt.savefig(self.directory + " the exploration times of 20 random edges with different monitor numbers")
            plt.close()

    def plot_edge_computed_rate_during_training(self,monitors_deployment_percentage,average_computed_edge_rate_during_training):
        x = []
        for per in monitors_deployment_percentage:
            x.append(str(per))
        y = average_computed_edge_rate_during_training
        fig = plt.figure(figsize=(10, 7))
        barwidth = 0.25
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel("% of computed edges")
        bar = np.array(x)
        plt.bar(bar, y, width=barwidth)
        plt.savefig(self.directory + 'MAB_edge_computed_rate_with_increasing_monitors.png')
        plt.close()

    def plot_edge_computed_rate_with_different_topology_size(self):
        percentage = ['0.1', '0.2', '0.3', '0.4', '0.5','0.6','0.7','0.8','0.9','1.0']
        edge_comput_rate_20nodes=[0.0, 0.05796490043874452, 0.10419244759440507, 0.11742978212772341, 0.2896838789515131, 0.3347301908726141, 0.5024843439457007, 0.7434657066786665,0.7759215509806128, 0.8975794052574343]
        edge_compute_number_50nodes=[1.256631071305546,6.163623837409577,20.436445056837755,30.779193937306236,43.56286600068894,64.29968997588702,64.19910437478471,73.87599035480537,81.72959007922839, 79.74061315880124]
        edge_compute_rate_50nodes=[x/96 for x in edge_compute_number_50nodes]
        barWidth = 0.25
        fig = plt.subplots()
        # set height of bar
        nodes_20 = edge_comput_rate_20nodes
        nodes_50 = edge_compute_rate_50nodes
        # Set position of bar on X axis
        br1 = np.arange(len(percentage))
        br2 = [x + barWidth for x in br1]

        # Make the plot
        plt.bar(br1, nodes_20, color='r', width=barWidth,
                edgecolor='grey', label='20 nodes')
        plt.bar(br2, nodes_50, color='g', width=barWidth,
                edgecolor='grey', label='50 nodes')

        # Adding Xticks
        plt.xlabel('%of nodes selected as monitors',  fontsize=15)
        plt.ylabel('%of the identified links',  fontsize=15)
        plt.xticks(np.arange(len(percentage)),percentage)
        plt.legend()
        plt.savefig(self.directory + '%of identified edges with different topology size and different number of monitors' , format="PNG")
        plt.close()

    def plot_average_regrets(self, averaged_regret_list):
        plt.figure()
        x = range(len(averaged_regret_list))
        plt.plot(x, averaged_regret_list)
        plt.xlabel("time")
        plt.ylabel("averaged regret of selected shortest path among monitors")
        plt.savefig(self.directory + 'averaged regret', format="PNG")
        plt.close()

    def plot_rate_of_correct_shortest_path(self, correct_shortest_path_selected_rate):
        plt.figure()
        x = range(len(correct_shortest_path_selected_rate))
        plt.plot(x, correct_shortest_path_selected_rate)
        plt.xlabel("time")
        plt.ylabel("rate of correctly selected shortest path among monitors")
        plt.savefig(self.directory + 'rate of correctly selected shortest path among monitors', format="PNG")
        plt.close()

    def plot_edge_delay_difference_for_some_edges(self, optimal_edges_delay_difference_after_inti,
                                                  optimal_edges_delay_difference_after_training):
        barWidth = 0.25
        fig = plt.subplots()
        # set height of bar
        init = optimal_edges_delay_difference_after_inti
        after_training = optimal_edges_delay_difference_after_training
        br1 = np.arange(len(init))
        br2 = [x + barWidth for x in br1]
        # Make the plot
        plt.bar(br1, init, color='r', width=barWidth,
                edgecolor='grey', label='after_init')
        plt.bar(br2, after_training, color='g', width=barWidth,
                edgecolor='grey', label='after_training')

        # Adding Xticks
        xlable = np.arange(len(init))
        plt.xlabel('linkID', fontweight='bold', fontsize=15)
        plt.ylabel('delay difference from the mean', fontweight='bold', fontsize=15)
        plt.xticks(xlable)
        plt.legend()
        plt.savefig(self.directory + 'delay difference of optimal edges from mean after init and after training')
        plt.close()

    def plot_diff_from_optimal_path_of_selected_shortest_paths(self, abs_diff_of_delay_from_optimal):
        plt.figure()
        x = range(len(abs_diff_of_delay_from_optimal))
        plt.plot(x, abs_diff_of_delay_from_optimal)
        plt.xlabel("time")
        plt.ylabel("mse of the selected shortest path from optimal shortest path")
        plt.savefig(self.directory + 'absolute difference of the selected shortest path from optimal shortest path', format="PNG")
        plt.close()

    def plot_optimal_path_selected_percentage_list_with_increasing_monitors(self, monitors_deployment_percentage, optimal_path_selected_rate):
        x=monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        y = optimal_path_selected_rate
        fig = plt.figure(figsize=(10, 7))
        barwidth = 0.25
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel(" % of the optimal paths selected")
        bar = np.arange(len(x_label))
        plt.bar(bar, y, width=barwidth)
        plt.xticks(bar,x_label)
        plt.savefig(self.directory + 'MAB_edge_computed_rate_with_increasing_monitors.png')
        plt.close()

    def plot_abs_diff_path_delay_from_the_optimal(self, monitors_deployment_percentage, optimal_path_selected_percentage_list):
        x=monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        y = optimal_path_selected_percentage_list
        fig = plt.figure(figsize=(10, 7))
        barwidth = 0.25
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel(" abs error from the optimal shortest paths")
        bar = np.arange(len(x_label))
        plt.bar(bar, y, width=barwidth)
        plt.xticks(bar, x_label)
        plt.savefig(self.directory + 'abs error from the optimal shortest paths.png')
        plt.legend()
        plt.close()

    def plot_percentage_of_optimal_path_selected_rate_BR_50nodes(self,monitors_deployment_percentage, subito_op_rate, UCB1_op_rate, subito_perfect_op_rate):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))

        # set height of bar
        x=monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        br1 = np.arange(len(UCB1_op_rate))
        br2 = [x + barWidth for x in br1]
        br3= [x + barWidth for x in br2]
        for i in range(len(UCB1_op_rate)):
            UCB1_op_rate[i]=UCB1_op_rate[i]*100
            subito_op_rate[i]=subito_op_rate[i]*100
            subito_perfect_op_rate[i]=subito_perfect_op_rate[i]*100
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_op_rate,  width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/', color='white')
        plt.bar(br2,subito_op_rate,  width=barWidth,
                edgecolor='grey', label='Subito', hatch='o', color='white')
        plt.bar(br3, subito_perfect_op_rate, width=barWidth,
                edgecolor='grey', label='Subito*', hatch='*', color='white')

        # Adding Xticks
        plt.xlabel('% of nodes selected as monitors')
        plt.ylabel('Rate(%) of expected paths selected')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='lower left')
        plt.savefig(self.directory + 'Scability_of Minitor_op_rate')
        plt.close()

    def plot_abs_delay_of_optimal_path_selected_from_mean_BR_50nodes(self,monitors_deployment_percentage,subito_diff, UCB1_diff, subito_perfect_diff):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))
        # set height of bar
        x = monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        br1 = np.arange(len(UCB1_diff))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_diff,  width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/', color='white')
        plt.bar(br2, subito_diff,  width=barWidth,
                edgecolor='grey', label='Subito', hatch='o', color='white')
        plt.bar(br3, subito_perfect_diff,  width=barWidth,
                edgecolor='grey', label='Subito*', hatch='*', color='white')

        # Adding Xticks
        plt.xlabel('% of nodes selected as monitors')
        plt.ylabel('Delay difference from expectation ')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='lower left')
        plt.savefig(self.directory + "Scability_of Minitor_delay_diff")
        plt.close()

    def plot_percentage_of_optimal_path_selected_rate_for_various_monitor_size(self, topology_size, subito_op_rate, UCB1_op_rate, subito_perfect_op_rate):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))
        # set height of bar
        x_label = [str(size) for size in topology_size]
        br1 = np.arange(len(UCB1_op_rate))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        for i in range(len(UCB1_op_rate)):
            UCB1_op_rate[i] = UCB1_op_rate[i] * 100
            subito_op_rate[i] = subito_op_rate[i] * 100
            subito_perfect_op_rate[i] = subito_perfect_op_rate[i] * 100
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_op_rate, width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/')
        plt.bar(br2, subito_op_rate, width=barWidth,
                edgecolor='grey', label='Subito', hatch='o')
        plt.bar(br3, subito_perfect_op_rate, width=barWidth,
                edgecolor='grey', label='Subito*', hatch='*')

        # Adding Xticks
        plt.xlabel('network size')
        plt.ylabel('Rate(%) of expected paths selected')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='lower left')
        plt.savefig(self.directory + 'scalability_of_network_size_op_rate')
        plt.close()

    def plot_abs_delay_of_optimal_path_selected_for_various_monitor_size(self,topology_size, subito_diff, UCB1_diff, subito_perfect_diff):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))
        # set height of bar
        x_label = [str(size) for size in topology_size]
        br1 = np.arange(len(UCB1_diff))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_diff, width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/')
        plt.bar(br2, subito_diff, width=barWidth,
                edgecolor='grey', label='Subito', hatch='o')
        plt.bar(br3, subito_perfect_diff, width=barWidth,
                edgecolor='grey', label='Subito*', hatch='*')

        # Adding Xticks
        plt.xlabel('network size')
        plt.ylabel('Delay difference from expectation')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='upper left')
        plt.savefig(self.directory + 'Scability_of_network_size_delay_diff')
        plt.close()



    def plot_percentage_of_optimal_path_selected_rate_BTN(self, monitors_deployment_percentage,myapproach,baseline):
        barWidth = 0.25
        fig = plt.subplots()
        # set height of bar
        x=monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        br1 = np.arange(len(baseline))
        br2 = [x + barWidth for x in br1]
        for i in range(len(baseline)):
            baseline[i]=baseline[i]*100
            myapproach[i]=myapproach[i]*100
        # Make the plot
        plt.bar(br1, baseline, color='r', width=barWidth,
                edgecolor='grey', label='baseline')
        plt.bar(br2, myapproach, color='g', width=barWidth,
                edgecolor='grey', label='our approach')

        # Adding Xticks
        plt.xlabel('% of nodes selected as monitors')
        plt.ylabel('% of optimal path selected')
        plt.xticks(br1, x_label)
        plt.legend()
        plt.savefig(self.directory + 'average percentage of the optimal shortest path selected rate with 10% - 50% monitors deployed_BTN')
        plt.close()

    def plot_abs_delay_of_optimal_path_selected_from_mean_BTN(self, monitors_deployment_percentage, myapproach, baseline):
        barWidth = 0.25
        fig = plt.subplots()
        # set height of bar
        x = monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        br1 = np.arange(len(baseline))
        br2 = [x + barWidth for x in br1]
        # Make the plot
        plt.bar(br1, baseline, color='r', width=barWidth,
                edgecolor='grey', label='baseline')
        plt.bar(br2, myapproach, color='g', width=barWidth,
                edgecolor='grey', label='our approach')

        # Adding Xticks
        plt.xlabel('% of nodes selected as monitors')
        plt.ylabel('avg abs difference of selected shortest paths from real ')
        plt.xticks(br1, x_label)
        plt.legend()
        plt.savefig(self.directory + 'average absolute difference of the selected shortest paths from real optimal paths with 10%-50% monitors deployed_BTN')
        plt.close()