import math

import Data_Generator_and_Loader as DG
import numpy as np
import scipy.stats
import tsp
class Heuristics:
    def calculate_weight(self, cov, selected_vars, availbale_vars,sensor_json, drone, size_of_data_collection):
        weights_dict = {}
        for j in range(len(cov[0])):  # observed var
            sum = 0
            for i in range(len(cov[0])):  # rest vars to be inffered
                sum += cov[i][j] ** 2
            weight_cor = sum / cov[j][j]
            # compute the distance cost
            weight_energy_cost = math.inf
            for var in selected_vars:
                coor_x = sensor_json[str(var)]['Easting']
                coor_y = sensor_json[str(var)]['Northing']
                current_var=availbale_vars[j]
                dist = np.sqrt((sensor_json[str(current_var)]['Easting']-coor_x)**2+(sensor_json[str(current_var)]['Northing']-coor_y)**2)
                #if we want to change it to cost of the operation fee. We need to know the time.
                #cost = dist/speed * unit_time_uav_operation_cost + (size_of_data_collection/ drone.comm_rate)* unit_time_uav_operation_cost
                #cost= (dist/speed + size_of_data_collection/dron.comm_rate) * unite_time_uav_operation_cost
                energy_cost = dist*drone.flying_energy_per_unit + (size_of_data_collection/drone.comm_rate)*drone.hovering_energy_per_unit
                if weight_energy_cost>energy_cost:
                    weight_energy_cost=dist
            weights_dict[j] = 1 / weight_energy_cost * weight_cor

        return weights_dict

    def calculate_weight_updated(self, cov, selected_vars, availbale_vars, sensor_map, drone, size_of_data_collection):
        weights_dict = {}
        #compute the distance of the selected vars
        coordinates_selected = {}
        for sensor in selected_vars:
            coordinates_selected[sensor] = (
                float(sensor_map[str(sensor)]['Easting']), float(sensor_map[str(sensor)]['Northing']))
        if len(coordinates_selected) == 2:  # only one sensor is selected, so the drone will go from the depot to the sensor and then come back
            n1 = coordinates_selected[list(coordinates_selected.keys())[0]]
            n2 = coordinates_selected[list(coordinates_selected.keys())[1]]
            dis_current = math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2) * 2
        else:
            my_tsp = tsp.tsp_solver(selected_vars, coordinates_selected)
            tour, dis_current = my_tsp.solve()
        for j in range(len(cov[0])):  # observed var
            sum = 0
            for i in range(len(cov[0])):  # rest vars to be inffered
                sum += cov[i][j] ** 2
            weight_cor = sum / cov[j][j]
            # compute the distance cost
            location_ids = selected_vars + [availbale_vars[j]]
            location_ids = [str(var) for var in location_ids]
            if len(location_ids) == 2:  # only one sensor is selected, so the drone will go from the depot to the sensor and then come back
                n1 = (
                    float(sensor_map[location_ids[0]]['Easting']), float(sensor_map[location_ids[0]]['Northing']))
                n2 = (
                    float(sensor_map[location_ids[1]]['Easting']), float(sensor_map[location_ids[1]]['Northing']))
                dis = math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2) * 2
            else:
                coordinates = {}
                for sensor in location_ids:
                    coordinates[sensor] = (
                        float(sensor_map[sensor]['Easting']), float(sensor_map[sensor]['Northing']))

                my_tsp = tsp.tsp_solver(location_ids, coordinates)
                tour, dis = my_tsp.solve()
            energy_cost = (dis-dis_current) * drone.flying_energy_per_unit + (
                            size_of_data_collection / drone.comm_rate) * drone.hovering_energy_per_unit

            weights_dict[j] = 1 /energy_cost  * weight_cor

        return weights_dict

    def topw_update_ranking_list(self, cov, sensor_map, drone, size_of_data_collection):
        available_vars = np.arange(len(cov[0]))
        selected_vars = ['Depot']
        cov_y = cov
        w=len(available_vars) ##select all the variables
        while w != 0:
            weights_dict = self.calculate_weight(cov_y,selected_vars,available_vars,sensor_map, drone, size_of_data_collection) #the parameter here is different
            #weights_dict= self.calculate_weight_updated(cov_y, selected_vars, available_vars, sensor_map, drone, size_of_data_collection)
            sorted_d = dict(sorted(weights_dict.items(), key=lambda x: x[1], reverse=True))
            top_one = list(sorted_d.keys())[0]
            top_one_var = available_vars[top_one]
            selected_vars.append(top_one_var)
            available_vars = np.delete(available_vars, np.where(available_vars == top_one_var))
            # update covariance matrix
            w = w - 1
            yy = list(weights_dict.keys())
            yy.remove(top_one)
            cov_yy = cov_y[np.ix_(yy, yy)]
            cov_yx = cov_y[np.ix_(yy, [top_one])]
            cov_ss = cov_y[top_one][top_one]
            cov_xy = cov_y[np.ix_([top_one], yy)]
            cov_y = cov_yy - np.dot(cov_yx, cov_xy) / cov_ss
        return selected_vars

    def topw_update(self, cov, w):
        available_vars=np.arange(len(cov[0]))
        selected_vars=[]
        cov_y=cov
        while w!=0:
            weights_dict=self.calculate_weight(cov_y)
            sorted_d=dict(sorted(weights_dict.items(), key= lambda x:x[1], reverse=True))
            top_one= list(sorted_d.keys())[0]
            top_one_var=available_vars[top_one]
            selected_vars.append(top_one_var)
            available_vars = np.delete(available_vars, np.where(available_vars == top_one_var))
            #update covariance matrix
            w=w-1
            yy=list(weights_dict.keys())
            yy.remove(top_one)
            cov_yy = cov_y[np.ix_(yy, yy)]
            cov_yx = cov_y[np.ix_(yy, [top_one])]
            cov_ss = cov_y[top_one][top_one]
            cov_xy = cov_y[np.ix_([top_one], yy)]
            cov_y = cov_yy - np.dot(cov_yx,cov_xy)/cov_ss
            #restore the new_cov back to the original matrix
        return selected_vars




    def topw(self, cov, w):
        available_vars = np.arange(len(cov[0]))
        selected_vars = []
        cov_y = cov
        while w != 0:
            weights_dict = self.calculate_weight(cov_y)
            sorted_d = dict(sorted(weights_dict.items(), key=lambda x: x[1], reverse=True))
            top_one = list(sorted_d.keys())[0]
            top_one_var = available_vars[top_one]
            selected_vars.append(top_one_var)
            available_vars = np.delete(available_vars, np.where(available_vars == top_one_var))
            # update covariance matrix
            w = w - 1
            yy = list(weights_dict.keys())
            yy.remove(top_one)
            cov_y = cov_y[np.ix_(yy, yy)]
        return selected_vars













