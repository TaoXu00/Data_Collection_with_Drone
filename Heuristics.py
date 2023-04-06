import math

import Data_Generation as DG
import numpy as np
import scipy.stats
class Heuristics:
    def calculate_weight(self, cov):
        weights_dict = {}
        for j in range(len(cov[0])): #observed var
            sum = 0
            for i in range(len(cov[0])): # rest vars to be inffered
                sum += cov[i][j] ** 2
            weight = sum / cov[j][j]
            weights_dict[j] = weight
        return weights_dict

    def topw_update_rank_list(self, cov):
        available_vars = np.arange(len(cov[0]))
        selected_vars = []
        cov_y = cov
        w=len(available_vars) ##select all the variables
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
            cov_yy = cov_y[np.ix_(yy, yy)]
            cov_yx = cov_y[np.ix_(yy, [top_one])]
            cov_ss = cov_y[top_one][top_one]
            cov_xy = cov_y[np.ix_([top_one], yy)]
            cov_y = cov_yy - np.dot(cov_yx, cov_xy) / cov_ss
            # restore the new_cov back to the original matrix
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














