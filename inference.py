import numpy as np


class Inference:
    def __init__(self, Dataset, mean_vec, cov):
        self.mean_vec = mean_vec
        self.cov = cov
        self.Dataset = Dataset

    def variable_random_selection(self, num_var_select):
        row, num_var = self.Dataset.shape
        indexes = np.arange(num_var)
        obs_var = []
        while len(obs_var) != num_var_select:
            index = np.random.choice(indexes, 1)[0]
            if index not in obs_var:
                obs_var.append(index)
        obs_var.sort()
        unknown_var = np.delete(indexes, obs_var)
        return obs_var, unknown_var

    def infer_unobs(self, obs_var, unknown_var, k, num_inference):
        """
        Estimate the sensing data of the unselected sensors based on the data observed from the selected sensors
        @param obs_var: the selected sensors
        @param unknown_var: the unselected sensors
        @param k: the index of the data used for training
        @param num_inference: the amount of data to be estimated
        @return: the total mse of the estimated value against their true data and the estimated values
        """
        mean_vec = self.mean_vec
        obs_mean_vec = np.take(mean_vec, obs_var)
        unknown_mean_vec = np.take(mean_vec, unknown_var)
        cov = self.cov
        unknown_obs_cov = cov[np.ix_(unknown_var, obs_var)]
        # obs_cov=np.delete(cov,unknown_var, axis=0)
        # obs_cov=np.delete(obs_var,unknown_var, axis=1)
        obs_v = cov[np.ix_(obs_var, obs_var)]
        mse_along_time = []
        inferred_all = np.empty((len(unknown_var), 1))
        for i in np.arange(k, k + num_inference): # for t =i to t=num_inference
            x = self.Dataset[i]  # real value at time i
            # implement the Guassian conditional probability
            true_value = np.take(x, unknown_var)
            # print("real unknown vars are %s" %(np.take(x,unknown_var)))
            x = np.take(x, obs_var)
            inverse = np.linalg.pinv(obs_v)
            a = np.dot(unknown_obs_cov, inverse)
            infer_var = unknown_mean_vec + np.dot(a, (x - obs_mean_vec))
            infer_var_r = np.array(infer_var).reshape(len(infer_var), 1)
            inferred_all = np.append(inferred_all, infer_var_r,
                                     axis=1)  # append the new inference value to the next column
            mse = 0
            # print("inferred vars are %s:" % (infer_var))
            for j in np.arange(len(unknown_var)):
                mse += (true_value[j] - infer_var[j]) ** 2
                # print("mse %f:" %(mse))
            mse_along_time.append(mse)
        return mse_along_time, inferred_all
