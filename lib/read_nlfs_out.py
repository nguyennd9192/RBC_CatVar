from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcess
from sklearn import preprocessing


def generate_gps(dframe, data_buf, target_variable, method, kernel):
    for index, row in dframe.iterrows():

        if method == 'gp':
        #Gaussian Process
            this_best_nugget = float(row['best_nugget'])
            this_best_theta0 = float(row['best_theta0'])
            gp = GaussianProcess(theta0=this_best_theta0,
                                 nugget=this_best_nugget, random_start=100)

        elif method == 'kr':
        # Kernel Ridge or Ridge regression
            if kernel == 'linear':
                this_best_alpha = float(row['best_alpha'])
                gp = Ridge(alpha=this_best_alpha)
            else:
                this_best_alpha = float(row['best_alpha'])
                this_best_gamma = float(row['best_gamma'])
                gp = KernelRidge(kernel=kernel, alpha=this_best_alpha, gamma=this_best_gamma)
        else:
        	print ("The method \"{}\" does not support in nlfs".format(method))
        	print ("Process exists!!!")
        	exit()

        predicting_variables = list(dframe.at[index, "label"].split("|"))

        if target_variable in predicting_variables:
            predicting_variables.remove(target_variable)

        X = data_buf[predicting_variables].values
        scaler = preprocessing.MinMaxScaler()
        X_norm = scaler.fit_transform(X)

        y_obs = data_buf[target_variable].values
        yield (gp, X_norm, y_obs, predicting_variables)
