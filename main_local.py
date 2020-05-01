import os
import time
import pandas as pd
import numpy as np
from lib.read_load_config import *
from lib.weight_score import weight_score
from lib.ini_sampling import rand_gidx_sampling
from lib.lasso_in_cluster import revise_KMEANs_by_LASSO
from lib.confusion_matrix import confusion_matrix
from lib.normalize import get_Xnorm
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
 
    # This project, Regression-based clustering is created to do instances clustering for a given data set.
    # The criteria for the group assignment of any instances is the absolute error of predicting ability
    # to the model working on its cluster.

    # Input:
    # 1. A dataframe contains source data, without missing value, the first column is the index
    # 2. Number of cluster or range of number of cluster for surveying
    # 3. Method of initial state
    # 4. A dataframe for predicting variables list

    t1 = time.time()

    # read_load_config
    print ("Start reading config")
    print ("========================")
    cf = load_config(cfg_file=None)
    config, function = read_config(cf)

    for key, value in config.items():
        print ("{0}: {1}".format(key, value))
    print ("Sucess in reading config") 
    print ("========================")


    target_variable = config["target_variable"]
    remove_variable = config["remove_variable"]
    predicting_variables = config["predicting_variables"]

    best_RBCs = []

    for i, predicting_variable in enumerate(predicting_variables):
        best_RBC = dict()

        if config["out_extend"]:
            text = '|'.join(element for element in predicting_variable)
            best_RBC["vars"] = text
        else:
            text = None

        # 0. Call main class
        # test = Regression_based(end_output=text, config=config)
        
        if function["manual_assign_group_index"] == "Active":
            # 8
            test.manual_assign_group_index(target_variable=target_variable,
                                           remove_variable=remove_variable,
                                           train_variable=predicting_variable)
            # 9
            test.confusion_matrix_2(target_variable=target_variable,
                                    remove_variable=remove_variable,
                                    train_variable=predicting_variable)


        if function["RBC_score_var"] == "Active":
            config["visualize"] = "Dimiss"

            data = []
            n_samples = 1
            log_df = pd.DataFrame(columns=np.arange(n_samples))
            data_df = pd.read_csv(config["input_file"], index_col=0)
            for i in range(n_samples):
                drow = dict()
                # test = Regression_based(end_output=text, config=config)

                this_group_index = rand_gidx_sampling(n_cluster=config["n_cluster"], length=len(data_df.index))

                config["predicting_variable"] = predicting_variable
                # # normalize data_df
                X_norm = get_Xnorm(X_matrix=data_df[predicting_variable].values)
                data_df[predicting_variable] = X_norm


                gidx, score, score_std, err, weight_all_score = revise_KMEANs_by_LASSO(
                    init_gidx=this_group_index, data_df=data_df,
                    params=config)

                # # get full information
                all_score_matrix, group_predict = confusion_matrix(
                            df=data_df, inst_idx=data_df.index,
                            group_index=gidx, pv=predicting_variable, tv=config["target_variable"], 
                            result_dir=config["out_dir"],
                            params=config)
                
                print (np.array(group_predict))
                # # calculate RBC score
                a = np.fabs(np.array(all_score_matrix))
                min_diag = min(np.diagonal(a))
                np.fill_diagonal(a, -np.inf)
                max_off_diag = a.max()
                RBC_score = np.log((1-min_diag)/min_diag) + float(max_off_diag)

                drow["index"] = i
                drow["min_diag"] = min_diag
                drow["max_off_diag"] = max_off_diag 
                drow["RBC_score"] = RBC_score

                # get "RBC_best_time" for automatic step
                if i == 0:
                    RBC_score_min = RBC_score
                    RBC_score_max = score
                    score_gs = np.diagonal(np.fabs(np.array(all_score_matrix)))

                    RBC_ws_max = weight_score(scores=score_gs, group_predict=group_predict)
                else:
                    if RBC_score < RBC_score_min:
                        RBC_score_min = RBC_score
                        # confusion matrix of max case, and score max
                        RBC_score_max = score
                        score_gs = np.diagonal(np.fabs(np.array(all_score_matrix)))
                        RBC_ws_max = weight_score(scores=score_gs, group_predict=group_predict)
                log_df[i] = group_predict
                log_df.to_csv("{0}/log_sampling_{1}.csv".format(config["out_dir"], config["n_cluster"]))
                data.append(drow)


            score_df = pd.DataFrame(data=data)
            score_df.to_csv("{0}/RBC_score_var_{1}.csv".format(config["out_dir"], config["n_cluster"]))


            best_RBC["R2_score"] = RBC_score_max
            best_RBC["R2_score_weight"] = RBC_ws_max
            for i in range(len(score_gs)):
                best_RBC["group_{0}".format(i)] = score_gs[i]
            best_RBCs.append(best_RBC)
        out_df = pd.DataFrame(data=best_RBCs)
        out_df.to_csv("{0}/{1}".format(config["out_dir"],config["out_conclusion"]))

        if function["analyze_RBC_score_var"] == "Active":

            log_file = "{0}/log_sampling_{1}.csv".format(config["out_dir"], config["n_cluster"])

            test.PCA_inst_group_(log_file=log_file,
                            train_variable=predicting_variable,
                            target_variable=target_variable,
                            remove_variable=remove_variable)



        if function["predict_test_rrbc"] == 'Active':
            if config["initial_method"] == "manual":
                test.dir += '/manual_{0}/final'.format(config["n_cluster"])

            y_test_predict = test.predict_test_rrbc(target_variable=target_variable,
                                       remove_variable=remove_variable,
                                       train_variable=predicting_variable)























