import time
import pandas as pd
import numpy as np
from read_load_config import *
import glob
import itertools
from lib.weight_score import weight_score
from lib.ini_sampling import rand_gidx_sampling
from lib.lasso_in_cluster import revise_KMEANs_by_LASSO
from lib.confusion_matrix import confusion_matrix

def main_spark():
    t1 = time.time()

    # read_load_config
    print ("Start reading config")
    print ("========================")
    cf = load_config()
    config, function = read_config(cf)

    for key, value in config.items():
        print ("{0}: {1}".format(key, value))
    print ("Sucess in reading config")
    print ("========================")

    target_variable = config["target_variable"]
    remove_variable = config["remove_variable"]
    predicting_variables = config["predicting_variables"]

    data_df = pd.read_csv(config["input_file"], index_col=0)
    n_instance = len(data_df.index)



    inits =[]
    if function["RBC_score_var"] == "Active":
        config["visualize"] = "Dimiss"

        for i in range(config["RBC_ntimes"]):
            this_group_index = rand_gidx_sampling(n_cluster=config["n_cluster"], length=n_instance)
            inits.append(this_group_index)


    elements = list(itertools.product(predicting_variables, inits))


    for i_th, element in enumerate(elements):
        #try:
        result_dict = spark_cal(data_df=data_df,
                  element=element, target_variable=target_variable,
                  config=config, index_job=i_th)
        #except:
        #    result_dict = np.nan
        print (result_dict)
        #break


def clean_unneed(file_path):
    if os.path.isfile(file_path):
        # Verifies CSV file was created, then deletes unneeded files.
        head, tail = os.path.split(file_path)
        for CleanUp in glob.glob("{0}/*.*".format(head)):
            if not CleanUp.endswith(tail):
                os.remove(CleanUp)


def spark_cal(data_df, element, target_variable, config, index_job):
    # input:
    # data_df: a source data frame
    # element: element[0] stores predicting variables combinations, element[1] stores initial state of

    this_pv = element[0]
    this_init_state = element[1]

    out_dir_org = config["out_dir"]
    tmp = '|'.join(element for element in this_pv)
    text = "/{0}_nc{1}_ith{2}".format(tmp, config["n_cluster"], index_job)
    config["out_dir"] += text


    return_dict = dict()

    # # #
    # # # RBC kernel
    config["predicting_variable"] = this_pv
    gidx, score, score_std, err, weight_all_score = revise_KMEANs_by_LASSO(
                    init_gidx=this_init_state, data_df=data_df,
                    params=config)

    # # get full information of confusion matrix
    all_score_matrix, group_predict = confusion_matrix(df=data_df, 
                inst_idx=data_df.index, group_index=gidx, 
                pv=config["predicting_variable"] , tv=config["target_variable"], 
                result_dir=config["out_dir"],
                params=config)




    # run = Regression_based(data_df=data_df, end_output=text, config=config)
    # s, std, this_best_error = run.revise_KMEANs_by_LASSO(target_variable=target_variable,
    #                                                       remove_variable=config["remove_variable"],
    #                                                       train_variable=this_pv,
    #                                                       manual_group_predict=this_init_state,
    #                                                       new_dir=False)


    # all_score_matrix, group_predict = run.confusion_matrix_2(target_variable=target_variable,
    #                                                           remove_variable=config["remove_variable"],
    #                                                           train_variable=this_pv)

    a = np.fabs(np.array(all_score_matrix))

    min_diag = min(np.diagonal(a))
    np.fill_diagonal(a, -np.inf)
    max_off_diag = a.max()
    return_dict["min_diag"] = min_diag
    return_dict["max_off_diag"] = max_off_diag

    # log_df = pd.DataFrame(index=data_df.index)
    # log_df["group_index"] = group_predict
    save_file = "{0}/out_gidx.csv".format(config["out_dir"])
    # log_df.to_csv(save_file)

    RBC_score = np.log((1 - min_diag) / min_diag) + float(max_off_diag)
    return_dict["RBC_score"] = RBC_score
    return_dict["final_group_file"] = save_file

    config["out_dir"] = out_dir_org
    clean_unneed(file_path=save_file)

    return return_dict

if __name__ == '__main__':
    main_spark()




















