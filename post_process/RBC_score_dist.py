
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import os

axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}

def ScoreDist(source_file, score_file, out_file, top_k):
    df = pd.read_csv(score_file)
    df = df.dropna(how="any")
    # max_off_diag	min_diag
    pairs = [["max_off_diag", "min_diag"],
             ["max_off_diag", "RBB_score"],
             ["min_diag", "RBB_score"]]
    if not os.path.isdir(source_file):
        os.makedirs(source_file)
    for pair in pairs:
        plt.figure(figsize=(8,8))
        sns.jointplot(x=pair[0], y=pair[1], data=df,
                      kind="hex",
                      color='orange',
                      stat_func=None
                      )
        plt.xlabel(pair[0], **axis_font)
        plt.ylabel(pair[1], **axis_font)
        plt.savefig("{0}/{1}_{2}.pdf".format(source_file, pair[0], pair[1]))







if __name__ == "__main__":

    config_AB = {"source_f": "data_ABcompound",
                 "main_dir": "spark_result_AB",
                 "n_cluster": 3,
                 "top_k": 100
                 }
    config_Tc = {"source_f": "TC_data_101_max",
                 "main_dir": "spark_result_Tc",
                 "n_cluster": 3,
                 "top_k": 10
                 }
    config_latt = {"source_f": "SVRdata",
                   "main_dir": "spark_result_latt_const",
                   "n_cluster": 4,
                   "top_k": 20
                   }

    #config = config_AB
    config = config_Tc
    #config = config_latt

    ScoreDist(source_file="{0}/{1}".format(config["main_dir"], config["source_f"]),
             score_file="spark_result/{0}.csv.nc{1}.out.csv".format(config["source_f"],
                                                                    config["n_cluster"]),
             out_file="{0}/clustering_result_{1}_top_{2}.csv".format(config["main_dir"],
                                                                     config["n_cluster"],
                                                                     config["top_k"]),
             top_k=config["top_k"])









