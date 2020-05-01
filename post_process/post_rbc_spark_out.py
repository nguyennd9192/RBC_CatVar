

import pandas as pd
import numpy as np
import shutil
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from itertools import combinations


from lib.general_lib import makedirs
from visualize_sim_mtrx import clustering, get_circular_tree

clr = ["red", "blue", "orange", "green", "brown"]

def get_score_min():
    input_dir = "latt_const_g3" # spark_result_AB, latt_const_g3
    source_df = pd.read_csv("{0}/latt_const_g3_100.csv".format(input_dir), index_col=0)
    # data_ABcompound, latt_const_g3_5
    df = pd.read_csv("{0}/latt_const_g3_100_full.csv.nc2.out.csv".format(input_dir))
    # data_ABcompound.csv.nc3.out, latt_const_g3_5.csv.nc2.out

    df_sort = df.sort_values(by="RBB_score", axis="index") # RBB_score
    print (df_sort.head())
    print (list(df_sort.RBB_score)[0]) # RBB_score
    group_index = list(df_sort.group_index)[0]

    str_to_array = group_index.replace('[', '').replace(']', '').split(", ")
    source_df["group_index"] = str_to_array
    source_df.to_csv("{0}/revise_group_2.csv".format(input_dir))

    print (group_index.replace('[', '').replace(']', '').split(", "))


def get_k(merge_file):

    df = pd.read_csv(merge_file)

    df = df.dropna(how="any")
    df = df.sort_values(by="label", axis="index")

    # we have two criterias here, n_cluster and combination
    n_clusters = set(df.n_cluster)

    for nc in n_clusters:
        sub_df = df[df.n_cluster == nc]

        print (sub_df.loc[comb_df.RBB_score.idxmin(), "label"], comb_df.RBB_score.min(axis='index'))
        #break


def merge_df(path, merge_file):
    all_files = glob.glob(pathname=path + "/*.csv")
    with open(merge_file, "wb") as out_file:
        for i, fname in enumerate(all_files):
            add_ncluster(outfile=fname)
            with open(fname, 'rb') as infile:
                if i != 0:
                    infile.readline()
                shutil.copyfileobj(infile, out_file)


def str2array(string):
    str_to_array = string.replace('[', '').replace(']', '').split(", ")
    return str_to_array


def get_topk(source_file, score_file, out_file, top_k):
    # sort clustering results, by the RBB score
    df = pd.read_csv(score_file)
    df = df.dropna(how="any")
    df = df.sort_values(by="RBB_score", axis="index")

    df_top = df.head(top_k)
    g_idxes = list(df_top.group_index)
    print (df_top)

    source_df = pd.read_csv(source_file, index_col=0)

    # for re labeling
    # new_label = {k : "{0}_{1}".format(idx, k) for idx, k in enumerate(source_df.columns)}
    # source_df = source_df.rename(index=new_label)
    # source_df.to_csv(source_file)


    out_df = pd.DataFrame(index=source_df.index)
    for ith, gidx in enumerate(g_idxes):
        g_vector = str2array(string=gidx)
        out_df.loc[:, "result{}".format(ith)] = g_vector
    makedirs(out_file)
    out_df.to_csv(out_file)


def get_similarity_df(gidx_file, similarity_out):

    # gidx_file: with indexes: contains source data instances; 
    # gidx_file: columns: "result0", "result1"... as Regression based clustering results
    

    gidx_df = pd.read_csv(gidx_file, index_col=0)
    instances = list(gidx_df.index)
    n_instances = len(instances)
    gidx_columns = gidx_df.columns

    similarity_df = pd.DataFrame(0, index=instances, columns=instances)

    for col in gidx_columns:
        for i in range(n_instances):
            inst_i = instances[i]
            this_g_i = gidx_df.loc[inst_i, col]
            for j in range(i, n_instances):
                inst_j = instances[j]
                this_g_j = gidx_df.loc[inst_j, col]
                if this_g_i == this_g_j:
                    similarity_df.loc[inst_i][inst_j] +=1
                    similarity_df.loc[inst_j][inst_i] = similarity_df.loc[inst_i][inst_j]
    print (similarity_df)
    similarity_df.to_csv(similarity_out)



def get_similarity_df2(gidx_file, similarity_out):

    # gidx_file: with indexes: contains source data instances; 
    # gidx_file: columns: "result0", "result1"... as Regression based clustering results
    

    gidx_df = pd.read_csv(gidx_file, index_col=0)
    instances = list(gidx_df.index)
    n_instances = len(instances)
    gidx_columns = gidx_df.columns

    similarity_df = pd.DataFrame(0, index=instances, columns=instances)

    for col in gidx_columns:
        lbl_array = gidx_df[col].values
        lbls = set(lbl_array)
        # print (lbl_array, np.where(lbl_array == 4)[0])
        # tmp_sim_df = pd.DataFrame(0, index=instances, columns=instances)
        tmp_sim = np.zeros((n_instances, n_instances))


        for lbl in lbls:
            # grp_inst = list(gidx_df[gidx_df[col] == lbl].index)

            grp_inst = list(np.where(gidx_df[col] == lbl)[0])

            grp_inst_pwise = list(combinations(grp_inst, 2))
            tmp_idx = [k[0] for k in grp_inst_pwise]
            tmp_col = [k[1] for k in grp_inst_pwise]

            print (lbl)
            tmp_sim[tmp_idx, tmp_col] = 1

        # get symmetrical
        tmp_sim += tmp_sim.T
        similarity_df += tmp_sim
    np.fill_diagonal(similarity_df.values, len(gidx_columns))

    similarity_df.to_csv(similarity_out)




def get_submatrix(optimize_file, considered_indexes, out_file, source_f, main_dir):

    # optimize_file contains a square affinity matrix that already re-arrange follow some order in HAC
    # for Tc

    new_label = []
    for k in considered_indexes:
        this_name = str(k)[:str(k).find("-")]
        if this_name == "Mn23Tb16":
            this_name = "Mn23Tb6"
        new_label.append(this_name)
    optimize_df = pd.read_csv(optimize_file, index_col=0)

    c_indexes = new_label
    optimize_df = optimize_df.loc[c_indexes, c_indexes]

    #new_label = {k : str(k)[: str(k).find("-")] for k in optimize_df.columns}
    #optimize_df = optimize_df.rename(index=new_label, columns=new_label)

    optimize_df.to_csv(out_file.replace(".pdf", ".csv"))
    s_mt = np.array(optimize_df.values, dtype=float)
    max = np.max(s_mt)
    s_mt = (s_mt / float(max))
    plt.figure(figsize=(11, 10))


    ax = sns.heatmap(s_mt, cmap='YlOrBr',
                     #xticklabels=labels_Tc,
                     #yticklabels=labels_Tc,
                     xticklabels=list(optimize_df.index),
                     yticklabels=list(optimize_df.index)
                     )
    #ax.set_xticks([])
    #ax.set_yticks([])


    sc_f = pd.read_csv(source_f, index_col=0)
    sc_f = sc_f.loc[considered_indexes, :]
    sc_f.to_csv("{0}/subdata.csv".format(main_dir))



    for item in ax.get_yticklabels():
        item.set_fontsize(10)
        item.set_rotation(0)
        item.set_fontname('serif')

    for item in ax.get_xticklabels():
        item.set_fontsize(10)
        item.set_fontname('serif')
        item.set_rotation(90)
    plt.savefig(out_file)


def plot_latt_noble(similarity_file, main_dir, n_cluster, top_k, font_size=4):
    similarity_df = pd.read_csv(similarity_file, index_col=0)
    noble_gas = ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]
    print_labels = []
    for name in similarity_df.index:
        #out_name = ""
        for n_gs in noble_gas:
            if str(name).find(n_gs) != -1:
                out_name = name
                print_labels.append(out_name)

    similarity_df = similarity_df.loc[print_labels, :]
    s_mt = np.array(similarity_df.values, dtype=float)
    max = np.max(s_mt)
    s_mt = (s_mt / float(max))


    plt.figure(figsize=(8, 8))

    y_label = [str(k)[str(k).find("_") + 1:] for k in similarity_df.index]

    ax = sns.heatmap(s_mt, cmap='YlOrBr',
                     # xticklabels=labels_Tc,
                     # yticklabels=labels_Tc,
                     #xticklabels=list(similarity_df.index),
                     yticklabels=y_label
                     )
    for item in ax.get_yticklabels():
        item.set_fontsize(font_size)
        item.set_rotation(0)
        item.set_fontname('serif')

    for item in ax.get_xticklabels():
        item.set_fontsize(font_size)
        item.set_fontname('serif')
        item.set_rotation(90)
    plt.savefig("{0}/noble_map_{1}_top_{2}.pdf".format(str(main_dir).split(".")[0], n_cluster, top_k))


def add_ncluster(outfile):
    outfile = str(outfile)
    df = pd.read_csv(outfile)
    k = outfile.find(".out.csv")
    n_cluster = outfile[outfile.find("nc") +2 :k]
    print (n_cluster)
    df["n_cluster"] = n_cluster
    df.to_csv(outfile)
    print (df.head())


if __name__ == "__main__":
    merge_file = "merge_Df.csv"
    #get_score_min()


    #add_ncluster(outfile="latt_const_g3/latt_const_g3_5.csv.nc3.out.csv")

    # merge_df to merge multiple regression-based result with different n_cluster
    #merge_df(path="spark_result", merge_file=merge_file)


    #get_k(merge_file=merge_file)


    config_AB = {"source_f": "data_ABcompound",
                 "main_dir": "spark_result_AB",
                 "n_cluster": 3,
                 "top_k": 200
                 }
    config_Tc = {"source_f": "TC_data_101_max",
                 "main_dir": "spark_result_Tc",
                 "n_cluster": 3,
                 "top_k": 50
                 }
    config_latt = {"source_f": "SVRdata",
                   "main_dir": "spark_result_latt_const",
                   "n_cluster": 3,
                   "top_k": 20
                 }

    config_thermal = {"source_f": "thermal_ofm_Tp",
                      "main_dir": "spark_result_thermal",
                      "n_cluster": 3,
                      "top_k": 20
                      }

    config_lines = {"source_f": "4_SierraCurve", # 1_CrossLine, 2_ParallelLine, 3_SineCurve, 4_SierraCurve
                      "main_dir": "spark_result_lines",
                      "n_cluster": 3,
                      "top_k": 20
                      }

    config_superconductivity = {"source_f": "sc_tmp_des",
                      "main_dir": "superconductivity",
                      "n_cluster": 4,
                      "top_k": 100
                      }

    config_OQMD_Co = {"source_f": "Co_des_rm_high_val",
                      "main_dir": "../input/OQMD/Co_nsp10k", # Co_nsp1k
                      "result_dir": "../result/OQMD/Co_nsp10k", # Co_nsp1k
                      "n_cluster": 5,
                      "top_k": 500
                      }


    config = config_OQMD_Co # config_AB, config_latt, config_thermal
    

    is_get_topk = False
    is_get_similarity_df = False
    is_clustering = False

    is_isomap = True
    is_visualize_large_matrix = False # visualize large matrix by a list of sub matrixes


    # 1. get top clustering results
    if is_get_topk:
        get_topk(source_file="{0}/{1}.csv".format(config["main_dir"], config["source_f"]),
                 score_file="{0}/{1}.csv.nc{2}.out.csv".format(config["main_dir"], 
                config["source_f"], config["n_cluster"]),
                 out_file="{0}/clustering_result_{1}_top_{2}.csv".format(config["result_dir"],
                            config["n_cluster"], config["top_k"]),
                 top_k=config["top_k"])
    

    # # 2 from top clustering result, get similarity df
    if is_get_similarity_df:
        get_similarity_df2(gidx_file="{0}/clustering_result_{1}_top_{2}.csv".format(
                            config["result_dir"], config["n_cluster"], config["top_k"]),
            similarity_out="{0}/similarity{1}_top_{2}.csv".format(
                            config["result_dir"], config["n_cluster"], config["top_k"]))


    # 3 do clustering basing on similarity matrix, plot dendrogram
    if is_clustering:

        linkage_matrix, ticklabels = clustering(similarity_file="{0}/similarity{1}_top_{2}.csv".format(
            config["result_dir"], config["n_cluster"], config["top_k"]),
                       result_dir=config["result_dir"],
                       n_cluster=config["n_cluster"], is_heatmap=False,
                       save_extend="_top{0}".format(config["top_k"]),
                       font_size=3)

        get_circular_tree(linkage_matrix=linkage_matrix, 
            ticklabels=ticklabels, result_dir=config["result_dir"],
            source_file="{0}/{1}.csv".format(config["main_dir"], config["source_f"]))


    if is_isomap:
        source_file="{0}/{1}.csv".format(config["main_dir"], config["source_f"])
        source_df = pd.read_csv(source_file, index_col=0)

        # gidx_file="{0}/hac_gidx_nc{1}__top{2}.csv".format(config["result_dir"], config["n_cluster"], config["top_k"])


        gidx_file="{0}/clustering_result_{1}_top_{2}.csv".format(config["result_dir"],
                            config["n_cluster"], config["top_k"])

        gidx_df = pd.read_csv(gidx_file, index_col=0)

        # colors = gidx_df["group_index"].values
        colors = gidx_df["result0"].values


        from lib.Isomap import Iso_map_with_lbl
        from lib.plot import get_color
        from lib.normalize import get_Xnorm


        pv = ["s2", "d7", "s1-s2", "s2-s1", "s2-s2", "s2-d1", "s2-d7", 
            "s2-d10", "s2-f14", "d1-s2", "d7-s2", "d7-d10", "d10-s2", 
            "d10-d7", "f14-s2"]

        tv = "c_magmom_pa"


        # save to .csv file of sub group
        for i in range(config["n_cluster"]):
            tmp_df = source_df.loc[gidx_df["result0"] == i, :]
            tmp_df.to_csv("{0}/NLFS/group_{1}.csv".format(config["main_dir"], i))
        # colors = get_Xnorm(X_matrix=source_df[tv].values.reshape(-1, 1))
        # print (colors[0])
        # col_names = get_color(source_df=source_df, col_clr=tv)
        Iso_map_with_lbl(X=source_df[pv].values,
                save_at="{0}/Isomap.pdf".format(config["result_dir"]), 
                is_plot=True, annot_lbl=source_df.index, 
                colors=colors)
                # source_df[tv].values





    # 4 # visualize large matrix by a list of sub matrixes, if neccessary
    if is_visualize_large_matrix:
        from display_large_sim_mt import display_large_dataframe

        similarity_file="{0}/optimize_df_nc_{1}_top_{2}.csv".format(config["result_dir"],
                                                      config["n_cluster"],
                                                      config["top_k"])

        s_matrix = pd.read_csv(similarity_file, index_col=0)

        dir2save = "{0}/submt".format(config["result_dir"])
        display_large_dataframe(large_matrix=s_matrix, output_dir=dir2save,
                             is_sample=True, df_tick_name=None,size_submt=200)



    considered_indexes_Tc = [ # 1
                            "Mn23Pr6-448", "Mn23Yb6-406", "Mn23Dy6-443", "Mn23Sm6-450", "Mn23Gd6-465",
        "Mn23Tb16-454", "Mn23Nd6-438", "Mn2Ho-25", "Co2Gd-405", "Co5Pr-931", "Co5Gd-1002", "Co5Dy-977",
        "Co5Tb-979", "Co5Ce-662", "Co5La-838", "Co17Ce2-1090", "Co7Tm2-640", "Co17Sm2-1193", "Co13La-1298",
        "Co5Nd-910",
                            # 2
                             "Fe17Tm2-271", "Fe17Er2-299", "Fe17Ho2-335", "Fe5Gd-470", "Fe17Tb2-411",
        "Fe17Ce2-233", "Fe17Pr2-280", "Fe17Sm2-395", "Fe17Nd5-303", "Fe17Gd2-479", "Fe17Nd2-327",
                            # 3
                             "Ni2Dy-28", "Ni2Ho-16", "Ni5Er-11", "Ni5Nd-7", "Ni5Pr-0", "Ni17Dy2-170",
        "Ni17Gd-189", "Co2Tm-4", "CoHo3-10", "CoTb3-77", "CoGd3-143", "CoSm3-78", "Co2Er-39", "Co2Pr-45",
        "Co7Ce2-50", "Co2Nd-108", "Co2Dy-141", "CoEr3-7", "Co2Ho-83"
                            # 4
                             ]



    # get_submatrix(optimize_file="{0}/optimize_df_nc_{1}_top_{2}.csv".format(config["main_dir"],
    #                                                           config["n_cluster"],
    #                                                            config["top_k"]),
    #            considered_indexes=considered_indexes_Tc,
    #            out_file="{0}/submatrix_top_{1}.pdf".format(config["main_dir"],
    #                                                        config["top_k"]),
    #              source_f="input/Tc/{0}.csv".format(config["source_f"]), # replace source dir of source file
    #              main_dir=config["main_dir"]
    #              )

   














