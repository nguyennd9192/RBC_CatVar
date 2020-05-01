import numpy as np
import pandas as pd


axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}
size_text = 6
alpha_point = 0.6
size_point = 80
option = {
    1: 'blue',
    2: 'orange',
    3: 'green',
    4: 'brown',
    5: 'violet',
    6: 'grey',
    7: 'ivory'
}


def rand_gidx_sampling(n_cluster, length):
    group_index = np.random.randint(low=0, high=n_cluster, size=length)
    return group_index

def result_sampling(files, n_clusters, filesave):
    fig = plt.figure(figsize=(8, 8))

    for file, n_cluster in zip(files, n_clusters):
        this_df = pd.read_csv(file, index_col=0)

        x = this_df["min_diag"]
        y = this_df["max_off_diag"]
        name = this_df.index

        for i, n in enumerate(name):
            plt.annotate(name[i], xy=(x[i], y[i]), size=size_text)
        plt.scatter(x, y, s=size_point, alpha=alpha_point, c=option[n_cluster], label=n_cluster)
    plt.xlabel('Min score diagonal', **axis_font)
    plt.ylabel('Max score off diagonal', **axis_font)
    plt.legend(loc=2, fontsize='small')

    plt.savefig(filesave)



if __name__ == "__main__":
    group_index = rand_gidx_sampling(n_cluster=2, length=20)
    print (group_index)


    dir = "Tc" # ABcompound, latt_const_, Tc
    n_clusters = [2, 3, 4]
    files = [ "{0}/RBB_score_var_{1}.csv".format(dir, k) for k in n_clusters]
    filesave = '{0}/RBC_score.pdf'.format(dir)
    result_sampling(files=files, n_clusters=n_clusters, filesave=filesave)








































