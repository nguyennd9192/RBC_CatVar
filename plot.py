import numpy as np
import gc
import matplotlib.pylab as plt

option = {
    0: 'red',
    1: 'blue',
    2: 'orange',
    3: 'green',
    4: 'brown',
    5: 'violet',
    6: 'grey',
    7: 'ivory'
}

option_cmap = {
    0: 'Reds',
    1: 'Blues',
    2: 'Oranges',
    3: 'Greens',
    4: 'brown',
    5: 'IlOrBr',
    6: 'Greys',
    7: 'BuGn'
}


axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}
size_text = 6
alpha_point = 0.6
size_point = 80

def release_mem(fig):
    fig.clf()
    plt.close()
    gc.collect()

def set_plot_configuration(x, y, config, size_fig=None):
    target_variable = config["target_variable"]
    y_min = min([min(x), min(y)])
    y_max = max([max(x), max(y)])
    y_mean = (y_max + y_min) / 2.0
    y_std = (y_max - y_mean) / 2.0
    y_min_plot = y_mean - 2.4 * y_std
    y_max_plot = y_mean + 2.4 * y_std

    # threshold = 0.1
    # plt.plot(x_ref, x_ref * (1 + threshold), 'g--', label=r'$\pm 10 \%$')
    # plt.plot(x_ref, x_ref * (1 - threshold), 'g--', label='')

    plt.ylim([y_min_plot, y_max_plot])
    plt.xlim([y_min_plot, y_max_plot])
    if size_fig == None:
        plt.ylabel(r'%s predicted (%s)' % (target_variable, config["dimensional"]), **axis_font)
        plt.xlabel(r'%s observed (%s)' % (target_variable, config["dimensional"]), **axis_font)
        plt.legend(loc=2, fontsize='small')

    return y_min_plot, y_max_plot


def plot_regression(x, y, config, n_cluster=None, group_index=None, name=None,):
    target_variable = config["target_variable"]
    fig = plt.figure(figsize=(8, 8))
    if group_index is None:
        plt.scatter(x, y, s=size_point, alpha=alpha_point, c='blue', label=None)
        y_min_plot, y_max_plot = set_plot_configuration(x=x, y=y, config=config)
        x_ref = np.linspace(y_min_plot, y_max_plot, 100)
        plt.plot(x_ref, x_ref, linestyle='-.', c='red', alpha=0.8)

        if name is not None:
            for i in range(len(name)):
                plt.annotate(str(name[i]), xy=(x[i], y[i]), size=size_text)

    else:
        if isinstance(group_index, int) == True:
            plt.scatter(x, y, c=option[group_index], s=size_point,
                        alpha=alpha_point, label='Group %d' % (group_index + 1))
            if name is not None:
                for i in range(len(x)):
                    plt.annotate(name[i], xy=(x[i], y[i]), size=size_text)
            y_min_plot, y_max_plot = set_plot_configuration(x=x, y=y,config=config)
            x_ref = np.linspace(y_min_plot, y_max_plot, 100)
            plt.plot(x_ref, x_ref, linestyle='-.', c='red', alpha=0.8)
        else:
            for this_group in range(config["n_cluster"]):
                # print this_group, np.where(group_index == this_group)
                x_plot = x[this_group]
                y_plot = y[this_group]

                plt.scatter(x_plot, y_plot, s=size_point, alpha=alpha_point,
                            c=option[this_group], label='Group %d' % (this_group + 1))
                if name is not None:
                    for i in range(len(x_plot)):
                        plt.annotate(name[this_group][i], xy=(x_plot[i], y_plot[i]), size=size_text)

            x_all = [item for sublist in x for item in sublist]
            y_all = [item for sublist in y for item in sublist]
            y_min_plot, y_max_plot = set_plot_configuration(x=x_all, y=y_all, config=config)
            x_ref = np.linspace(y_min_plot, y_max_plot, 100)

            plt.plot(x_ref, x_ref, linestyle='-.', c='red', alpha=0.8)
    return fig


def plot_PCA(x, y, c, name, group_index, config, savename, color_colum, size_column,  size=None, size_text=None):
    fig = plt.figure(figsize=(9, 8))

    if size is None:
        size = 50
    else:
        max_size = max(size)
        size = 100*(0.3 + size / max_size)

    if size_text is None:
        size_text = 4

    if isinstance(group_index, int):
        plt.scatter(x[0], y[0], c=option[group_index], s=size[0], alpha=alpha_point,  # tab20c, Oranges
                    label='Group %d' % (group_index + 1))
        plt.scatter(x, y, c=c, s=size, cmap=option_cmap[group_index], # tab20c, Oranges
                    edgecolors=None)
    else:
        plt.scatter(x, y, c=c, s=size, alpha=alpha_point, cmap='Oranges', edgecolors='face')

    #for i in range(len(c)):
            #plt.scatter(x[i], y[i], c=option[c[i]], s=size_point)
    plt.colorbar()
    if name is not None:
        for i in range(len(name)):
            this_name = str(name[i])
            this_name = this_name[this_name.find("_")+1:]
            plt.annotate(this_name, xy=(x[i], y[i]), size=size_text)
    y_min_plot, y_max_plot = set_plot_configuration(x=x, y=y, config=config)
    if size_column is not None:
        plt.title("Colored by {0} \n Denote size by {1}".format(color_colum, size_column))
    else:
        plt.title("Colored by {0}".format(color_colum))

    plt.ylabel(r'PCA 2', **axis_font)
    plt.xlabel(r'PCA 1', **axis_font)

    plt.savefig(savename)
    release_mem(fig=fig)

def plot_PCA_full(x, c, features, savename, color_column, size=None):
    fig = plt.figure(figsize=(25, 25))
    fig_idx = 0
    if size is None:
        size = 30

    n_features = len(features)
    #if n_features != x.size[1]:
    #    print "Error in plot_PCA_full !!! "
    #    print "Matrix size and n_features do not match."
    #    quit()
    for i in range(n_features):
        for j in range(n_features):
            plt.subplot(n_features, n_features, fig_idx + 1)
            if i == j:
                plt.hist(x[:, i], bins=30, histtype='bar', normed=0, rwidth=1.0)
                frame = plt.gca()
                # frame.axes.xaxis.set_ticklabels([])
                frame.axes.yaxis.set_ticklabels([])
            elif i < j:
                plt.scatter(x[:, j], x[:, i], c=c, s=size, cmap='Oranges')
                frame = plt.gca()
                frame.axes.xaxis.set_ticklabels([])
                frame.axes.yaxis.set_ticklabels([])
            else:
                plt.scatter(x[:, j], x[:, i],  c=c, s=size, cmap='Oranges')
                plt.legend()
                frame = plt.gca()
                frame.axes.xaxis.set_ticklabels([])
                frame.axes.yaxis.set_ticklabels([])

            if j == 0:
                plt.ylabel(features[i], **axis_font)
            if i == n_features:
                plt.xlabel(features[j], **axis_font)
            fig_idx += 1
    plt.title("Colored by {0}".format(color_column))

    plt.savefig(savename)
    release_mem(fig=fig)




















































