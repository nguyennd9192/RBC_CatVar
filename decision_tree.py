import pandas
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

import matplotlib.pylab as plt
import numpy as np
import collections, pydotplus
import graphviz as gv

styles = {
    'graph': {
        'label': 'Tc decision tree',
        'fontname': 'Courier',
        'fontsize': '16',
        'fontcolor': 'black',
        'bgcolor': 'white',
        #'rankdir': 'BT',
    },
    'nodes': {
        'fontname': 'Courier',
        'shape': 'ellipse',
        'fontcolor': 'black',
        'color': 'black',
        'style': 'filled',
        'fillcolor': 'white',
    },
    'edges': {
        #'style': 'dashed',
        'color': 'black',
        'arrowhead': 'open',
        'fontname': 'Courier',
        'fontsize': '12',
        'fontcolor': 'black',
    }
}

def combine_decision_tree(input_file, tree_vars, trans_file, tree_file,  n_cv=101):

    df = pandas.read_csv(input_file, index_col=0)
    group_obs = df["group_index"]
    data = df
    n_cluster = max(group_obs) + 1
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      splitter='best', # best, random
                                      #min_samples_split=0.1,
                                      max_depth=5,
                                      random_state=9192,
                                      #min_samples_leaf=10
                                        )

    X = []
    le = preprocessing.LabelEncoder()
    num_data = ["A_e_negativity", "A_boiling_point", "A_first_ionization", "A_valence_e",
                "B_e_negativity", "B_boiling_point", "A_Z", "B_Z", 'B_melting_point',
                'B_row', 'A_row', 'mass_A', 'mass_B',
                "B_first_ionization", "B_valence_e", "C_R",
                "Density",  "diff_elecneg_A_B", 'period_B', 'period_A', 'atom_orbit_B_plus_diff_elecneg_A_B',
                'elec_neg_diff']
    # add different of electron negativity difference
    df['elec_neg_diff'] = np.abs(df['A_e_negativity'] - df['B_e_negativity'])
    tree_vars.append('elec_neg_diff')

    for tree_var in tree_vars:
        y = df[tree_var]
        if tree_var not in num_data:
            y = le.fit_transform(y)
            df['{0}_transform'.format(tree_var)] = y
        X.append(y)
        #occ_dummies = pandas.get_dummies(df[tree_var], drop_first=True)
        #data = pandas.concat([data.drop(tree_var, axis=1), occ_dummies], axis=1)

    X = np.array(X).T
    #group_predict = cross_val_predict(estimator=clf, X=X, y=group_obs, cv=n_cv)

    clf.fit(X, group_obs)
    group_predict = clf.predict(X)

    print classification_report(group_obs, group_predict)
    clf = clf.fit(X, group_obs)

    df.to_csv(trans_file)

    dot_data = tree.export_graphviz(clf,
                         feature_names=tree_vars,
                         max_depth=4,
                         #out_file='Tc_tree.dot',
                                    out_file=None,
                         filled=True, rounded=True,
                         class_names=['Group {0}'.format(i+1) for i in range(n_cluster)]
                         )
    graph = pydotplus.graph_from_dot_data(dot_data)
    colors = ('Red', 'Blue', 'Orange')
    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))


    #for edge in edges:
    #    edges[edge].sort()
    #    for i in range(n_cluster):
    #        dest = graph.get_node(str(edges[edge][i]))[0]
    #        dest.set_fillcolor(colors[i])

    graph.write_png(tree_file)


def add_nodes(graph, nodes):
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph

def add_edges(graph, edges):
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph


def apply_styles(graph, styles):
    graph.graph_attr.update(
        ('graph' in styles and styles['graph']) or {}
    )
    graph.node_attr.update(
        ('nodes' in styles and styles['nodes']) or {}
    )
    graph.edge_attr.update(
        ('edges' in styles and styles['edges']) or {}
    )
    return graph

def Tc_graph(outfile):
    g1 = gv.Digraph(format='pdf')

    g1.node('3.2', label="{Group 3|{P:6, N:0}}")

    g1 = add_nodes(g1, [
        ('1', {'label': 'T_metal'}),
        ('2.1', {'label': 'C_R'}),
        ('2.2', {'label': 'T_metal'}),
        ('3.1', {'label': 'C_R'}),
        ('3.2', {'label': 'Group 3 \n P:6, N:0'}),
        #('3.2', {'label': '{Group 3|{P:6, N:0}}'}),

        ('3.3', {'label': 'Group 2 \n P:19, N:7'}),
        ('3.4', {'label': 'T_metal'}),
        ('4.1', {'label': 'Group 1 \n P:15, N:0'}),
        ('4.2', {'label': 'C_R'}),
        ('4.3', {'label': 'Group 1 \n P:10, N:1'}),
        ('4.4', {'label': 'Group 3 \n P:20, N:5'}),
        ('5.1', {'label': 'Group 2 \n P:4, N:3'}),
        ('5.2', {'label': 'Group 3 \n P:10, N:1'}),
    ])


    g1 = add_edges(g1, [
                    (('1', '2.1'), {'label': '=Co'}),
                    (('1', '2.2'), {'label': '!=Co'}),
                    (('2.1', '3.1'), {'label': '<=0.027'}),
                    (('2.1', '3.2'), {'label': '>0.027'}),
                    (('2.2', '3.3'), {'label': '=Fe'}),
                    (('2.2', '3.4'), {'label': '!=Fe'}),
                    (('3.1', '4.1'), {'label': '<=0.012'}),
                    (('3.1', '4.2'), {'label': '>0.012'}),
                    (('3.4', '4.3'), {'label': '=Mn'}),
                    (('3.4', '4.4'), {'label': '!=Mn'}),
                    (('4.2', '5.1'), {'label': '<=0.02'}),
                    (('4.2', '5.2'), {'label': '>0.02'}),
                    #({'label': ''}),
                    ])

    styles['graph']['label'] = ''
    g1 = apply_styles(g1, styles)


    g1.render(outfile)


def AB_graph(outfile):
    g1 = gv.Digraph(format='pdf')
    g1 = add_nodes(g1, [
        ('1', {'label': 'B_period <= 3'}),

        ('2.1', {'label': 'B_period <= 2'}),
        ('2.2', {'label': 'B_type \n in {Halogen, Metaloid, P}'}),

        ('3.1', {'label': 'B_type \n in {Halogen, Metaloid, N, C}'}),
        ('3.2', {'label': 'A_type = Lanthanoid'}),
        ('3.3', {'label': 'Group 1 \n P: 23, N: 9'}),
        ('3.4', {'label': 'B_period <= 4'}),

        ('4.1', {'label': 'B_type \n in {Halogen, Metaloid}'}),
        ('4.2', {'label': 'Group 2 \n P: 13, N: 10'}),
        ('4.3', {'label': 'Group 2 \n P: 22, N: 6'}),
        ('4.4', {'label': 'Group 1 \n P: 11, N: 13'}),
        ('4.5', {'label': 'Group 3 \n P: 16, N: 11'}),
        ('4.6', {'label': 'Group 1 \n P: 19, N: 7'}),

        ('5.1', {'label': 'Group 3 \n P: 30, N: 16'}),
        ('5.2', {'label': 'A_period <= 7'}),

        ('6.1', {'label': 'Group 3 \n P: 12, N: 5'}),
        ('6.2', {'label': 'Group 1 \n P: 8, N: 7'}),
    ])

    g1 = add_edges(g1, [
        (('1', '2.1'), {'label': 'True'}),
        (('1', '2.2'), {'label': 'False'}),
        ('2.1', '3.1'), ('2.1', '3.2'), ('2.2', '3.3'), ('2.2', '3.4'),
        ('3.1', '4.1'), ('3.1', '4.2'),
        ('3.2', '4.3'), ('3.2', '4.4'),

        ('3.4', '4.5'), ('3.4', '4.6'),
        ('4.1', '5.1'), ('4.1', '5.2'),
        ('5.2', '6.1'), ('5.2', '6.2'),
    ])

    styles['graph']['label'] = ''
    g1 = apply_styles(g1, styles)
    g1.render(outfile)


def latt_const_graph(outfile):
    g1 = gv.Digraph(format='pdf')

    g1 = add_nodes(g1, [
        ('1', {'label': 'Period_A <= 2'}),
        ('2.1', {'label': 'Period_B <= 3'}),
        ('2.2', {'label': 'Period_A <= 3'}),

        ('3.1', {'label': 'Period_A = 1'}),
        ('3.2', {'label': 'Group 1: \n P: 339 N: 126'}),
        ('3.3', {'label': 'Density <= 13.88'}),
        ('3.4', {'label': 'Diff_elecneg_AB < 191.089'}),

        ('4.1', {'label': 'Group 2: \n P: 99 N: 80'}),
        ('4.2', {'label': 'Group 2: \n P: 239 N: 120'}),
        ('4.3', {'label': '(..)'}),
        ('4.4', {'label': '(..)'}),

        ('4.5', {'label': 'Group 1: \n P: 202 N: 37'}),
        ('4.6', {'label': 'Group 3: \n P: 20 N: 8'}),
        ('4.7', {'label': 'Group 3: \n P: 135 N: 129'}),
        ('4.8', {'label': 'Group 1: \n P: 4 N: 0'}),
        ('5.1', {'label': '(..)'}),
        ('5.2', {'label': '(..)'}),
        ('5.3', {'label': '(..)'}),
        ('5.4', {'label': '(..)'}),
        ('5.5', {'label': '(..)'}),
        ('5.6', {'label': '(..)'}),
        ('5.7', {'label': '(..)'}),
        ('5.8', {'label': '(..)'}),
        ('5.9', {'label': '(..)'}),
        ('5.10', {'label': '(..)'}),

    ])

    g1 = add_edges(g1, [
        (('1', '2.1'), {'label': 'True'}),
        (('1', '2.2'), {'label': 'False'}),
        ('2.1', '3.1'), ('2.1', '3.2'),
        ('2.2', '3.3'), ('2.2', '3.4'),

        ('3.1', '4.1'), ('3.1', '4.2'),
        ('3.2', '4.3'), ('3.2', '4.4'),
        ('3.3', '4.5'), ('3.3', '4.6'),
        ('3.4', '4.7'), ('3.4', '4.8'),

        ('4.1', '5.1'), ('4.1', '5.2'),
        ('4.2', '5.3'), ('4.2', '5.4'),
        #('4.3', '5.5'), ('4.3', '5.6'),
        #('4.4', '5.7'), ('4.4', '5.8'),
        ('4.5', '5.5'), ('4.5', '5.6'),
        ('4.6', '5.7'), ('4.6', '5.8'),
        ('4.7', '5.9'), ('4.7', '5.10'),
        #('4.8', '5.1'), ('4.8', '5.2'),

    ])

    styles['graph']['label'] = ''
    g1 = apply_styles(g1, styles)
    g1.render(outfile)
if __name__ == '__main__':

    # 'input/ABcompound/data_ABcompound_group_index.csv', 'input/ABcompound/data_ABcompound_trans.csv', 'AB_tree.png'

    #combine_decision_tree(input_file = 'input/Tc/TC_data_101_max_gidx.csv',  #
    #                    trans_file = 'input/Tc/TC_data_101_max_trans.csv',
    #                    tree_file = 'Tc_tree_2.png',
    #                    tree_vars = ['C_R', 'R_metal', 'T_metal'], n_cv = 10)

    combine_decision_tree(input_file='input/ABcompound/data_ABcompound_gidx5.csv', #
                          trans_file='input/ABcompound/data_ABcompound_trans.csv',
                          tree_file='AB_tree.png',
                          tree_vars=['volumne',
                                     'B_valence_e',
                                     'B_e_negativity', 'A_e_negativity',
                                     'A_row',
                                     'A_type', 'B_type',
                                     'A_Z', 'B_Z',
                                     'B_melting_point',
                                     #'B_block',
                                     'A_valence_e',
                                     #'Formation_energy'
                                     ],
                          n_cv=50)


    #combine_decision_tree(input_file='input/latt_const/SVRdata_gidx.csv',
    #                      trans_file='input/latt_const/SVRdata_trans.csv',
    #                      tree_file='latt_const_tree.png',
    #                      tree_vars=[ 'period_B',
    #                                  'period_A',
    #                                  'Density',
                                      #'group_A', 'group_B',
                                      #'atom_A','atom_B',
                                      #'mass_A', 'mass_B',
                                      #'Atomic_Number_A', 'Atomic_Number_B',
                                      #'B_type',# 'A_type',
                                      #'A_block', 'B_block',
                                      #'density_A', 'density_B',
                                      #'diff_elecneg_A_B',
                                      #'group_index3'
                                      #'atom_orbit_B_plus_diff_elecneg_A_B'
    #                                ],
    #                     n_cv=10)

    #Tc_graph(outfile='Tc_tree_manual_2')

    #AB_graph(outfile='AB_tree_manual')

    #latt_const_graph(outfile='latt_const_manual')