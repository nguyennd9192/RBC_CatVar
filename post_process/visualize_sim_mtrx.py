
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster

from lib.general_lib import *
from lib.plot import get_color



def get_ticklabels(old_idx):

	new_idx = old_idx
	
	# for Tc
	# new_label = dict()
	# for k in similarity_df.columns:
	#    new_label[k] = str(k)[:str(k).find("-")]
	#    if k == "Mn23Tb16":
	#        new_label[k] = "Mn23Tb6"

	# for latt
	# new_label = {k : "{0}_{1}".format(idx, k) for idx, k in enumerate(similarity_df.columns)}
	# similarity_df = similarity_df.rename(index=new_label, columns=new_label)

	return new_idx


def clustering(similarity_file, result_dir, n_cluster, 
			save_extend, is_heatmap, font_size=4):

	similarity_df = pd.read_csv(similarity_file, index_col=0)

	s_mt = np.array(similarity_df.values, dtype=float)
	max_vl = np.max(s_mt)
	s_mt = 1 - (s_mt / float(max_vl))
	dist_condens = squareform(s_mt)


	plt.figure(figsize=(15, 8))
	ticklabels = get_ticklabels(old_idx=similarity_df.index)


	linkage_matrix = hc.linkage(dist_condens, method="ward",)  # method = 'complete'--max; 'single'==min, 'average'
	hc.set_link_color_palette(['r', 'b', 'y',])

	dendrogram = hc.dendrogram(linkage_matrix,
		   labels=ticklabels, leaf_rotation=90, 
		   leaf_font_size=4, color_threshold=0.5) # Tc: 4, latt: 15, AB: 4.2

	clusters = fcluster(linkage_matrix, n_cluster, criterion='maxclust')

	plt.savefig("{0}/hac_nc{1}_{2}.pdf".format(result_dir, n_cluster, save_extend))

	out_df = pd.DataFrame(index=similarity_df.index, columns=["group_index"])
	out_df["group_index"] = clusters - 1
	out_df.to_csv("{0}/hac_gidx_nc{1}_{2}.csv".format(result_dir, n_cluster, save_extend))
	#print dendrogram["ivl"]

	tmp_data = []
	for old_lbl, opt_lbl in zip(similarity_df.index, dendrogram['ivl']):
		tmp_data.append(similarity_df.loc[old_lbl, :])
	optimize_df = pd.DataFrame(tmp_data)
	optimize_df = optimize_df.reindex(dendrogram['ivl'], axis=1)
	
	save2csv = "{0}/optimize_df_nc{1}_{2}.csv".format(result_dir, n_cluster, save_extend)
	makedirs(save2csv)
	optimize_df.to_csv(save2csv)


	if is_heatmap:
		s_mt = np.array(optimize_df.values, dtype=float)
		max = np.max(s_mt)
		s_mt = (s_mt / float(max))

		plt.figure(figsize=(8,8))
		ax = sns.heatmap(s_mt, cmap='YlOrBr',
						 #xticklabels=labels_Tc,
						 #yticklabels=labels_Tc,
						 # xticklabels=list(optimize_df.index),
						 # yticklabels=list(optimize_df.index)
						 ) 

		for item in ax.get_yticklabels():
			item.set_fontsize(font_size)
			item.set_rotation(0)
			item.set_fontname('serif')

		for item in ax.get_xticklabels():
			item.set_fontsize(font_size)
			item.set_fontname('serif')
			item.set_rotation(90)
		plt.savefig("{0}/final_map_{1}_{2}.pdf".format(result_dir, n_cluster, save_extend))

	return linkage_matrix, ticklabels


def transform(x):
	return 1 - 1 / (x +1)**7




def get_circular_tree(linkage_matrix, ticklabels, result_dir, source_file):
	import networkx as nx

	try:
		import pygraphviz
		from networkx.drawing.nx_agraph import graphviz_layout
	except ImportError:
		try:
			import pydot
			from networkx.drawing.nx_pydot import graphviz_layout
		except ImportError:
			raise ImportError("This example needs Graphviz and either "
							  "PyGraphviz or pydot")

	
	source_df = pd.read_csv(source_file, index_col=0)

	G = nx.Graph()
	n_instances = len(ticklabels)
	source_indexes = source_df.index


	parent_node = n_instances

	color_map = []


	rootnode, nodelist = hc.to_tree(linkage_matrix, rd=True)
	# print (rootnode)
	# print (nodelist)
	node_dict = dict({})
	for nd in nodelist:
	#     print(nd.id, nd.get_count())
		node_dict[nd.id] = nd.get_count()

	col_names = get_color(source_df=source_df, col_clr="c_magmom_pa")

	for node_1, node_2, distance, n_ele in linkage_matrix:
		node_1 = int(node_1)
		node_2 = int(node_2)
		n_ele = int(n_ele)

	#     print (node_1, node_2, distance, n_ele)
	#     print (node_1, node_dict[node_1], node_2, node_dict[node_2])
		if node_1 < n_instances:
			G.add_node(node_1, label=ticklabels[node_1])
			# this_col = col_names[ticklabels[node_1]]
			this_col = source_df.loc[ticklabels[node_1], "c_magmom_pa"]

			color_map.append(this_col) # "red"

		if node_2 < n_instances:
			G.add_node(node_2, label=ticklabels[node_2])
			# this_col = col_names[ticklabels[node_1]]
			this_col = source_df.loc[ticklabels[node_2], "c_magmom_pa"]

			color_map.append(this_col) # "red"

		if n_ele == n_instances:
			G.add_node(parent_node, label="Root")
			color_map.append("orange")
		else:
			G.add_node(parent_node, label="") # parent_node
			color_map.append("green")

		weight_n1 = transform(x=node_dict[node_1] /int(n_instances))
		weight_n2 = transform(x=node_dict[node_2] /int(n_instances))
		
		
		weight_n1 = 1
		weight_n2 = 1

		G.add_edge(parent_node, node_1, weight= 2*weight_n1) # 1 - np.log(weight_n1), distance, 30*weight_n1
		G.add_edge(parent_node, node_2, weight= 2*weight_n2) # 1 - np.log(weight_n2), distance
	#     if weight_n1 < 0.5:
	#         G.add_edge(parent_node, node_1, weight= 30*weight_n1) # 1 - np.log(weight_n1), distance, 30*weight_n1
	#     else:
	#         G.add_edge(parent_node, node_1, weight= 20*weight_n1) # 1 - np.log(weight_n1), distance, 30*weight_n1

	#     if weight_n2 < 0.5:
	#         G.add_edge(parent_node, node_2, weight= 30*weight_n2) # 1 - np.log(weight_n2), distance
	#     else:
	#         G.add_edge(parent_node, node_2, weight= 20*weight_n2) # 1 - np.log(weight_n1), distance, 30*weight_n1

		parent_node += 1


	pos = graphviz_layout(G, prog='twopi',args='')

	fig = plt.figure(figsize=(16,16))

	weights = [G[u][v]['weight'] for u,v in G.edges()]
	# d = nx.degree(G) node_size=[v * 1000 for v in d],
	nx.draw(G, pos, node_color=color_map, width=weights,
				cmap="YlOrBr") # color_map
 
	node_labels = nx.get_node_attributes(G,'label')

	nx.draw_networkx_labels(G, pos, labels=node_labels,
						width=weights, font_size=3)

	plt.title("Circular Tree - 2D HAC visualization")
	plt.axis('equal')
	plt.savefig("{}/tree_circular.pdf".format(result_dir))



