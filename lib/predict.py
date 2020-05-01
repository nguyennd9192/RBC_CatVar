import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge
import time

import matplotlib.pylab as plt
from sklearn import mixture
import random

import pickle


from lib.kr_parameter_search import CV_predict, kernel_ridge_parameter_search



# # NORMALIZE
def norm_Xtrain_Xtest(X_train, X_test):
	frames = [X_train, X_test]
	train_test_df = pd.concat(frames)
	X = np.array(train_test_df)

	scaler = preprocessing.MinMaxScaler()
	X_norm = scaler.fit_transform(X)
	X_train_norm = X_norm[:len(X_train)]
	X_test_norm = X_norm[len(X_train):]

	return X_train_norm, X_test_norm

def norm_X(X):
	scaler = preprocessing.MinMaxScaler()
	X_norm = scaler.fit_transform(X)
	
	return X_norm

def param_search(model_params, X_train, y_train, X_test, y_test=None):

	X_train_norm, X_test_norm = norm_Xtrain_Xtest(X_train=X_train, X_test=X_test)

	X_train_norm = norm_X(X=X_train)

	# print (y_train)

	if model_params["model"] == "krr":
		kernel = model_params["kernel"]
		if model_params["is_param_search"]:
			if model_params["n_folds"] == "LOO":
				n_folds = len(y_train)
			else:
				n_folds = model_params["n_folds"]
			alpha, gamma, score, scores_std = kernel_ridge_parameter_search(X=X_train_norm, 
				y_obs=y_train,
				kernel=kernel, n_folds=n_folds, n_times=model_params["n_times"])
		else:
			alpha, gamma = model_params["alpha"], model_params["gamma"]

		model = KernelRidge(kernel=kernel,  alpha=alpha, gamma=gamma)
			# alpha, gamma, score, scores_std = 0.005, 4.0, 0, 0
		print ("n_folds: {0}, n_times: {1}".format(n_folds, model_params["n_times"]))
		print ("kernel: {0}, alpha: {1}, gamma: {2}, score: {3}, scores_std: {4}".
			format(kernel, alpha, gamma, score, scores_std))

	return model, X_train_norm

# # get predict model config
def get_pred_model(gps_extended, df_train, df_test, target_variable, model_params, ):
	# model_params hold hyper parameter search
	pred_model = []
	for md, x, y, pv in gps_extended:
		# pv.remove("C_R")
		# if "C_T" not in pv:
		X_train = df_train.loc[:, pv] 
		name_train = list(df_train.index)
		name_test = list(df_test.index)
		y_train = df_train.loc[:, target_variable]
		X_test = df_test.loc[:, pv]



		is_krr = True
		if model_params["is_param_search"]:
			if model_params["model"] is "krr":
				model, X_train_norm = param_search(model_params=model_params, 
					X_train=X_train, y_train=y_train,
					X_test=X_test, y_test=None,)
			else:
				model = KNeighborsRegressor(n_neighbors=5)
				y_predict = CV_predict(model=model, 
					X=X_train, y=y_train, n_folds=len(y_train)-1, n_times=1)
				score = r2_score(y_train, y_predict[0])
				n_neighbors = 5
				_, indices = model.kneighbors(X=X_test, n_neighbors=n_neighbors)


				for name_test, id_nb in zip(name_test, indices):
					name_nb = map(lambda x: name_train[x], id_nb)
					print ("Test_inst: {0}, nb: {1}".format(name_test, "|".join(name_nb)))
				print ("score: {0},".format(score))

				n_neighbors = model.kneighbors()
		else:
			model = md
			X_train_norm = norm_X(X_train)


		pred_model.append([model, X_train_norm, y_train, pv])
	return pred_model


# # for predict test set
def predict_new_instant(gps_extended, df_train, df_test, target_variable, train_idx=None):

	y_pred_plot = []
	for gp, X, y_obs, pv in gps_extended:
		if train_idx is not None:
			df_train = df_train.iloc[train_idx][:]
	  
		y_train = df_train[target_variable]
		X_train_norm, X_test_norm = norm_Xtrain_Xtest(X_train=df_train[pv], X_test=df_test[pv])

		#X_train_norm = X_all_norm[:len(X_train)]
		gp.fit(X_train_norm, y_train)
		y_new_predict = gp.predict(X_test_norm)
		y_pred_plot.append(np.array(y_new_predict))

	return y_pred_plot


def gaussian(x, mu, sig):
	return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def multi_predict_test(pred_models, target_variable, df_train, 
	df_test, train_idx, bagging=False, is_plot=True, 
	sav_dir=None, n_components_gmm=3, bagsize=0.65):

	n_insts = df_train.shape[0]
	if bagging:
		n_shuffle = 100000
		bagsize = int(bagsize * n_insts) # 65

		y_new_predict_extend, y_pred_plot = predict_by_bagging(gps_extended=pred_models,
									df_train=df_train, df_test=df_test,
									target_variable=target_variable, train_idx=train_idx,
									sav_dir=sav_dir, n_components=n_components_gmm,
									n_shuffle=n_shuffle, bagsize=bagsize)

	else:
		y_new_predict_extend = predict_new_instant(gps_extended=pred_models,
										df_train=df_train, df_test=df_test,
										target_variable=target_variable, train_idx=train_idx)
		y_pred_plot = y_new_predict_extend

	n_records = len(pred_models)
	print ("Number of total predicted val: ", n_records)

	# averagine predicted value
	out_df = df_test.copy()
	y_new_predict_average = np.zeros(len(df_test))
	for i, y_new_predict in enumerate(y_new_predict_extend):
		if i == 0:
			y_new_predict_average = y_new_predict / float(n_records)
		else:
			y_new_predict_average += y_new_predict / float(n_records)

		out_df['pred_%s_model_%d' %(target_variable, i)] = y_new_predict

	out_df['pred_%s' %target_variable] = y_new_predict_average

	if is_plot:
		# plot obs_val of test 

		test_val = { "SmFe12": 555,"YFe12": 483, "Co5La-838": 838} #"NdFe12": 508,
		# for cross validation only
		# test_val = dict({k: v for k, v in zip(df_test.index, df_test[target_variable])})
		means_init =  {"NdFe12": [[280], [360], [500]], 
						"SmFe12": [[372], [423], [542]],
						"YFe12": [[253], [350], [500]],
						"DyFe12": [[285], [380], [500]],
						"GdFe12": [[363], [400],[500]],
						"Co5La-838": [[717], [918], [1100]]}
		weighs_init = {"NdFe12": [[0.5], [0.4], [0.1]], 
						"SmFe12": [[0.4], [0.4], [0.2]],
						"YFe12": [[0.4], [0.5], [0.1]],
						"DyFe12": [[0.4], [0.5], [0.1]],
						"GdFe12": [[0.4], [0.3], [0.3]],
						"Co5La-838": [[0.3], [0.4], [0.3]]}
		n_components_init = {"NdFe12": 3, "SmFe12": 3,
						"YFe12": 3, "DyFe12": 3,
						"GdFe12": 3, "Co5La-838": 3}

		test_point = list(df_test.index)[0]
		pred_val = y_new_predict_average[0]

		if test_point in test_val.keys():
			obs_val = test_val[test_point]

		try:
			obs_val = df_test.loc[test_point, target_variable]
		except:
			pass

	   
		X_plot = np.array(y_pred_plot).reshape(-1)
		# X_plot = X_plot[X_plot > 0]


		save_col = "{0}_{1}".format(test_point, bagsize)
		save_csv_file = "{0}/pred_ensemble_xaxis_2.csv".format(sav_dir) # None
		save_csv_file = None
		save_fig_file = "{0}/{1}.pdf".format(sav_dir, test_point)
		save_gmm_file = "{0}/{1}_gmmm.sav".format(sav_dir, test_point)

		from plot import plt_hist_gmm

		plt_hist_gmm(X_plot=X_plot, save_fig_file=save_fig_file, test_point=test_point,
					is_kde=False, is_gmm=True, 
					n_components_gmm=n_components_gmm, save_gmm_file=save_gmm_file,
					save_csv_file=save_csv_file, save_col=save_col, pred_val=None, 
					obs_val=obs_val
					)
	return out_df


# # for cross-validation
def cross_val_y_predict(gps_extended, cv_k_fold, is_fit):
	for gp, X, y_obs, pv in gps_extended:
		gp.fit(X, y_obs)

		if is_fit:
			y_predict = gp.predict(X)
			y_predict = np.array([y_predict])
		else:
			if cv_k_fold == 'LOO':

				y_predict = CV_predict(model=gp, X=X, y=y_obs, n_folds=len(y_obs)-1)
				#y_predict = cross_val_predict(gp, X, y_obs, cv=len(y_obs))
			else:
				y_predict = CV_predict(model=gp, X=X, y=y_obs, n_folds=cv_k_fold)
			print (y_predict)
			#y_predict = cross_val_predict(gp, X, y_obs, cv=self.cv_k_fold)
		yield y_predict


def multi_cv_gps(gps_extended, y_obs, 
	n_times_cv, cv_k_fold, is_fit=False):
	best_scores = []
	best_MAEs = []
	y_preds = []
	n_records = len(gps_extended)
	for i_time_cv in range(n_times_cv):

		# get all y_predict for all models
		y_predicts = cross_val_y_predict(gps_extended=gps_extended, 
			cv_k_fold=cv_k_fold, is_fit=is_fit)
		sum_y_predict = np.sum(y_predicts, axis=0)

		this_y_predict = [i / n_records for i in sum_y_predict[0]]
		this_score = r2_score(y_obs, this_y_predict)
		this_MAE = mean_absolute_error(y_obs, this_y_predict)

		best_scores.append(this_score)
		best_MAEs.append(this_MAE)
		y_preds.append(this_y_predict)

	best_score = np.mean(best_scores, axis=0)
	best_score_std = np.std(best_scores, axis=0)

	best_MAE = np.mean(best_MAEs, axis=0)
	best_MAE_std = np.std(best_MAEs, axis=0)

	y_pred = np.mean(y_preds, axis=0)

	#fig_file_name += '_%.3f|%.5f_%.3f|%.5f' %(best_score, best_score_std, best_MAE, best_MAE_std)
	# fig_file_name += '_%.3f_%.3f' %(best_score, best_MAE)

	print("best_core = {0}".format(best_score))

	y_pred = np.asarray(y_pred)

	return y_pred



def get_sample(X, n, k, std_th):

	std_trial = 0

	if std_th is None:
		return random.sample(n, k)
	else:
		while std_trial < std_th:
			sample = random.sample(n, k)
			X_trial = X[sample]
			std_trial = np.std(X_trial)
			print (X_trial)
			print (std_trial)
		return sample



# # for ensemble bagging
def predict_by_bagging(gps_extended, df_train, df_test, 
						target_variable, train_idx=None, 
						sav_dir=None,  n_components=3, n_shuffle=20000, bagsize=65):
	from sklearn.ensemble import BaggingRegressor, GradientBoostingClassifier
	from kr_parameter_search import CV_predict_score
	y_return = []
	y_pred_plot = []


	models = dict({"md1": dict({"mean": 717, "sigma": 40, "context": []}),
				"md2": dict({"mean": 422, "sigma": 20, "context": []}) ,
				"md3": dict({"mean": 535, "sigma": 50, "context": []}) })

	out_pred_models = dict({"md_{}".format(k): dict({"pv":[], "krr": [], "m_inst": [] }) for k in range(n_components)})

	for gp, X, y_obs, pv in gps_extended:
		if train_idx is not None:
			df_train = df_train.iloc[train_idx][:]

		y_train = df_train[target_variable]
		X_train_norm, X_test_norm = norm_Xtrain_Xtest(X_train=df_train[pv], 
			X_test=df_test[pv])
		# pred_md.fit(X=X_train_norm, y=y_train)
		# print (pred_md.estimators_samples_)
		# print (pred_md.__dict__)

		# samples = pred_md.estimators_samples_

		
		name_train = df_train.index
		train_idx = range(len(name_train))

		y_new_predict = np.zeros(len(X_test_norm))
		n_preds = 0
		# for sample in samples:

		t_start = time.time()

		for i in range(n_shuffle):

			sample = get_sample(X=X_train_norm, n=range(len(name_train)), k=bagsize, 
				std_th=None)
			X=X_train_norm[sample]
			y=y_train[sample]

			gp.fit(X=X, y=y)

			this_idx = name_train[sample]
			# this_idx = sample

			# r2, r2_std, mae, mae_std = CV_predict_score(model=gp, X=X, y=y, n_folds=10, n_times=3)
			# print ("This r2: ", r2)
			# if r2 > 0.9:

			this_pred = gp.predict(X_test_norm)
			try:
				# save bagging models to files
				out_pred_models = collected_model_insts(pred_md=gp, m_inst=this_idx,
					y_pred=this_pred[0], pv=pv,
					gmm_model_file="{0}/{1}_gmmm.sav".format(sav_dir, df_test.index[0]), 
					out_pred_models=out_pred_models)
			except Exception as e:
				pass # False for the first run to get config of gmm, then turn it on, run again 
				
			if this_pred.all() > 0:
				n_preds += 1
				y_new_predict += this_pred
			# print (this_pred)

			# support to count_freq
			# models = gating(y_pred=this_pred, name_train=this_idx, models=models)

			y_pred_plot.append(this_pred)


		t_end = time.time()
		print ("Lose: {} seconds".format(t_end - t_start))

		y_return.append(y_new_predict / float(n_preds))


	pickle.dump(out_pred_models, open("{0}/{1}_pred_models.sav".format(sav_dir, df_test.index[0]), 'wb'))


		# y_new_predict = pred_md.predict(X_test_norm)
	# count_freq(models=models, inst_list=list(df_train.index))


	return y_return, y_pred_plot

def collected_model_insts(pred_md, m_inst, y_pred, pv, gmm_model_file, out_pred_models):
	gmm_models = pickle.load(open(gmm_model_file, "rb"))
	labels = gmm_models.predict(y_pred)
	probs = gmm_models.predict_proba(y_pred)[0]

	pred_md_idx = np.where(probs > 0.8)[0]
	try:
		assgn_md = "md_{}".format(pred_md_idx[0])
		if  len(out_pred_models[assgn_md]["krr"]) <= 400:
			out_pred_models[assgn_md]["krr"].append(pred_md)
			out_pred_models[assgn_md]["pv"].append(pv)
			out_pred_models[assgn_md]["m_inst"].append(m_inst)
			# this_coef_df = out_pred_models[assgn_md]["coef_df"]
	except:
		pass



	return out_pred_models


def gating(y_pred, name_train, models):
	# models = {"md1": {"mean": m1, "sigma": sg1, "context": []},... }
	for md, cfg in models.items():
		y_lb = cfg["mean"] - cfg["sigma"]
		y_ub = cfg["mean"] + cfg["sigma"]
		if y_pred > y_lb and y_pred < y_ub:
			cfg["context"] = np.concatenate((cfg["context"], name_train), axis=0) 
			# cfg["context"].append(name_train)
	return models


def count_freq(models, inst_list):
	from collections import Counter

	out_df = pd.DataFrame(0, index=inst_list, columns=list(models.keys()))
	for md, cfg in models.items():
		context = cfg["context"]
		with open("../result/Tc/tmp/{}.txt".format(md), "w") as f:
			# np.savetxt(f, context)
			for item in context:
				f.write("%s\n" % item)
		counter = Counter(context)

		for k, v in counter.items():
			out_df.loc[k, str(md)] = v
	out_df.to_csv("../result/Tc/tmp/count_freq.csv", "w")



if __name__ == '__main__':
	collected_model_insts(pred_md="Co5La-838", y_pred=700, 
		gmm_model_file="/home/nguyen/Bureau/work/GraphicalRegression/thay Chi/data-8a_export/result/Tc/tmp/interpolation_laplacian/Co5La-838_gmmm.sav",
		out_pred_models=[])