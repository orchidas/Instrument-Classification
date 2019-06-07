import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d




trainpath = '../data/instruments/train/data_with_labels.csv'
testpath = '../data/instruments/test/data_with_labels.csv'
nClasses = 4
instruments = ['bassoon','violin','saxphone','clarinet']


def get_data_and_labels(filepath):
	data = np.genfromtxt(filepath,delimiter = ',')
	#X is the feature matrix, y are the labels
	X = data[:,:-1]
	y = data[:,-1]
	y = y.astype('int')
	return (X,y)


def cross_valid(X, y, K=3, method='logistic', plot_weights=False, hyperparam = 'Lambda', param = np.logspace(-3,3,7)):
	#hyperparam is the parameter to be selected by cross validation
	#parameter is the value of hyperparam


	########################################################################################
	#cross validation for logistic regression

	if method is 'logistic':
		np.random.seed(39103)
		L = len(param)
		(n,d) = np.shape(X)
		weights = np.zeros([nClasses,d,L]) 
		kf = KFold(n_splits = K, shuffle = True)
		valid_scores = np.zeros([K,L])

		#try different regularization params
		for l in range(L):
			#k fold cross-validation
			k = 0
			for train_index, valid_index in kf.split(X):
				# print("TRAIN:", train_index, "VALID:", valid_index)
				X_train, X_valid = X[train_index], X[valid_index]
				y_train, y_valid = y[train_index], y[valid_index]
				#if regularization parameter is to be chosen
				if hyperparam is 'Lambda':
					model = train_weights(X_train,y_train, method, Lambda=param[l])
				weights[:,:,l] += model.coef_
				pred = model.predict(X_valid)
				valid_scores[k,l] = model.score(X_valid,y_valid)
				k += 1

			#average weights over the K folds
		  	weights[:,:,l] /= K

		# best_param = np.argmax(np.mean(valid_scores, axis = 0))
		# print('Best ' + hyperparam, best_param)
		# print('Cross validation scores with best '+ hyperparam, valid_scores[:,best_param])
		# print('Variance in weights across folds', np.var(weights[:,:,best_param], axis = 0))


		if plot_weights:
			for n in range(nClasses):
				# plt.figure()
				plt.semilogx(param,weights[n,:,:].T)
				plt.xlabel('Regularization coefficient')
				plt.ylabel('Weights')
				plt.grid(True)
				plt.savefig('../plots/l1_'+str(n))

		return param,valid_scores
		
		#############################################################################3
		#cross validation for SVM 		

	elif method is 'svm':
		all_hyperparams = dict()
		HP = len(hyperparam)
		for i in range(HP):
			all_hyperparams.update({hyperparam[i]:param[i]})
		
		svc = svm.SVC(kernel='rbf')
		svm_cv = GridSearchCV(svc, all_hyperparams, cv = K)
		model = svm_cv.fit(X,y)
		valid_scores = svm_cv.cv_results_['mean_test_score']
		best_score = svm_cv.best_score_
		best_params = svm_cv.best_params_

		return valid_scores,best_score,best_params

	

########################################################################################################

def train_weights(X_train,y_train, method='logistic', reg = 'l1', Lambda = 0.1, niter = 1000, C = 1.0 , gamma= 1.0):

	if method is 'logistic':
		lr = LogisticRegression(solver='saga',multi_class='multinomial',\
			penalty = reg, C = 1.0/Lambda, max_iter = niter, tol = 0.001)
		clf = lr.fit(X_train,y_train)
		return  clf

	elif method is 'svm':
		svc = svm.SVC(C = C, kernel='rbf', gamma=gamma)
		clf = svc.fit(X_train,y_train)
		return clf

#########################################################################################################


def predict(X,y,model):
	pred_labels = model.predict(X)
	# predict_prob = model.predict_proba(X)
	score = model.score(X, y)
	conf_mat = confusion_matrix(y, pred_labels)
	return pred_labels,score,conf_mat


########################################################################################################

def pipeline(trainpath, testpath, method):

	#get data
	X_train,y_train = get_data_and_labels(trainpath)

	# cross-validate to select regularization parameter for logistic regression
	if method is 'logistic':
		Lambda, cv_scores = cross_valid(X_train,y_train, method=method, plot_weights = False)
		print('Cross-validation set accuracies', cv_scores,[''])	
		Lambda_best = Lambda[1] 
		# print('Cross validation scores with lambda = 1', cv_scores[:,3])

		# train model on whole training set with all features
		train_model = train_weights(X_train,y_train, method=method, Lambda=Lambda_best)
		# print(train_model.coef_)

		# reduce model order 
		sfm = SelectFromModel(train_model, prefit=True)
		new_feature_inds = np.where(sfm.get_support(indices = False) == True)
		print('Remaining feature indices after L1 ', new_feature_inds)
		X_train_new = sfm.transform(X_train)


	elif method is 'svm':
		C_grid = np.logspace(-2,2,5)
		gamma_grid = np.logspace(-3,2,6)
		scores,best_score,best_params = cross_valid(X_train, y_train, method=method, hyperparam=['C','gamma'],\
			param=[C_grid, gamma_grid])
		print(best_score, best_params)

		# train model on whole training set with all features
		train_model = train_weights(X_train,y_train, method=method,C=best_params['C'],gamma=best_params['gamma'])
		# print(train_model.support_vectors_)

	
	#test on data
	X_test,y_test = get_data_and_labels(testpath)
	y_pred, test_score, conf_mat = predict(X_test,y_test,train_model)
	print('Test set accuracy', test_score)
	print('Confusion matrix ')
	print(conf_mat)
	plot_predicted_classes(X_test,y_test,y_pred)


#############################################################################################


def plot_training_error(Xt,yt,method,niter):
	scores = []
	iter = np.arange(1,niter,2)
	for it in iter:
		scores.append(train_weights(Xt,yt,method = method, niter=it).score(Xt,yt))


	plt.plot(iter,scores)
	plt.xlabel('# iterations')
	plt.ylim([0.94, 1.01])
	plt.grid(True)
	plt.ylabel('Classification accuracy')
	plt.show()
	return scores

################################################################################################

def plot_predicted_classes(Xtest,ytest,ypred):
	
	#use SVD to reduce data to two dimensions and look at predicted labels
	svd=TruncatedSVD(n_components=2).fit_transform(Xtest)
	colors=['r','g','b','k']

	plt.figure()	
	plt.xlabel('x_1')
	plt.ylabel('x_2')
	plt.grid(True)

	# ax = plt.axes(projection='3d')
	# ax.set_xlabel('x_1')
	# ax.set_ylabel('x_2')
	# ax.set_zlabel('x_3')
	
	for c in range(nClasses):
		plt.plot(svd[ytest==c][:,0],svd[ytest==c][:,1],colors[c]+'+',markersize=8)
		# ax.plot3D(svd[ytest==c][:,0],svd[ytest==c][:,1],svd[ytest==c][:,2],colors[c]+'+',markersize=8)

	errX,errY=svd[ypred!=ytest],ytest[ypred!=ytest]
	for c in range(nClasses):
		plt.plot(errX[errY==c][:,0],errX[errY==c][:,1],colors[c]+'o')

	plt.legend(instruments, loc = 'lower left')
	plt.show()

 #############################################################################################  


def main():

	#Logistic regression
	pipeline(trainpath, testpath, method='logistic')
	Xt, yt = get_data_and_labels(trainpath)
	scores = plot_training_error(Xt,yt,'logistic',150)

	#SVM
	pipeline(trainpath, testpath, method = 'svm')

	


if __name__ == '__main__':
    main()


