import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



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
				model = train_weights(X_train,y_train, Lambda=param[l])
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




########################################################################################################

def train_weights(X_train,y_train, method='logistic', reg = 'l1', Lambda = 0.1,niter = 1000):
	#reg is the type of regularization
	#lam is the regularization coefficient

	if method is 'logistic':
		lr = LogisticRegression(solver='saga',multi_class='multinomial',\
			penalty = reg, C = 1.0/Lambda, max_iter = niter, tol = 0.001)
		clf = lr.fit(X_train,y_train)
		return  clf

	# if method is 'svm':
	# 	svm = 

#########################################################################################################


def predict(X,y,model):
	pred_labels = model.predict(X)
	predict_prob = model.predict_proba(X)
	score = model.score(X, y)
	conf_mat = confusion_matrix(y, pred_labels)
	return score,conf_mat


########################################################################################################

def pipeline(trainpath, testpath, method):

	#get data
	X_train,y_train = get_data_and_labels(trainpath)

	# cross-validate to select regularization parameter
	Lambda, cv_scores = cross_valid(X_train,y_train, method=method, plot_weights = False)
	print('Cross-validation set accuracies', cv_scores)	
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

	
	#test on data
	X_test,y_test = get_data_and_labels(testpath)
	print(len(np.where(y_test == 2)))
	test_score, conf_mat = predict(X_test,y_test,train_model)
	print('Test set accuracy', test_score)
	print('Confusion matrix ')
	print(conf_mat)

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



def main():
	#Logistic regression
	# pipeline(trainpath, testpath, method='logistic')

	Xt, yt = get_data_and_labels(trainpath)
	scores = plot_training_error(Xt,yt,'logistic',150)


	#TO-DO SVM with gaussian kernel


	
	

	


if __name__ == '__main__':
    main()


