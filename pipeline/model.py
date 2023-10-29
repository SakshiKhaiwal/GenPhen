import json
import pandas as pd
import numpy as np
from numpy import *
from scipy import stats
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error
import json
import pickle




class Results:

  def __init__(self, results, y_train_predicted, y_test_predicted, features_importance_scores, model_type="RandHypOPt_Ridge_regression"):
      self.results = results
      self.model_type = model_type
      self.y_train_predicted = y_train_predicted
      self.y_test_predicted = y_test_predicted
      self.features_importance_scores = features_importance_scores


  def save_to_file(self, filename):
      results = { 'results': self.results.tolist(),
                 'y_train_predicted': self.y_train_predicted.tolist(),
                 'y_test_predicted': self.y_test_predicted.tolist(),
                  'Features_importance_scores' : self.features_importance_scores.tolist(),
                 }
      with open(filename, 'w+') as f:
          json.dump(results, f)






class ModelBuilder:

    def __init__(self, X_train, y_train, X_test, y_test, train_data_path=None, dataset_name=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # self.train_data_path= train_data_path
        self.model = None
        self.results = None
        self.trained = False
        self.dataset_name = dataset_name

    def train_model(self, train_method, n_iterations=1000, cross_val=3, num_jobs=-1):
        if train_method == 'BayesHypOPt_Ridge_regression':
            result = self.BayesHypOPt_Ridge_regression(n_iterations=100, cross_val=3, num_jobs=-1)
        elif train_method == 'BayesHypOPt_Elanet_regression':
            result = self.BayesHypOPt_Elanet_regression(n_iterations=100, cross_val=3, num_jobs=-1)
        elif train_method == 'BayesHypOPt_GBM_regression':
            result = self.BayesHypOPt_GBM_regression(n_iterations=100, cross_val=3, num_jobs=-1)
        elif train_method == 'BayesHypOPt_SVR_regression':
            result = self.BayesHypOPt_SVR_regression(n_iterations=100, cross_val=3, num_jobs=-1)
        elif train_method == 'BayesHypOPt_NN_regression':
            result = self.BayesHypOPt_NN_regression(n_iterations=100, cross_val=3, num_jobs=-1)
        elif train_method == 'RandHypOPt_Ridge_regression':
            result = self.RandHypOPt_Ridge_regression(n_iterations=1000, cross_val=3, num_jobs=-1)
        elif train_method == 'RandHypOPt_Elanet_regression':
            result = self.RandHypOPt_Elanet_regression(n_iterations=1000, cross_val=3, num_jobs=-1)
        elif train_method == 'RandHypOPt_GBM_regression':
            result = self.RandHypOPt_GBM_regression(n_iterations=100, cross_val=3, num_jobs=-1)
        elif train_method == 'RandHypOPt_SVR_regression':
            result = self.RandHypOPt_SVR_regression(n_iterations=100, cross_val=3, num_jobs=-1)
        elif train_method == 'RandHypOPt_NN_regression':
            result = self.RandHypOPt_NN_regression(n_iterations=100, cross_val=3, num_jobs=-1)
        else:
            print('Undefined train method')
            return

        return result

    def BayesHypOPt_Ridge_regression(self, n_iterations=10, cross_val=3, num_jobs=-1):
        from sklearn import linear_model
        import skopt
        from skopt import BayesSearchCV
        from skopt.space import Real, Categorical, Integer
        from scipy import stats

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        Ridge_distributions = dict(alpha=Real(1, 1e+4,
                                              prior='uniform'))  # Regularization parameter that varies between 0 and inf and must be a non-negative float. Alpha=0 corresponds to the OLS fit. The higher this value, teh stronger the regularization.

        Ridge_model = BayesSearchCV(linear_model.Ridge(
            # The maximum iterations are fixed to avoid extremely long computational times. ‘auto’ chooses the solver automatically based on the type of data.
            max_iter=1000),  # Maximum iterations in case the model doesn't converge before.
            Ridge_distributions, n_iter=n_iterations, verbose=2, cv=cross_val, n_jobs=num_jobs).fit(X_train,
                                                                                                     y_train).best_estimator_

        feature_importance_scores_ridge = pd.Series(list(Ridge_model.coef_), index=X_train.columns)

        y_test_predicted = Ridge_model.predict(X_test)
        y_train_predicted = Ridge_model.predict(X_train)

        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        res = {
            'Test r2score': test_r2score,

            'Train r2 score': train_r2score,

            'Test pearson value': test_pears_val,

            'Test pearson p-value': test_pears_pval,

            'Train pearson value': train_pears_val,

            'Train pearson p-value': train_pears_pval,

        }
        res = pd.Series(list(res.values()), index=(res.keys()))
        results = Results(res, y_train_predicted, y_test_predicted)#(Ridge_model, feature_importance_scores_ridge, res, y_train_predicted, y_test_predicted)

        return results

    def BayesHypOPt_Elanet_regression(self, n_iterations=1000, cross_val=3, num_jobs=-1):
        from sklearn import linear_model
        import skopt
        from skopt import BayesSearchCV
        from skopt.space import Real, Categorical, Integer

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        ### Setting the hyperparameters for Elastic Net regression
        elanet_distributions = dict(alpha=Real(1e-3, 1, prior='uniform'),  # Regularization parameter
                                    l1_ratio=Real(0, 1, prior='uniform')
                                    # Parameter controlling the weights of the contribution of the L1 and L2 regularization.
                                    )  # Maximum iterations in case the model doesn't converge before.

        Elanet_model = BayesSearchCV(linear_model.ElasticNet(max_iter=10000), elanet_distributions,
                                     n_iter=n_iterations, verbose=10, cv=cross_val, n_jobs=num_jobs).fit(X_train,
                                                                                                         y_train).best_estimator_

        feature_importance_scores_Elanet = pd.Series(list(Elanet_model.coef_), index=X_train.columns)
        # feature_importance_scores_Elanet.columns = ['Scores']
        y_test_predicted = Elanet_model.predict(X_test)
        y_train_predicted = Elanet_model.predict(X_train)

        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        res = {
            'Test r2score': test_r2score,

            'Train r2 score': train_r2score,

            'Test pearson value': test_pears_val,

            'Test pearson p-value': test_pears_pval,

            'Train pearson value': train_pears_val,

            'Train pearson p-value': train_pears_pval,

        }
        res = pd.Series(list(res.values()), index=(res.keys()))
        results = Results(res, y_train_predicted, y_test_predicted)
        #(Elanet_model, feature_importance_scores_Elanet, res, y_train_predicted, y_test_predicted)

        return results

    def BayesHypOPt_GBM_regression(self, n_iterations=100, cross_val=3, num_jobs=-1):
        from sklearn.ensemble import GradientBoostingRegressor
        import skopt
        from skopt import BayesSearchCV
        from skopt.space import Real, Categorical, Integer

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        ### Setting the hyperparameters for Gradient boosted decision trees
        GBM_distributions = dict(
            # loss=Categorical(["squared_error","huber","absolute_error"]),  #Loss function to be optimized.
            max_features=Categorical(["auto", "log2", "sqrt"]),
            # The number of features to consider when looking for the best split.
            learning_rate=Real(1e-4, 1, prior='uniform'),
            # Learning rate shrinks the contribution of each tree by learning_rate.
            subsample=Real(0, 1, prior='uniform'),
            # The fraction of samples to be used for fitting the individual base learners.
            min_samples_split=Integer(2, 100, prior='uniform'),
            # The minimum number of samples required to split an internal node.
            min_samples_leaf=Integer(2, 100, prior='uniform'),
            # The minimum number of samples required to be at a leaf node.
            n_estimators=Integer(100, 1000, prior='uniform'),  # The number of boosting stages to perform.
            criterion=Categorical(['friedman_mse', 'mse']),  # The function to measure the quality of a split.
            max_depth=Integer(2, 10, prior='uniform')  # Maximum depth of the individual regression estimators.
        )

        GBM_model = BayesSearchCV(GradientBoostingRegressor(loss="squared_error"),
                                  # To decide if early stopping will be used to terminate training when validation score is not improving.
                                  GBM_distributions, n_iter=n_iterations, verbose=10,
                                  cv=cross_val, n_jobs=num_jobs).fit(X_train, y_train).best_estimator_

        feature_importance_scores_GBM = pd.Series(list(GBM_model.feature_importances_), index=X_train.columns)
        feature_importance_scores_GBM.columns = ['Scores']
        y_test_predicted = GBM_model.predict(X_test)
        y_train_predicted = GBM_model.predict(X_train)

        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        res = {
            'Test r2score': test_r2score,

            'Train r2 score': train_r2score,

            'Test pearson value': test_pears_val,

            'Test pearson p-value': test_pears_pval,

            'Train pearson value': train_pears_val,

            'Train pearson p-value': train_pears_pval,

        }
        res = pd.Series(list(res.values()), index=(res.keys()))

        results = Results(GBM_model, feature_importance_scores_GBM, res, y_train_predicted, y_test_predicted)

        return results

    def BayesHypOPt_SVR_regression(self, n_iterations=100, cross_val=3, num_jobs=-1):
        from sklearn.svm import SVR
        import skopt
        from skopt import BayesSearchCV
        from skopt.space import Real, Categorical, Integer

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        ### Setting the hyperparameters for Support Vector regression
        SVR_distributions = dict(gamma=Categorical(["auto", "scale"]),
                                 # C is the regularization parameter inversely proportional to the regularization strength.
                                 epsilon=Real(1e-2, 1.0, prior='log-uniform'),
                                 # loguniform(1e-2,1),   #Epsilon is the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value
                                 C=Real(1e-5, 1.0, prior='log-uniform'),
                                 # Gamma defines the Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,and if ‘auto’, uses 1 / n_features.
                                 kernel=Categorical(
                                     ["linear", "rbf"]))  # Specifies the kernel type to be used in an algorithm.

        ### Setting the hyperparameters for Gradient boosted decision trees
        SVR_model = BayesSearchCV(SVR(), SVR_distributions,
                                  n_iter=n_iterations, cv=cross_val, n_jobs=num_jobs, verbose=10).fit(X_train,
                                                                                                      y_train).best_estimator_

        feature_importance_scores_SVR = pd.Series(list(SVR_model.coef_[0]), index=X_train.columns)

        # feature_importance_scores_SVR.columns = ['Scores']

        y_test_predicted = SVR_model.predict(X_test)
        y_train_predicted = SVR_model.predict(X_train)

        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        res = {
            'Test r2score': test_r2score,

            'Train r2 score': train_r2score,

            'Test pearson value': test_pears_val,

            'Test pearson p-value': test_pears_pval,

            'Train pearson value': train_pears_val,

            'Train pearson p-value': train_pears_pval,

        }
        res = pd.Series(list(res.values()), index=(res.keys()))

        results = Results(SVR_model, feature_importance_scores_SVR, res, y_train_predicted, y_test_predicted)
        return results

    def BayesHypOPt_NN_regression(self, n_iterations=1000, cross_val=3, num_jobs=-1):
        import skopt
        from skopt import BayesSearchCV
        from skopt.space import Real, Categorical, Integer
        from sklearn.neural_network import MLPRegressor
        from sklearn import metrics
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        NN_parameters = dict(hidden_layer_sizes=Integer(10, 20, prior='uniform'),
                             activation=Categorical(["logistic", "relu"]),
                             solver=Categorical(["lbfgs", "sgd"]),
                             alpha=Real(1e-5, 1, prior='log-uniform'),
                             batch_size=Integer(10, 50, prior='uniform'),
                             learning_rate=Categorical(["invscaling", "adaptive"]))

        NN_model = BayesSearchCV(MLPRegressor(max_iter=1000, verbose=True), NN_parameters,
                                 n_iter=n_iterations, cv=cross_val, n_jobs=num_jobs).fit(X_train,
                                                                                         y_train).best_estimator_

        y_test_predicted = NN_model.predict(X_test)
        y_train_predicted = NN_model.predict(X_train)

        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        res = {
            'Test r2score': test_r2score,

            'Train r2 score': train_r2score,

            'Test pearson value': test_pears_val,

            'Test pearson p-value': test_pears_pval,

            'Train pearson value': train_pears_val,

            'Train pearson p-value': train_pears_pval,

        }
        res = pd.Series(list(res.values()), index=(res.keys()))

        results = Results(res, y_train_predicted, y_test_predicted)
        return results

    def RandHypOPt_Ridge_regression(self, n_iterations=1000, cross_val=10, num_jobs=-1):
        from sklearn import linear_model
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform, truncnorm, randint, loguniform
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        Ridge_distributions = dict(alpha=loguniform(1, 1e2))  # Regularization parameter that varies between 0 and inf and must be a non-negative float. Alpha=0 corresponds to the OLS fit. The higher this value, teh stronger the regularization.

        Ridge_model = RandomizedSearchCV(linear_model.Ridge(
            # The maximum iterations are fixed to avoid extremely long computational times. ‘auto’ chooses the solver automatically based on the type of data.
            max_iter=10000),  # Maximum iterations in case the model doesn't converge before.
            Ridge_distributions, n_iter=n_iterations, verbose=10, cv=cross_val, n_jobs=num_jobs).fit(X_train,
                                                                                                     y_train).best_estimator_

        feature_importance_scores_ridge = pd.Series(list(Ridge_model.coef_), index=X_train.columns)

        y_test_predicted = (Ridge_model.predict(X_test))
        y_train_predicted = (Ridge_model.predict(X_train))

        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        MAPE = mean_absolute_percentage_error(y_train, y_train_predicted)
        MeAE = median_absolute_error(y_train, y_train_predicted)
        res = {
            'Test r2score': test_r2score,

            'Train r2 score': train_r2score,

            'Test pearson value': test_pears_val,

            'Test pearson p-value': test_pears_pval,

            'Train pearson value': train_pears_val,

            'Train pearson p-value': train_pears_pval,
            'MAPE': MAPE,
            'MeAE': MeAE,
        }
        res = pd.Series(list(res.values()), index=(res.keys()))

     #  results = Results(Ridge_model, feature_importance_scores_ridge, res, y_train_predicted, y_test_predicted)
        results = Results(res,y_train_predicted,y_test_predicted,feature_importance_scores_ridge)
        return results

    def RandHypOPt_Elanet_regression(self, n_iterations=1000, cross_val=3, num_jobs=-1):
        from sklearn import linear_model
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform, truncnorm, randint, loguniform

        # X_train= self.features
        # y_train= self.target
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        ### Setting the hyperparameters for Elastic Net regression
        elanet_distributions = dict(alpha=uniform(1e-3, 1),  # Regularization parameter
                                    l1_ratio=uniform(0.0, 1.0)
                                    # Parameter controlling the weights of the contribution of the L1 and L2 regularization.
                                    )  # Maximum iterations in case the model doesn't converge before.

        Elanet_model = RandomizedSearchCV(linear_model.ElasticNet(max_iter=10000), elanet_distributions,
                                          n_iter=n_iterations, verbose=10, cv=cross_val, n_jobs=num_jobs).fit(X_train,
                                                                                                              y_train).best_estimator_

        feature_importance_scores_Elanet = pd.Series(list(Elanet_model.coef_), index=X_train.columns)
        # feature_importance_scores_Elanet.columns = ['Scores']
        y_test_predicted = np.array(Elanet_model.predict(X_test))
        y_train_predicted = np.array(Elanet_model.predict(X_train))

        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        res = {
            'Test r2score': test_r2score,

            'Train r2 score': train_r2score,

            'Test pearson value': test_pears_val,

            'Test pearson p-value': test_pears_pval,

            'Train pearson value': train_pears_val,

            'Train pearson p-value': train_pears_pval,

        }
        res = pd.Series(list(res.values()), index=(res.keys()))

        results = Results(Elanet_model, feature_importance_scores_Elanet, res, y_train_predicted, y_test_predicted)

        return results

    def RandHypOPt_GBM_regression(self, n_iterations=100, cross_val=3, num_jobs=-1):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform, truncnorm, randint, loguniform

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        # X_train= self.features
        # y_train= self.target

        ### Setting the hyperparameters for Gradient boosted decision trees

        GBM_distributions = dict(max_features=["auto", "log2", "sqrt"], learning_rate=uniform(1e-3, 1),
                                 subsample=uniform(0, 1),
                                 min_samples_split=randint(2, 100), min_samples_leaf=randint(2, 100),
                                 n_estimators=randint(4, 100), criterion=['friedman_mse', 'absolute_error'],
                                 max_depth=randint(2, 10))

        GBM_model = RandomizedSearchCV(GradientBoostingRegressor(loss="squared_error", n_iter_no_change=5),
                                       # To decide if early stopping will be used to terminate training when validation score is not improving.
                                       GBM_distributions, n_iter=n_iterations, verbose=10,
                                       cv=cross_val, n_jobs=num_jobs).fit(X_train, y_train).best_estimator_

        # feature_importance_scores_GBM= transpose(pd.DataFrame(data=GBM_model.feature_importances_,
        #                                              columns= X_train.columns))
        # feature_importance_scores_GBM.columns = ['Scores']
        feature_importance_scores_GBM = pd.Series(list(GBM_model.feature_importances_), index=X_train.columns)
        y_test_predicted = GBM_model.predict(X_test)
        y_train_predicted = GBM_model.predict(X_train)

        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        res = {
            'Test r2score': test_r2score,

            'Train r2 score': train_r2score,

            'Test pearson value': test_pears_val,

            'Test pearson p-value': test_pears_pval,

            'Train pearson value': train_pears_val,

            'Train pearson p-value': train_pears_pval,

        }
        res = pd.Series(list(res.values()), index=(res.keys()))

        results = Results(GBM_model, feature_importance_scores_GBM, res, y_train_predicted, y_test_predicted)

        return results

    def RandHypOPt_SVR_regression(self, n_iterations=100, cross_val=3, num_jobs=-1):
        from sklearn.svm import SVR
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform, truncnorm, randint, loguniform

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        ### Setting the hyperparameters for Support Vector regression
        SVR_distributions = dict(gamma=["auto", "scale"],
                                 # C is the regularization parameter inversely proportional to the regularization strength.
                                 epsilon=loguniform(1e-2, 1),
                                 # Epsilon is the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value
                                 C=loguniform(1e-5, 1),
                                 # Gamma defines the Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,and if ‘auto’, uses 1 / n_features.
                                 kernel=["linear", "rbf"])  # Specifies the kernel type to be used in an algorithm.

        ### Setting the hyperparameters for Gradient boosted decision trees
        SVR_model = RandomizedSearchCV(SVR(), SVR_distributions,
                                       n_iter=n_iterations, cv=cross_val, n_jobs=num_jobs, verbose=10).fit(X_train,
                                                                                                           y_train).best_estimator_

        # feature_importance_scores_SVR = transpose(pd.DataFrame(data=SVR_model.coef_,
        #                                       columns= X_train.columns))
        # feature_importance_scores_SVR.columns = ['Scores']
        feature_importance_scores_SVR = pd.Series(list(SVR_model.coef_[0]), index=X_train.columns)
        y_test_predicted = SVR_model.predict(X_test)
        y_train_predicted = SVR_model.predict(X_train)

        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]
        res = {
            'Test r2score': test_r2score,

            'Train r2 score': train_r2score,

            'Test pearson value': test_pears_val,

            'Test pearson p-value': test_pears_pval,

            'Train pearson value': train_pears_val,

            'Train pearson p-value': train_pears_pval,

        }
        res = pd.Series(list(res.values()), index=(res.keys()))

        results = Results(SVR_model, feature_importance_scores_SVR, res, y_train_predicted, y_test_predicted)
        return results

    def RandHypOPt_NN_regression(self, n_iterations=1000, cross_val=3, num_jobs=-1):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform, truncnorm, randint, loguniform
        from sklearn.neural_network import MLPRegressor
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        NN_parameters = dict(hidden_layer_sizes=randint(3, 10),
                            # activation=["relu"],
                            # solver=["adam"],
                             alpha=loguniform(0.1, 0.9),
                             batch_size=randint(10, 50))
                             #learning_rate=["invscaling"])
        NN_model = RandomizedSearchCV(MLPRegressor(max_iter=200,activation="relu", solver="adam", verbose=True, learning_rate="invscaling"), NN_parameters,
                                      n_iter=n_iterations, cv=cross_val, n_jobs=num_jobs).fit(X_train,
                                                                                              y_train).best_estimator_

        pickle.dump(NN_model, open(self.dataset_name, 'wb'))

        y_test_predicted = NN_model.predict(X_test)
        y_train_predicted = NN_model.predict(X_train)

        test_r2score = metrics.r2_score(y_test, y_test_predicted)
        train_r2score = metrics.r2_score(y_train, y_train_predicted)
        test_pears_val = stats.pearsonr(y_test, y_test_predicted)[0]
        test_pears_pval = stats.pearsonr(y_test, y_test_predicted)[1]
        train_pears_val = stats.pearsonr(y_train, y_train_predicted)[0]
        train_pears_pval = stats.pearsonr(y_train, y_train_predicted)[1]

        res = {
            'Test r2score': test_r2score,

            'Train r2 score': train_r2score,

            'Test pearson value': test_pears_val,

            'Test pearson p-value': test_pears_pval,

            'Train pearson value': train_pears_val,

            'Train pearson p-value': train_pears_pval,



        }
        res = pd.Series(list(res.values()), index=(res.keys()))

        results = Results(NN_model, res, y_train_predicted, y_test_predicted)

        return results

