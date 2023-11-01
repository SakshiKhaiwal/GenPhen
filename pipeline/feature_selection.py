import pandas
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.feature_selection import SelectFromModel
from skopt import BayesSearchCV
from skopt.space import Real
from hi_lasso.hi_lasso import HiLasso


class FeatureSelection:
    """
    Class for selection of features from high dimensional data.
    """
    def __init__(self , X_train: pandas.DataFrame , X_test: pandas.DataFrame, y_train: np.ndarray):
        """
        Instantiate Feature Selection.
        Args:
            X_train: training input data.
            X_test: test input data.
            y_train: training output data.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test =  X_test
        self.selector = None

    def select_features(self, method='lasso_selection_grid'):
        """

        Parameters
        ----------
        method

        Returns
        -------

        """
        if method == 'lasso_selection_grid':
            features = self.lasso_selection_grid()
        elif method == 'high_lasso':
            features = self.high_lasso()
        elif method == 'lasso_selection_random':
            features = self.lasso_selection_random()
        elif method == 'lasso_selection_bayes':
            features = self.lasso_selection_bayes()
        else:
            print('unrecognized feature selection method')
            return

        return features

    def lasso_selection_grid(self ,nfolds=5 ,njobs=-1):


        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train


        lasso = Lasso(random_state=0, max_iter=10000)
        alphas = (0.001, 0.01, 0.1 ,0.5)
        tuned_parameters = [{'alpha': alphas}]
        Tuned_lasso_alpha= GridSearchCV(lasso, tuned_parameters, cv=nfolds ,n_jobs=njobs, refit=False).fit(X_train,y_train).best_params_
        selector = SelectFromModel(estimator=Lasso(random_state=1000, max_iter=50000 ,alpha=Tuned_lasso_alpha['alpha'])).fit(X_train, y_train)
        LASSO_X_train = selector.transform(X_train)
        LASSO_X_test = selector.transform(X_test)
        LASSO_X_train = pd.DataFrame(data = LASSO_X_train,
                                     columns = X_train.columns[np.where(selector.get_support() == True)[0]])
        LASSO_X_test =  pd.DataFrame(data = LASSO_X_test,
                                     columns = X_test.columns[np.where(selector.get_support() == True)[0]])

        self.LASSO_X_train = LASSO_X_train
        self.LASSO_X_test = LASSO_X_test

        return {'X_train': LASSO_X_train ,'X_test': LASSO_X_test }

    def lasso_selection_random(self ,n_iterations=500,nfolds=5 ,njobs=-1):


        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        # y_test = self.y_test

        lasso = Lasso(random_state=0, max_iter=10000)
        tuned_parameters = dict(alpha=loguniform(1e-4, 1))
        Tuned_lasso_alpha= RandomizedSearchCV(lasso,  tuned_parameters, n_iter=n_iterations,cv=nfolds ,n_jobs=njobs, refit=False).fit(X_train,y_train).best_params_
        selector = SelectFromModel(estimator=Lasso(random_state=1000, max_iter=50000 ,alpha=Tuned_lasso_alpha['alpha'])).fit(X_train, y_train)
        LASSO_X_train = selector.transform(X_train)
        LASSO_X_test = selector.transform(X_test)
        LASSO_X_train = pd.DataFrame(data = LASSO_X_train,
                                     columns = X_train.columns[np.where(selector.get_support() == True)[0]])
        LASSO_X_test =  pd.DataFrame(data = LASSO_X_test,
                                     columns = X_test.columns[np.where(selector.get_support() == True)[0]])

        self.LASSO_X_train = LASSO_X_train
        self.LASSO_X_test = LASSO_X_test

        return {'X_train': LASSO_X_train ,'X_test': LASSO_X_test }

    def lasso_selection_bayes(self, n_iterations=200,nfolds=5 ,njobs=-1):

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        # y_test = self.y_test

        lasso = Lasso(random_state=0, max_iter=10000)
        tuned_parameters = dict(alpha=Real(1e-4, 1,prior='log-uniform'))
        Tuned_lasso_alpha= BayesSearchCV(lasso,  tuned_parameters, n_iter=n_iterations,cv=nfolds ,n_jobs=njobs, refit=False).fit(X_train,y_train).best_params_
        selector = SelectFromModel(estimator=Lasso(random_state=1000, max_iter=50000 ,alpha=Tuned_lasso_alpha['alpha'])).fit(X_train, y_train)
        LASSO_X_train = selector.transform(X_train)
        LASSO_X_test = selector.transform(X_test)
        LASSO_X_train = pd.DataFrame(data = LASSO_X_train,
                                     columns = X_train.columns[np.where(selector.get_support() == True)[0]])
        LASSO_X_test =  pd.DataFrame(data = LASSO_X_test,
                                     columns = X_test.columns[np.where(selector.get_support() == True)[0]])

        self.LASSO_X_train = LASSO_X_train
        self.LASSO_X_test = LASSO_X_test

        return {'X_train': LASSO_X_train ,'X_test': LASSO_X_test }

    def high_lasso(self ,L=30 ,alpha=0.05 ,njobs= 50):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        hilasso = HiLasso(q1=175, q2=175, L=L, alpha=alpha, logistic=False, random_state=None, parallel=True, n_jobs=njobs)
        hilasso.fit(X_train ,y_train, sample_weight=None)
        hilasso_coef = hilasso.coef_

        X_train_HiLasso = X_train.iloc[: ,np.where(hilasso_coef)[0]]
        # pd.DataFrame(data = X_train[:,np.where(hilasso_coef != 0)[0]],
        #            columns = X_train.columns[np.where(hilasso_coef != 0)[0]])
        X_test_HiLasso = X_test.iloc[: ,np.where(hilasso_coef)[0]]
        # pd.DataFrame(data = X_test[:,np.where(hilasso_coef != 0)[0]],
        #            columns = X_test.columns[np.where(hilasso_coef != 0)[0]])
        self.X_train_HiLasso = X_train_HiLasso
        self.X_test_HiLasso  = X_test_HiLasso
        return {'X_train': X_train_HiLasso ,'X_test': X_test_HiLasso}
        # return (hilasso_coef)

