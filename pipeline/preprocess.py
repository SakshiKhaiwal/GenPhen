import numpy as np
import functools
import operator
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

class data_preprocessing:

    """
    ## This is the section on data preprocessing which includes, splitting the test and training set, removing NA and scaling the data.
    ## This is done via two strategies: 1. Random split and 2. Split by clade and considering only one clade.

    """
    def __init__(self, data):
        self.data = data  ### CSV file with first column as phenotypes and the rest with features
                            ### The rownames should be the name of the strains to return strains in test and training set.
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        ## Removing data with missing labels
        self.X = pd.DataFrame(self.data.iloc[:,1:]) ## Retrieving features
        self.y = pd.DataFrame(self.data.iloc[:,0])  ##Retrieving labels
        self.strains = pd.DataFrame(data.index,index=data.index) ##Retrieving sample name
        self.missing_values_strains = [index for index, row in self.y.iterrows() if row.isnull().any()] ##Strains for which label is not available
        self.X = self.X.drop(self.missing_values_strains, axis=0, inplace=False) ## Removing missing strains from input information
        self.y = self.y.drop(self.missing_values_strains, axis=0, inplace=False) ## Removing missing strains from  labels
        self.strains = self.strains.drop(self.missing_values_strains, axis=0, inplace=False)
        threshold_for_nan =0.75
        self.X = self.X.loc[:,self.X.isna().mean() <= threshold_for_nan]
        #self.X = self.X.fillna(0)
        #self.y = self.y.fillna(0)

    def preprocess_data_randomsplit(self, test_split_size=0.25):
        sss = ShuffleSplit(n_splits=1, test_size=test_split_size)
        sss.get_n_splits(self.X, self.y,)
        train_index, test_index = next(sss.split(self.X, self.y))
        X_train, X_test = self.X.iloc[train_index, :], self.X.iloc[test_index, :]
        y_train, y_test = self.y.iloc[train_index, :], self.y.iloc[test_index, :]
        strains_training,strains_testing = self.strains.iloc[train_index,:], self.strains.iloc[test_index, :]
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.fit_transform(X_test.values), columns=X_test.columns, index=X_test.index)
        X_train_means = X_train.mean()
        X_test_means = X_test.mean()
        X_train = X_train.fillna(X_train_means)
        X_test = X_test.fillna(X_test_means)
        y_test = np.asarray(functools.reduce(operator.iconcat, np.asarray(y_test), []))
        y_train = np.asarray(functools.reduce(operator.iconcat, np.asarray(y_train), []))
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test,'strains_training':strains_training,'strains_testing':strains_testing}

    def preprocess_data_cladesplit(self,clades):
        self.clades = clades
        clades_filter =  self.clades.drop(self.missing_values_strains, axis=0, inplace=False)
        WE_index = np.where(clades_filter == '01.Wine_European')[0]
        test_index = np.random.choice(WE_index, size=50, replace=False)
        train_index = WE_index[~np.in1d(WE_index, test_index)]
        X_train, X_test = self.X.iloc[train_index,:], self.X.iloc[test_index,:]
        y_train, y_test = self.y.iloc[train_index,:], self.y.iloc[test_index,:]
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.fit_transform(X_test.values), columns=X_test.columns, index=X_test.index)
        X_train_means = X_train.mean()
        X_test_means = X_test.mean()
        X_train = X_train.fillna(X_train_means)
        X_test = X_test.fillna(X_test_means)
        y_test = np.asarray(functools.reduce(operator.iconcat, np.asarray(y_test), []))
        y_train = np.asarray(functools.reduce(operator.iconcat, np.asarray(y_train), []))
        test_clades = self.clades.values[test_index]
        train_clades = self.clades.values[train_index]
        strains_training, strains_testing = self.strains.iloc[train_index, :], self.strains.iloc[test_index, :]
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'test_clades': test_clades, 'train_clades': train_clades,
                'strains_training':strains_training,'strains_testing':strains_testing}