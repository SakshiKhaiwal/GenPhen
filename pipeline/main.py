import sys, types, os
import pandas as pd
import json
import numpy as np
import time
from model import ModelBuilder
from parameters import get_parameters
from preprocess import data_preprocessing
from feature_selection import FeatureSelection


if __name__ == '__main__':
    params = get_parameters()
    data = pd.read_csv(params.data_path,index_col=0)
    data_preprocessor = data_preprocessing(data)
    if params.data_splitting_criteria == 'preprocess_data_randomsplit':
        preprocessed_data = data_preprocessor.preprocess_data_randomsplit(test_split_size=0.25)
    else:
        clades = pd.read_csv(params.clades_data_path,index_col=0)
        preprocessed_data = data_preprocessor.preprocess_data_cladesplit(clades)

    if params.do_feature_selection:
        features_selector = FeatureSelection(preprocessed_data['X_train'],
                                             preprocessed_data['X_test'],
                                             preprocessed_data['y_train'])
        features = features_selector.select_features(method=params.feature_selection_strategy)
    else:
        features = preprocessed_data

    model = ModelBuilder(X_train=features['X_train'], X_test=features['X_test'],
                         y_train=preprocessed_data['y_train'], y_test=preprocessed_data['y_test'])
    start_time = time.time()
    r = model.train_model(train_method=params.model_type)
    end_time = time.time()
    training_time = end_time - start_time
    Test_r2score= r.results['Test r2score']
    Train_r2score = r.results['Train r2 score']
    Test_pears_val = r.results['Test pearson value']
    Train_pears_val = r.results['Train pearson value']
    testing_strains = preprocessed_data['strains_testing'].iloc[:,0].tolist()
    y_test_predicted = r.y_test_predicted.tolist()
    training_strains = preprocessed_data['strains_training'].iloc[:,0].tolist()
    y_train_predicted = r.y_train_predicted.tolist()
    feature_importance_scores = r.feature_importance_scores.to_dict()


    x_name_save = 'CNVs_Lasso_gridFS'
    target_name = 'CLS_CRday21'


    with open(f'{params.data_path_out}{x_name_save}_{target_name}_{params.model_type}_with_traintime.json', 'w+') as f:
        d =    {'Test r2 score' : Test_r2score,
                'Train r2 score' : Train_r2score,
                'Test pears value': Test_pears_val,
                'Train pears value': Train_pears_val,
                'Training time' : training_time
               }

        json.dump(d,f)

    with open(f'{params.data_path_out}{x_name_save}_{target_name}_{params.model_type}_with_traintime_additional_information.json','w+') as f:
        d =     {'y_train_predicted': y_train_predicted,
                 'y_test_predicted': y_test_predicted,
                 'training_strains': training_strains,
                 'testing_strains': testing_strains,
                 'Features importance scores': feature_importance_scores
                 }
        json.dump(d, f)

