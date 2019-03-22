import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import sys
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    import pdb
    from evaluate_model import evaluate_model

    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_seed = int(sys.argv[3])

# Read the data set into meory
# parameter variation
    hyper_params = {
        'n_estimators': (10, 100, 1000),
        'min_weight_fraction_leaf': (0.0, 0.25, 0.5),
        #'max_features': ('sqrt','log2',None),
        #'criterion': ('gini','entropy')
    }

# create the classifier
    clf = RandomForestRegressor()
    clf_name = 'RF'
# evaluate the model
    evaluate_model(dataset, save_file, random_seed, clf, clf_name, hyper_params,
                   classification=False)
