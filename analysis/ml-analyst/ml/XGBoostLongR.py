import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import sys
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    import pdb
    from evaluate_model import evaluate_model
    import numpy as np

    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_seed = int(sys.argv[3])

# Read the data set into meory
# parameter variation
    hyper_params = {
        'n_estimators': (10, 100, 200, 500, 1000),
        'max_depth': (3,4,5,6,7),
        'gamma': np.logspace(-3,3,7),
        'learning_rate':np.linspace(0,1,11)
    }
# create the classifier
    clf = XGBRegressor()
    clf_name = 'XGB_long'
# evaluate the model
    evaluate_model(dataset, save_file, random_seed, clf, clf_name, hyper_params,
                   classification=False)
