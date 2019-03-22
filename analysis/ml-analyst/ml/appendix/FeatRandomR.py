import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import sys
    from feat import Feat
    from sklearn.model_selection import GridSearchCV
    import pdb
    from evaluate_model import evaluate_model
    from read_file import read_file

    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_seed = int(sys.argv[3])

    print('random_seed:', random_seed)
    # Read the data set into meory
    X,y,names = read_file(dataset, classification=False )
    # parameter variation
    hyper_params = [{
                    'hillclimb': [True],
                    'iters': [1, 10, 100],
                    }, 
                    {
                    'backprop': [True],
                    'iters': [1, 10, 100],
                    }
                    ]
    # create the classifier
    clf = Feat(pop_size=10000,
               gens=0,
               ml = "LinearRidgeRegression",
               sel='random',
               surv='random',
               max_depth=10,
               max_dim=min([X.shape[1]*2,50]),
               random_state=random_seed,
               n_threads=1,
               verbosity=1,
               fb=0.0,
               logfile=save_file.split('.csv')[0]+'_'+str(random_seed)+'.csv')
  #functions 
    # 10-fold CV score for the pipeline
    clf_name = 'FeatRandom'
# evaluate the model
    evaluate_model(dataset, save_file, random_seed, clf, clf_name, hyper_params, classification=False )

