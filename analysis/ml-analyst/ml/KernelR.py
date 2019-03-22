
import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import sys
    import numpy as np

    from sklearn.kernel_ridge import KernelRidge 
    from evaluate_model import evaluate_model

    # inputs
    dataset = sys.argv[1]
    save_file = sys.argv[2]
    random_seed = int(sys.argv[3])

    print('random_seed:', random_seed)

    np.random.seed(random_seed)

    # parameters for method
    hyper_params = {
            'kernel': ['rbf'],
            "alpha": [1e0, 0.1, 1e-2, 1e-3],
            "gamma": np.logspace(-2, 2, 5)
            }
     
    clf = KernelRidge()
    clf_name = 'KernelRidge' 
    #evaluate
    evaluate_model(dataset, save_file, random_seed, clf, clf_name, hyper_params,
                   classification=False)
