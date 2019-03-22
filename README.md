Code to reproduce the experiments in the paper 
"Learning concise representations for regression by evolving networks of trees"

# Experiments

Experiments can be run using `analysis/ml-analyst/submit_jobs.py`. See the command line options (`python submit_jobs.py -h`) for help.

As example, this command would launch the main experiment from the paper:

    python submit_jobs.py --r -ml Feat,FeatCN,FeatCorr,RandomForest,MLPmod,Kernel,Linear,XGBoostLong -n_trials 5 ../penn-ml-benchmark/datasets/regression/

# Notebooks

`analysis/` contains these notebooks:
 - `results_iclr.ipynb` produces the main results figures.
 - `results_appendix.ipynb` produces the results comparing different stochastic optimization approaches.
 - `stats.ipynb` depends on `results_iclr.ipynb` and produces the statistical tests.
 - `archive.ipynb` contains code to reproduce the illustrative example.

# Dependencies
 - [FEAT](http://github.com/lacava/feat) 
 - [scikit-learn](http://scikit-learn.org/stable/index.html)
 - [xgboost](https://github.com/dmlc/xgboost)
 - datasets come from [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks)
