## Fairness Experiments (Tabular Data)

Python scripts can be run directly in the shell:  
```shell
nohup python ./model/model_operation.py --dataset census > train.log 2>&1 &
```
```shell
nohup python ./tutorials/fuzzing.py --dataset census --sens_param 8 > fuzz.log 2>&1 &
```

### Datasets
- Census Income  : https://www.kaggle.com/datasets/uciml/adult-census-income
- German Credit  : https://www.kaggle.com/datasets/uciml/german-credit
- Bank Marketing : https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing


### File Structures 
```shell
├── baseline            # baseline testing methods 
├── clusters            # clusters for testing
├── data                # data process
├── datasets            # prepared tabular datasets 
├── model               # model structures and train models
├── trained_models      # trained example models 
├── tutorials           # scripts for our experiments
    ├── evaluate_fairness.py   # evaluate model empirical fairness
    ├── fuzzing.py             # generate unfair samples
    ├── gen_ds.py              # generate discriminatory sample pairs
    ├── metrics.py             # testing metrics  
    ├── quote.py               # iteratively testing for enhacing model fairness to reach the requirement  
    ├── select_retrain.py      # select test cases and retrain the model to enhance fairness
    └── utils.py               # local helper functions
└── utils               # global helper functions
```

