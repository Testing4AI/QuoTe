## Fairness Experiments 

Python scripts can be run directly in the shell:
```shell
nohup python ./Tabular/model/model_operation.py --dataset census > train.log 2>&1 &
```
```shell
nohup python ./FairFace/model/train_resnet50.py > train.log 2>&1 &
```

### Datasets
- Census Income  : https://www.kaggle.com/datasets/uciml/adult-census-income
- German Credit  : https://www.kaggle.com/datasets/uciml/german-credit
- Bank Marketing : https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
- FairFace       : https://github.com/joojs/fairface


### Tabular Data (Census, Credit, Bank)
```shell
├── baseline            # baseline testing methods 
├── clusters            # data clusters 
├── data                # data process
├── datasets            # prepared tabular datasets (Census, Credit, Bank)
├── model               # model structures and train models
├── trained_models      # trained example models 
├── tutorials           # scripts for our experiments
    ├── evaluate_fairness.py    # evaluate model empirical fairness
    ├── fuzzing.py              # generate unfair samples
    ├── gen_ds.py               # generate discriminatory sample pairs
    ├── metrics.py              # testing metrics  
    ├── quote.py                # iteratively testing to enhance model fairness to reach the requirement
    ├── select_retrain.py       # select test cases and retrain the model to enhance fairness
    └── utils.py                # local helper functions
└── utils               # global helper functions
```


### Image Data (FairFace)
```shell
├── cyclegan            # train CycleGAN (semantic transformer)
├── data                # data process 
├── datasets            # prepared image datasets (FairFace)
├── model               # model structures and train models
├── tutorials           # scripts for our experiments
    ├── evaluate.py             # evaluate model clean accuracy  
    ├── evaluate_fairness.py    # evaluate model empirical fairness
    ├── gen_ds.py               # generate discriminatory sample pairs (Gradient, Random, Gaussian)
    ├── metrics.py              # testing metrics  
    ├── select_retrain.py       # select test cases and retrain the model to enhance fairness
    └── transform.py            # utilize transformer to transform images across attributes 
└── utils               # global helper functions
```


