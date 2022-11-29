1. Generate Discriminatory Sample Pairs 

2. Test Case Selection

3. Test Case Generation

4. running script quote.py  



### FairFace
```shell
├── cyclegan            # code for train the CycleGAN (semantic transformer)
├── data                # data process for CycleGAN
├── datasets            # prepared dataset (FairFace)
├── model               # model structures and train models
├── tutorials           # scripts for our experiments
    ├── evaluate.py            # evaluate model clean accuracy  
    ├── evaluate_fairness.py   # evaluate model empirical fairness
    ├── gen_ds.py              # generate discriminatory sample pairs (Gradient, Random, Gaussian)
    ├── metrics.py             # testing metrics  
    ├── select_retrain.py      # select test cases and retrain the model to enhance fairness
    └── transform.py           # utilize GAN to transform images across attributes (Race)   
└── utils               # helper functions
```
