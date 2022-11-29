## Fairness Experiments (Image Data)

Official implementation of CycleGAN is available at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
Python scripts can be run directly using the following command: 
```shell
python ./model/train_resnet50.py  
```

### File Structures 
```shell
├── cyclegan            # code for train the CycleGAN (semantic transformer)
├── data                # data process for CycleGAN
├── datasets            # FairFace dataset
├── model               # model structures and model training
├── tutorials           # scripts for our experiments
    ├── evaluate.py            # evaluate model clean accuracy  
    ├── evaluate_fairness.py   # evaluate model empirical fairness
    ├── gen_ds.py              # generate discriminatory sample pairs (Gradient, Random, Gaussian)
    ├── metrics.py             # testing metrics  
    ├── select_retrain.py      # select test cases and retrain the model to enhance fairness
    └── transform.py           # utilize GAN to transform images across attributes (Race)   
└── utils               # helper functions
```

