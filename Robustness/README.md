## Robustness Experiments 

Python scripts can be run directly in the shell:
```shell
nohup python ./MNIST/train_model.py > training.log 2>&1 &
```
```shell
nohup python ./tutorials/gen_adv.py > advgen.log 2>&1 &
nohup python ./tutorials/select_retrain.py > retrain.log 2>&1 &
```

### Datasets
- MNIST  : https://www.kaggle.com/datasets/uciml/adult-census-income
- FASHION  : https://www.kaggle.com/datasets/uciml/german-credit
- SVHN : https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
- CIFAR-10       : https://github.com/joojs/fairface


### File Strucutures
```shell
├── CIFAR-10
├── FASHION
├── MNIST
├── SVHN
├── baseline        # baseline testing methods
├── model
├── trained_models
├── tutorials
    ├── attack.py   # implementation of traditional adversarial attacks, including FGSM and PGD. 
    ├── robot.py    # iteratively testing for enhacing model robustness to reach the requirement.  
    └── gen_adv.py  # generate adversarial examples automatically. 
```




