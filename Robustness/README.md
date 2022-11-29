## Robustness Experiments 

Python scripts can be run directly in the shell:
```shell
nohup python ./MNIST/train_model.py > train.log 2>&1 &
```
```shell
nohup python ./tutorials/gen_adv.py > advgen.log 2>&1 &
nohup python ./tutorials/select_retrain.py > retrain.log 2>&1 &
```

### Datasets
- MNIST  : http://yann.lecun.com/exdb/mnist/
- FASHION  : https://github.com/zalandoresearch/fashion-mnist
- SVHN : http://ufldl.stanford.edu/housenumbers/
- CIFAR-10       : https://www.cs.toronto.edu/~kriz/cifar.html


### File Strucutures
```shell
├── baseline            # baseline testing methods
├── model               # model structures and train models  
├── trained_models      # trained example models. 
├── tutorials           # scripts for our experiments   
    ├── attack.py                # batch adversarial attacks (FGSM and PGD)
    ├── evaluate_robustness.py   # evaluate model empirical robustness   
    ├── fuzzing.py               # generate test cases
    ├── gen_adv.py               # generate adversarial examples 
    ├── metrics.py               # testing metrics  
    ├── robot.py                 # iteratively testing for enhacing model robustness to reach the requirement
    └── select_retrain.py        # select test cases and retrain the model to enhance robustness
```


