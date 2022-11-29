1. Generate Adversarial Examples (FGSM, PGD, C&W supported)

2. Test Case Selection for Model Retraining (DeepGini, KM-ST, BE-ST)
 
3. Fol-guided Fuzzing for Test Case Generation

4. running script robot.py 

```shell
├── CIFAR-10
├── FASHION
├── MNIST
├── SVHN
├── baseline
├── model
├── trained_models
├── tutorials
    ├── attack.py   # implementation of traditional adversarial attacks, including FGSM and PGD. 
    ├── robot.py    # iteratively testing for enhacing model robustness to reach the requirement.  
    └── gen_adv.py  # generate adversarial examples automatically. 
```

