# QuoTe: Quality-oriented Testing for DL Systems
 
This repository contains code for the paper [QuoTe: Quality-oriented Testing for Deep Learning Systems] 

<!--and [RobOT: Robustness-Oriented Testing for Deep Learning Systems](https://doi.org/10.1109/ICSE43902.2021.00038) (ICSE'21). -->


## Prerequisite (Py3 & TF2) 
The code is run successfully using **Python 3.6.10** and **Tensorflow 2.2.0**. 

We recommend using **conda** to install the tensorflow-gpu environment:

```shell
$ conda create -n tf2-gpu tensorflow-gpu==2.2.0
$ conda activate tf2-gpu
```

To run code in the jupyter notebook, you should add the kernel manually: 

```shell
$ pip install ipykernel
$ python -m ipykernel install --name tf2-gpu
```

To run jupyter notebook on a sever:  
```shell
nohup jupyter notebook --ip 0.0.0.0 & 
```

## Work Flow 
![Snipaste_2022-11-18_17-38-26](https://user-images.githubusercontent.com/95740042/202670553-002de81e-20f5-4a75-a9f6-1b56ce94d6e6.png)


## Files
- #### **`Robustness`: Experiments for Robustness Property**
   - **Datasets**: MNIST (image) / FASHION (image) / CIFAR10 (image) / SVHN (image) 
- #### **`Fairness`: Experiments for Fairness Property**
   - **Datasets**: Census (tabular) / Credit (tabular) / Bank (tabular) / FairFace (image)    
- #### `utils`: Utils for figure plotting

<!-- **Reference:**  -->


## To Run
See the `README.md` in each directory for a quick start. 

Example models are provided in the corresponding directory. 


## Publications 
```
@article{quote2022,
  author    = {Jialuo Chen and
               Jingyi Wang and
               Xingjun Ma and
               Youcheng Sun and
               Jun Sun and
               Peixin Zhang and
               Peng Cheng},
  title     = {QuoTe: Quality-oriented Testing for Deep Learning Systems},
  booktitle = {},
  year      = {2022},
}
```
```
@inproceedings{robot2011,
  author    = {Jingyi Wang and
               Jialuo Chen and
               Youcheng Sun and
               Xingjun Ma and
               Dongxia Wang and
               Jun Sun and
               Peng Cheng},
  title     = {RobOT: Robustness-Oriented Testing for Deep Learning Systems},
  booktitle = {43rd IEEE/ACM International Conference on Software Engineering, ICSE 2021, Madrid, Spain, 22-30 May 2021},
  year      = {2021},
}
```



