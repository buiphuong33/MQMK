# Multiple Queries with Multiple Keys (MQMK) Implementation

This repository is the official implementation of MQMK: A Precise Prompt Matching Paradigm for Prompt-based Continual Learning.

## Environment
The environment we are using is as follows:
* Ubuntu 16.04.6 LTS
* Python 3.9.19
* NVIDIA GeForce RTX 4090
* NVIDIA GeForce RTX 2080Ti

## Usage
First, clone this project.
Then, install the packages below:
```
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
torchprofile==0.0.4
torchvision==0.13.1
```
You can also install it by running:
```
pip install -r requirements.txt
```

## Data preparation
If you have the data, you just need to specify `--data-path` when running the command. If you don't have CIFAR100 and ImageNet-R, you can change the parameters in `datasets.py` to automatically download them.

**CIFAR-100**
```
datasets.CIFAR100(download=True)
```

**ImageNet-R**
```
Imagenet_R(download=True)
```
Sorry, we do not provide the DomainNet dataset. You need to download it from the [official website](https://ai.bu.edu/M3SDA/) and store it in your `--data-path`.
The first time you run the project, it may be slower because the project is processing the dataset, performing dataset splits, and other operations.
This operation only needs to be done once.


## Training
To train a model via command line:
**CIFAR-100**
```
sh ./train_cifar_10tasks.sh
```
**ImageNet-R**
```
sh ./train_imr_5tasks.sh
sh ./train_imr_10tasks.sh
sh ./train_imr_20tasks.sh
```
**DomainNet**
```
sh ./train_dmn_10tasks.sh
```
We have fixed the random seed, and we hope you will obtain the same results as we did.

## Parameter Description
`--multi_query True` corresponds to MQ, while the opposite is SQ.  
`--multi_key True` corresponds to MK, while the opposite is SK.  
You can easily set these two parameters to achieve the same ablation studies as in the paper.

`--e_prompt_layer_idx` specifies the depth of the e-prompt, and `--length` specifies the length of the e-prompt.
`--class_group` specifies how many categories share a single key. Setting it to 1 indicates that class-level keys are used.
`--k_key` corresponds to $K$ in the paper.
`--perfect_match` is the switch for the ideal perfect match scenario.
By adjusting these parameters, you can almost run all the experiments from the paper.


## License
This repository is private during the review process and is intended for use by reviewers and chairs.
