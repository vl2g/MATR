# MATR: Aligning Moments in Time using Video Queries (CVPR 2025 Submission)

[![paper](https://img.shields.io/badge/paper-paper-cyan)](https://github.com/vl2g/MATR?tab=readme-ov-file)
[![Webpage](https://img.shields.io/badge/Webpage-green)](https://github.com/vl2g/MATR?tab=readme-ov-file)

## Overview
<p align="center">
    <img src="assets/model.png" width="700px"/>
</p>

This repo contains the code for training, inference, and evaluation of MATR, which implements a SOTA approach for video moment retrieval using video queries.

## To setup environment
```
# create new env MATR
$ conda create -n MATR python=3.10.4

# activate MATR
$ conda activate MATR

# install pytorch, torchvision
$ conda install -c pytorch pytorch torchvision

# install other dependencies
$ pip install -r requirements.txt
```

## Training 
```
# set the path and required parameters in the train.sh
$ bash train.sh
```


## Inference
```
# set the path and required parameters in the inference.py
$ python inference.py
```

## Evaluation
```
# set the path and required parameters in the eval.py
$ python eval.py
```

## Qualitative Results
<p align="center">
    <img src="assets/qual_res.png" width="700px"/>
</p>

# Acknoledgment
Our code base is built upon the following open-source codes
1. https://github.com/showlab/UniVTG
2. https://github.com/SamsungLabs/Drop-DTW

