# MATR: Aligning Moments in Time using Video Queries (ICCV 2025)

[![paper](https://img.shields.io/badge/paper-paper-cyan)](https://github.com/vl2g/MATR?tab=readme-ov-file)
[![Webpage](https://img.shields.io/badge/Webpage-green)](https://github.com/vl2g/MATR?tab=readme-ov-file)

## Overview
<p align="center">
    <img src="assets/model.png" width="700px"/>
</p>

This repo contains the official code for training, inference, and evaluation of *MATR* from the *ICCV'25* paper ["Aligning Moments in Time using Video Queries"](https://drive.google.com/file/d/1GlaroeUz6uqOIV9SHz8yzx-zJzKfZKV_/view), which implements a SOTA approach for video moment retrieval using video queries.

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
[checkpoint](https://drive.google.com/file/d/1C2sKb_JGPY2ho8aM7Lz_4UC2_anP6Stt/view?usp=drive_link)
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

# Acknowledgement
Our codebase is built upon the following open-source repositories:
1. https://github.com/showlab/UniVTG
2. https://github.com/SamsungLabs/Drop-DTW
3. https://github.com/jayleicn/moment_detr

## Contact

Please feel free to open an issue or email us at [kumar.204@iitj.ac.in](mailto:kumar.204@iitj.ac.in) / [agarwaluday@iitj.ac.in](mailto:agarwaluday@iitj.ac.in)
