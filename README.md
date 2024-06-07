# VisusArbiter 

Capstone project for T5 Bootcamp (Data Science and Artificial Intelligence)

![logo](https://github.com/khuzamas/VisusArbiter/assets/46532822/6c570f7b-42ae-4768-8369-46acca311a70)


## Problem Description

**Task:** Multi-View foul detection

**Input:** 1 video of football foul

**Output:** 2 labels predictions (described below)

This task is part of [SoccerNet's](https://www.soccer-net.org/challenges/2024) challenges for 2024!

## Labels/Classes

**Label 1**: Offence (0: 0: No foul, 1: Foul and No card, 2: Foul and Yellow card, 3: Foul and Red Card)

**Label 2**: Foul type (0: Tackling, 1: Standing Tackling, 2: Holding, 3: Pushing, 4: High Leg, 5: Dive, 6: Challenge, 7: Elbowing)

## Data

Download data using 

```
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

mySNdl = SNdl(LocalDirectory="./SoccerNet")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password="PASSWORD")
```

## Libraries Used
* SoccerNet
* numpy
* pandas
* transformers
* torch
* keras
* json
* os
* sklearn
* streamlit

## Model Description

1. Feature Extraction using a pretrained VideoMAE [found here](https://huggingface.co/anirudhmu/videomae-base-finetuned-soccer-action-recognitionx4)
2. Model using the MultiView Classification architecture (one model for each label)
3. Use majority voting to predict classes for the action (1-4 multiple angled foul videos) 

