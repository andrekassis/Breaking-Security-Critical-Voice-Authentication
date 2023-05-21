# Breaking-Security-Critical-Voice-Authentication
Source code for paper "Breaking Security-Critical Voice Authentication".
Our paper has been accepted by 44th IEEE Symposium on Security and Privacy (IEEE S&P, Oakland), 2023.

If you use this work, please consider citing it as:
```
@INPROCEEDINGS {kassis2023breaking,
    author = {Andre Kassis and Urs Hengartner},
    booktitle = {2023 IEEE Symposium on Security and Privacy (S&P)},
    title = {Breaking Security-Critical Voice Authentication},
    year = {2023}
}
```

## Demo sample are available [here] (https://drive.google.com/drive/folders/1LpyXdWo3O5qdGOitxqzn2wHxUDurXUh8).

## Requirements
```
torch
functools
SoundFile
tqdm
tensorboardX
matplotlib
resampy
librosa
adversarial_robustness_toolbox
pyyaml
pydub
webrtcvad
scipy
fairseq

jiwer
google-cloud-speech

tensorflow
resemblyzer
kaldi_io
```

## Prepare Environment

 1) Download the ASVSpoof2019 Dataset from this [link] (https://drive.google.com/file/d/1_DzDEpEpWjavJ7YhWuYUkzuWuvDzUt55/view?usp=sharing)
 2) You need to download the pretrained CM and ASV models, which are available [here](https://drive.google.com/file/d/1qK1FLPokwwBKHyTMDYoStUxRenev3yn5/view?usp=sharing)
 3) Extract the downloaded tars in this directory: 
  ```
  tar -zxvf trained_models.tar.gz
  tar -zxvf datasets.tar.gz
  ```

## Run Adversarial Attacks
To run the attacks, execute: 
```
python attack.py --device "cuda:0"
```
