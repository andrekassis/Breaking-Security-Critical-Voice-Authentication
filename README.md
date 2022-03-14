# adversarial_cms

## Requirements
#### torch>=1.8.0
#### kaldi_io
#### SoundFile
#### tqdm
#### tensorboardX
#### matplotlib
#### resampy
#### librosa
#### adversarial_robustness_toolbox
#### pyyaml
#### pydub

## Prepare Environment
### 1) To run the attacks, you will first need to download the ASVSpoof2019 Dataset and extract all the samples (to a single folder together, redardless of the segment in the dataset they belong to - i.e., train/dev/eval) to a single folder asvspoofWavs/wavs/. 
### 2) You need to download the pretrained CM and ASV models, which are available at https://drive.google.com/file/d/1qK1FLPokwwBKHyTMDYoStUxRenev3yn5/view?usp=sharing
### 3) Extract the downloaded tar in this directory: tar -zxvf trained_models.tar.gz

## Run Adversarial Attacks
### To run the attacks, exectute: python attack.py --conf conf.yaml --device "cuda:0"

## TODO
### Explain hopw to train, test and attack any custom model. 
### Explain how to change attack type and hyperparameters using the configuration file

