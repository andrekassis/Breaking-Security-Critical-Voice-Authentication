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

## Demo sample are available [here](https://drive.google.com/drive/folders/1LpyXdWo3O5qdGOitxqzn2wHxUDurXUh8).

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
```

## Prepare Environment

 1) Download the ASVSpoof2019 Dataset from this [link](https://drive.google.com/file/d/1_DzDEpEpWjavJ7YhWuYUkzuWuvDzUt55/view?usp=sharing)
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

The results will be saved under the directory "experiments" under a subdirectory with a unique id EXP_ID.

## Attack Configurations
The (default) attack configuration file is found in inputs/conf.yaml. To reproduce the paper's results, do not change this file. You can also change the configurations in this file to experiment with different hyperparameters or baseline adversarial attacks.  

Note: TIME_DOMAN_ATTACK stands for the known I-FGSM attack referenced in the paper, while the currently used attack is CM_Attack, which is the novel attack involving the transformations the paper introduces.

## Training/testing models
Although all pretrained models are available, you can train your own version of these models or even evaluate their performance on the relevant dataset using the train.py and test.py scripts (currently, training is only possible for CMs, while testing supports both ASVs and CMs).

To train a CM, just run:
```
python train.py --config inputs/configMap.yaml --system SYSTEM --subset SUBSET --task cm --devices cuda:0,cuda:1
```

Where SYSTEM is the system you wish to train - see inputs/configMap.yaml for examples.
SUBSET can either be "train" or "dev".

You can use test.py similarly.

## Transcription

If you want to run the transcription tests with Google Cloud or Azure, you need to install the following requirements:
```
jiwer
google-cloud-speech
```

Then, depending on the service you wish to use, you need to activate this service (enable STT in the service's console) and get an API key. The API key should be stored in transcription/keys/google.json for Google Cloud or transcription/keys/azure.json for Azure.

Note: Google cloud's API key will automatically be a JSON file, while Azure will give you a string from which you need to generate a json file and store in the above location. Given the Azure key KEY, the file's content should be:
```
{
  key: KEY
}
```

To run transcription, cd into "transcription" and execute:
```
python transcription_test.py --exp EXP_ID --service SERVICE
```

Where service is one of azure|google and EXP_ID is the id discussed earlier you get after executing the attack script.

## Blackbox ASVs
To evaluate the attacks against the blackbox ASVs discussed in the paper (with the exception of AWS), you need the following requirements:
```
tensorflow
resemblyzer
kaldi_io
```

In addition, you will have to install [kaldi](https://kaldi-asr.org/doc/install.html).

Finally, you need to download and extract the pre-trained xvector model on the Voxceleb dataset. cd into blackBoxASV/xVector and download the model from [here](https://drive.google.com/file/d/1SW66KM17mmPly61wMPmRMKuCvz4WWQ-t/view?usp=share_link). Then execute:
```
tar -zxvf xvector_nnet.tar.gz
```

