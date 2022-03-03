module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch==1.9.0
pip install --no-index wheels/kaldi_io-0.9.4-py3-none-any.whl 
pip install --no-index wheels/SoundFile-0.10.3.post1-py2.py3-none-any.whl
pip install --no-index tqdm
pip install --no-index tensorboardX
pip install --no-index matplotlib
pip install --no-index wheels/pooch-1.2.0-py3-none-any.whl
pip install --no-index wheels/resampy-0.2.2.tar.gz
pip install --no-index wheels/librosa-0.8.0.tar.gz
pip install --no-index /home/akassis/adversarial_robustness_toolbox-1.7.2-py3-none-any.whl
pip install --no-index pyyaml
pip install --no-index torchaudio
pip install --no-index --no-deps scikit-image
pip install --no-index wheels/pytorch_forecasting-0.9.0-py3-none-any.whl
pip install --no-index wheels/noisereduce-2.0.0-py3-none-any.whl
pip install --no-index wheels/webrtcvad-2.0.10.tar.gz
pip install --no-index wheels/pydub-0.25.1-py2.py3-none-any.whl

