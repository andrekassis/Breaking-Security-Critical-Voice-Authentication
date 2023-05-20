#!/bin/bash

export KALDI_ROOT=$HOME/kaldi
export train_cmd="run.pl"
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PWD:$PATH
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
set -e

if [ $# != 2 ]; then
    exit 1
fi

exp=$1
typ=$2

data_dir=../../experiments/$exp/xvector_dir/$typ
trials=test_$exp/trials
nnet_dir=xvector_nnet

local/make_voxceleb1_v2.pl $data_dir test test_$exp

python extract_mfcc.py --test-wav-file test_$exp/wav.scp --num-workers 5 \
                       --out-dir test_$exp/mfcc --test-rstfilename test || exit 1

cp test_$exp/mfcc/test.scp test_$exp/feats.scp || exit 1

python local/compute_vad.py --feats-scp test_$exp/feats.scp --vad-ark test_$exp/mfcc/asvspoof_vad.ark --vad-scp test_$exp/vad.scp || exit 1
python local/utt2num_frame.py --vad-scp test_$exp/vad.scp --utt2num_frames test_$exp/utt2num_frames || exit 1
utils/fix_data_dir.sh test_$exp || exit 1

sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 1 $nnet_dir test_$exp test_$exp/xvectors
sed -i 's/\.wav//g' $trials

$train_cmd test_$exp/log.log \
  ivector-plda-scoring --normalize-length=true \
  "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train/plda - |" \
  "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:test_$exp/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:test_$exp/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
  "cat '$trials' | cut -d\  --fields=1,2 |" test_$exp/scores || exit 1;

cp test_$exp/scores $data_dir/
python attack_success_rate.py -d $data_dir
rm -rf test_$exp
