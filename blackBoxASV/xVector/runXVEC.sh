#!/bin/bash

exp=$1
inputs=$2

./genTestSamples.sh $exp $inputs
./test_xvector.sh $exp spoof
./test_xvector.sh $exp adv
