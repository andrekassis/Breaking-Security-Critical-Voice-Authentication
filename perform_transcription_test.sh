#!/bin/bash

cat $1/results.txt | cut -f1 -d":" > $1/ids.txt
python transcription_test.py --exp $1
