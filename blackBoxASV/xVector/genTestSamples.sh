#!/bin/bash

exp=$1
inputs=$2
ref=../../experiments/$exp/ref_fin.txt
wavdir=../../datasets/asvspoofWavs/wavs
outdir=../../experiments/$exp/xvector_dir
name=`echo $RANDOM | md5sum | head -c 20; echo`

outdirAdv=$outdir/adv
outdirSpoof=$outdir/spoof

makeTestMal () {
    out=$1/test
    ext=$2
    experiment=$3

    cat $out/voxceleb1_test_v2.txt | cut -f2 -d" " | while read -r line; do 
        dir=`echo $line | cut -f-2 -d"/"`
	file=`echo $line | cut -f1 -d"." | cut -f3 -d"/"`
	mkdir -p $out/wav/$dir 
	cp ../../experiments/$experiment/wavs/${file}-${ext}.wav $out/wav/$dir/${file}.wav
    done
}

makeTestBen () {
    out=$1/test
    wavs=$2

    cat $out/voxceleb1_test_v2.txt | cut -f3 -d" " | while read -r line; do 
        dir=`echo $line | cut -f-2 -d"/"`
	file=`echo $line | cut -f1 -d"." | cut -f3 -d"/"`
	mkdir -p $out/wav/$dir
	cp $wavs/${file}.wav $out/wav/$dir/${file}.wav
    done
}

if [ ! -d ../../experiments/$exp/ ]; then
    echo "$exp does'nt exist"
    exit 1
fi

if [ -d $outdir ]; then
    echo "$outdir exists"
    exit 1
fi

mkdir -p $outdirAdv/test/wav
mkdir -p $outdirSpoof/test/wav

echo "generating test file"

paste <(cat ../../inputs/$inputs | cut -f1 -d" ") $ref -d" " | while read -r line; do 
    spkr=`echo $line | cut -f1 -d" "`
    mal=`echo $line | cut -f2 -d" "`
    rest=`echo $line | cut -f3- -d" " | sed 's/ /\n/g'`
    while read -r rst; do
	echo "0 $spkr/$mal/${mal}.wav $spkr/$rst/${rst}.wav" >> voxceleb1_test_v2-${name}.txt
    done < <(echo "$rest")
done

cp voxceleb1_test_v2-${name}.txt $outdirSpoof/voxceleb1_test_v2.txt
mv voxceleb1_test_v2-${name}.txt $outdirAdv/voxceleb1_test_v2.txt

cp $outdirAdv/voxceleb1_test_v2.txt $outdirAdv/test/voxceleb1_test_v2.txt
cp $outdirSpoof/voxceleb1_test_v2.txt $outdirSpoof/test/voxceleb1_test_v2.txt

echo "generting benign test dirs"

makeTestBen $outdirAdv $wavdir
cp -r $outdirAdv/test/wav/* $outdirSpoof/test/wav/

echo "generating adv test dirs"
makeTestMal $outdirAdv adv $exp

echo "generating spoof test dirs"
makeTestMal $outdirSpoof orig $exp

echo "done"
