#!/bin/bash
substr () {
    t=$1
    searchstring=$2
    rest=${t#*$searchstring}
    start=$(( ${#t} - ${#rest} ))
    searchstring=$3
    rest=${t#*$searchstring}
    end=$(( ${#t} - ${#rest} - ${#searchstring} - $start ))
    echo "${t:start:end}"
}

status () {
    echo $(substr "$1" "\"RecognitionStatus\":" ",\"Offset\":")
}

text () {
    echo $(substr "$1" ",\"DisplayText\":" "\}")\"
}
#?language=en-$3"
transcribe () {
    res=$(curl --location --request POST \
    "https://westus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-$3" \
    --header "Ocp-Apim-Subscription-Key: $2" \
    --header "Content-Type: audio/wav" \
    --data-binary @$1 2>/dev/null)
    echo $res
}

res=`transcribe "$1" "$2" "$3"`
#echo $res
status=`status "$res"`
text=`text "$res"`
echo "$status|$text"
