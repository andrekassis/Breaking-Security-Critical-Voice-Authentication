import os
import argparse
import jiwer
from google.cloud import speech
import io
import json
from pathlib import Path
from collections import OrderedDict
import librosa
import tempfile
import soundfile as sf

werMap = {"azure": 0.1651, "google": 0.1582}


class azure:
    def __init__(self, key=None):
        assert key is not None
        with Path(key).open("rt", encoding="utf8") as handle:
            config = json.load(handle, object_hook=OrderedDict)
        self.key = config["key"]
        # with open(key, 'r') as f:
        #    self.key = [ line.strip() for line in f ][0]

    def __call__(self, audio, sr, dialect="US"):
        # print("./azure_stt.sh " + audio + " " + self.key + " " + dialect)
        text = [
            string[1:-1]
            for string in os.popen(
                "./azure_stt.sh " + audio + " " + self.key + " " + dialect
            )
            .read()
            .strip()
            .split("|")
        ]
        if text[0] != "Success":
            return None
        return text[1].capitalize()


class google:
    def __init__(self, key):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key
        self.client = speech.SpeechClient()

    def _response(self, sample, sr, dialect="US", **kwargs):
        model = "video"
        config = speech.types.RecognitionConfig(
            encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
            audio_channel_count=1,
            language_code="en_" + dialect,
            sample_rate_hertz=sr,
            model=model,  # "phone_call", #"video",
            use_enhanced=True,
            **kwargs
        )
        try:
            response = self.client.recognize(
                config=config,
                audio=sample,
            )
            return response.results[0].alternatives[0]  # .transcript.capitalize()
        except Exception as e:
            print(e)
            return None

    def __call__(self, audio, sr, dialect="US", **kwargs):
        with io.open(audio, "rb") as audio_file:
            content = audio_file.read()
            sample = speech.types.RecognitionAudio(content=content)
        if sr is None:
            sr = 16000
        return self._response(sample, sr, dialect, **kwargs)


def _wer(orig_text, adv_text):
    if (orig_text is None and adv_text is not None) or (
        orig_text is not None and adv_text is None
    ):
        wer = 1
    elif orig_text is None:
        wer = 0
    else:
        orig_text, adv_text = jiwer.ExpandCommonEnglishContractions()(
            [orig_text, adv_text]
        )
        orig_text, adv_text = jiwer.RemovePunctuation()([orig_text, adv_text])
        orig_text, adv_text = jiwer.Strip()([orig_text, adv_text])
        orig_text, adv_text = jiwer.ToLowerCase()([orig_text, adv_text])
        orig_text, adv_text = jiwer.RemoveMultipleSpaces()([orig_text, adv_text])
        wer = jiwer.wer(orig_text, adv_text)
    return wer


def _log(f, s_id, orig_text, adv_text, WER_limit, verbose):
    wer = _wer(orig_text, adv_text)
    Message = "######################\n"
    Message += s_id + " original: " + str(orig_text) + "\n"
    Message += s_id + " adversarial: " + str(adv_text) + "\n"
    Message += "WER: " + str(wer) + "\n"
    Message += "########################\n\n"
    f.write(Message)
    if verbose:
        print(Message)
    return wer <= WER_limit


def load(f, sr=None):
    if sr is None:
        return f
    temp = tempfile.NamedTemporaryFile(suffix=".wav")
    temp.close()
    x = librosa.load(f, sr=sr)[0]
    sf.write(temp.name, x, sr)
    return temp.name


def unload(f, sr=None):
    if sr is None:
        return
    os.unlink(f)


def run_stt_eval(wav_dir, out_f, inputs, transcribe, WER_limit, dialects, verbose):
    with open(inputs, "r", encoding="utf-8") as f:
        inputs = [line.strip().split(" ")[1] for line in f]
    score = 0

    sr = 8000
    with open(out_f, "w", encoding="utf-8", buffering=1) as f:
        for l, line in enumerate(inputs):
            orig = os.path.join(wav_dir, line + "-orig.wav")
            adv = os.path.join(wav_dir, line + "-adv.wav")
            orig = load(orig, sr)
            adv = load(adv, sr)
            orig_text = transcribe(orig, sr, "US")  # dialects[l])
            adv_text = transcribe(adv, sr, "US")  # dialects[l])
            unload(orig, sr)
            unload(adv, sr)
            score += _log(f, line, orig_text, adv_text, WER_limit, verbose)
        score /= len(inputs)
        f.write("score: " + str(score))

    print("score is: " + str(score))
    print("logs are in: " + out_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp")
    parser.add_argument("--ids", default="../inputs/transcription_inputs.txt")
    parser.add_argument("--service", default="google")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    wav_dir = os.path.join(os.path.join("../experiments", args.exp), "wavs")
    out_f = os.path.join(
        os.path.join("../experiments", args.exp),
        "transcription_output_" + args.service + ".txt",
    )

    transcribe = eval(args.service)(os.path.join("keys", args.service + ".json"))
    WER_limit = werMap[args.service]
    dialects = [
        "IE",
        "US",
        "SG",
        "SG",
        "SG",
        "NZ",
        "SG",
        "NZ",
        "NZ",
        "NZ",
        "IE",
        "SG",
        "SG",
        "SG",
        "SG",
        "SG",
        "HK",
        "TZ",
        "SG",
        "SG",
        "SG",
        "NZ",
        "SG",
        "SG",
        "IE",
    ]
    run_stt_eval(
        wav_dir, out_f, args.ids, transcribe, WER_limit, dialects, args.verbose
    )
