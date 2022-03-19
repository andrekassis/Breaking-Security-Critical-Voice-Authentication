import os
import argparse
import speech_recognition as sr


def transcribe(r, audio):
    try:
        with sr.AudioFile(audio) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
        return text.capitalize()
    except:
        return None


def log(f, s_id, orig_text, adv_text):
    f.write("######################\n")
    f.write(s_id + " original: " + str(orig_text) + "\n")
    f.write(s_id + " adversarial: " + str(adv_text) + "\n")
    f.write("########################\n\n")
    if (orig_text is None and adv_text is not None) or (
        orig_text is not None and adv_text is None
    ):
        return False
    return int(orig_text == adv_text)


def run_stt_eval(wav_dir, out_f, inputs):
    with open(inputs, "r", encoding="utf-8") as f:
        inputs = [line.strip() for line in f]
    r = sr.Recognizer()
    score = 0

    with open(out_f, "w", encoding="utf-8", buffering=1) as f:
        for line in inputs:
            orig = os.path.join(wav_dir, line + "-orig.wav")
            adv = os.path.join(wav_dir, line + "-adv.wav")

            orig_text = transcribe(r, orig)
            adv_text = transcribe(r, adv)

            score += log(f, line, orig_text, adv_text)
        score /= len(inputs)
        f.write("score: " + str(score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp")
    args = parser.parse_args()
    wav_dir = os.path.join(args.exp, "wavs")
    out_f = os.path.join(args.exp, "transcription_output.txt")
    run_stt_eval(wav_dir, out_f, os.path.join(args.exp, "ids.txt"))
