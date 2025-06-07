
### IMPORTS ###

print('[Importing libraries...]')

import soundfile as sf
import torch
import librosa
import numpy as np
import sys
import os
from os.path import exists, join, basename, splitext
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC, logging
from datasets import load_dataset

from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from encoder.audio import preprocess_wav
from vocoder import inference as vocoder
from pathlib import Path
import argparse
from utils.argutils import print_args

import jiwer
import speechmetrics
from asrtoolkit import cer
import nltk

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
### STD OUT SUPPRESSION UTILITY ###

class suppress_output:
    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args):
        if self.suppress_stdout:
            sys.stdout = self._stdout
        if self.suppress_stderr:
            sys.stderr = self._stderr


### MODELS DOWNLOAD ###
print('[Loading models...]')

dir = os.getcwd()

logging.set_verbosity_error()

with suppress_output(suppress_stdout=True, suppress_stderr=True):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

    encoder.load_model(Path(dir + "/saved_models/default/encoder.pt"))
    synthesizer = Synthesizer(Path(dir + "/saved_models/default/synthesizer.pt"))
    vocoder.load_model(Path(dir + "/saved_models/default/vocoder.pt"))

### FUNCTIONS and GLOBAL VARIABLES ###

SAMPLE_RATE = 16000

def synthesize(embed, text):
    print('[Synthesizing new audio...]')
    print('Text: ' + text + '\n')
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    return generated_wav

### MAIN ###

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", type=Path, default=None, help="Folder z plikami WAV, które będą targetem (głos)")
    parser.add_argument("--string", type=str, required=True, help="Tekst do syntezy (ten sam dla wszystkich)")
    parser.add_argument("--metrics", action="store_true", help="Wypisz metryki")
    parser.add_argument("--enhance", action="store_true", help="Usuń ciszę z wyjściowego audio")
    parser.add_argument("--seed", type=int, default=None, help="Seed do losowości")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.input_dir is None or not args.input_dir.is_dir():
        raise Exception("Musisz podać prawidłowy folder z plikami WAV jako --input_dir")

    transcription = args.string

    for target_wav in sorted(args.input_dir.glob("*.wav")):
        print(f"\nProcessing target voice file: {target_wav.name}")

        # Wczytaj plik jako target_audio
        target_audio, _ = librosa.load(target_wav, sr=SAMPLE_RATE)
        embedding = encoder.embed_utterance(encoder.preprocess_wav(target_audio, SAMPLE_RATE))

        # Syntezuj z tym embeddingiem i tekstem
        out_audio = synthesize(embedding, transcription)

        if args.enhance:
            out_audio = preprocess_wav(out_audio)

        output_path = args.input_dir / f"synth_{target_wav.stem}.wav"
        sf.write(output_path, out_audio, SAMPLE_RATE)
        print(f"Saved synthesized audio to {output_path}")

        if args.metrics:
            input_values = tokenizer(np.asarray(out_audio), return_tensors="pt").input_values
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription_out = tokenizer.batch_decode(predicted_ids)[0]

            ground_truth = transcription
            hypothesis = transcription_out

            wer_before_lemma = jiwer.wer(ground_truth, hypothesis)

            nltk.download('wordnet', quiet=True)
            wnl = nltk.stem.WordNetLemmatizer()
            stm = nltk.stem.snowball.EnglishStemmer()

            transcription_lemma = " ".join((stm.stem(wnl.lemmatize(s)) for s in ground_truth.split())).upper()
            transcription_out_lemma = " ".join((stm.stem(wnl.lemmatize(s)) for s in hypothesis.split())).upper()

            wer_after_lemma = jiwer.wer(transcription_lemma, transcription_out_lemma)
            cer_score = cer(ground_truth, hypothesis)
            mer_score = jiwer.mer(ground_truth, hypothesis)
            wil_score = jiwer.wil(ground_truth, hypothesis)

            with suppress_output(suppress_stdout=True, suppress_stderr=True):
                metrics = speechmetrics.load('absolute.mosnet', None)
            results = metrics(str(output_path))

            print('\n[METRICS]')
            print(f"Detected text: {transcription_out}")
            print(f"Original lemmatized: {transcription_lemma}")
            print(f"Synthesized lemmatized: {transcription_out_lemma}")
            print(f"WER before lemmatization: {wer_before_lemma:.4f}")
            print(f"WER after lemmatization: {wer_after_lemma:.4f}")
            print(f"CER: {cer_score/100:.4f}")
            print(f"MER: {mer_score:.4f}")
            print(f"WIL: {wil_score:.4f}")
            print(f"MOSNet: {results['mosnet'][0][0]:.4f}")

    print("\n[All done]\n")

