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

    # args parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=None, help=\
        "Source speaker for voice conversion.")
    parser.add_argument("--target", type=Path, default=None, help=\
        "Target speaker for voice conversion.")
    parser.add_argument("--text", type=str, default=None, help=\
        "Sentence used by the source.")
    parser.add_argument("--string", type=str, default=None, help=\
        "Sentence used by the synthesized voice.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("-m", "--metrics", action="store_true", help=\
        "Print some metrics.")
    parser.add_argument("-e", "--enhance", action="store_true", help=\
        "Trims output audio silences.")

    args = parser.parse_args()
    # print_args(args, parser)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        #synthesizer = Synthesizer(args.syn_model_fpath, verbose=False)

    if args.source is not None:
        source_audio, _ = librosa.load(args.source, sr=SAMPLE_RATE)

    if args.target is not None:
        target_audio, _ = librosa.load(args.target, sr=SAMPLE_RATE)

    if args.string is not None:
        if args.source is not None:
            raise Exception("[ERROR] Can't specify both source and string args.")
        transcription = args.string
        # text = listToString(transcription)
    elif args.text is not None and args.source:
        transcription = args.text
    else:
        input_values = tokenizer(np.asarray(source_audio), return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]

    embedding = encoder.embed_utterance(encoder.preprocess_wav(target_audio, SAMPLE_RATE))

    out_audio = synthesize(embedding, transcription)
    if args.enhance:
        out_audio = preprocess_wav(out_audio)

    sf.write(dir + "/audio_out.wav", out_audio, 16000)

    if args.metrics:
        input_values = tokenizer(np.asarray(out_audio), return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription_out = tokenizer.batch_decode(predicted_ids)[0]
        # text_out = listToString(transcription_out)

        ground_truth = transcription
        hypothesis = transcription_out

        wer_before_lemma = jiwer.wer(ground_truth, hypothesis)  #word error rate

        print('\n')

        nltk.download('wordnet')
        wnl = nltk.stem.WordNetLemmatizer()
        stm = nltk.stem.snowball.EnglishStemmer()

        transcription_lemma = ""
        transcription_out_lemma = ""

        for s in transcription.split(" "):
            transcription_lemma += (stm.stem(wnl.lemmatize(s)) + " ").upper()

        for s in transcription_out.split(" "):
            transcription_out_lemma += (stm.stem(wnl.lemmatize(s)) + " ").upper()

        wer_after_lemma = jiwer.wer(transcription_lemma, transcription_out_lemma)
        cer = cer(ground_truth, hypothesis)        #character error rate
        mer = jiwer.mer(ground_truth, hypothesis)  #match error rate
        wil = jiwer.wil(ground_truth, hypothesis)  #word information lost

        #mosnet

        window_length = None
        with suppress_output(suppress_stdout=True, suppress_stderr=True):
            metrics = speechmetrics.load('absolute.mosnet',window_length)
        results = metrics(dir + "/audio_out.wav")

        print('\n\n[+++METRICS+++]\n')

        print('Detected text: ' + transcription_out)

        print('Original text lemmatized: ' + transcription_lemma)
        print('Synthesized text lemmatized: ' + transcription_out_lemma)

        print('\nWord Error Rate before lemmatization is: ', wer_before_lemma)
        print('Word Error Rate after lemmatization is: ', wer_after_lemma)
        print('Character Error Rate is: ', cer/100)
        print('Match Error Rate is: ', mer)
        print('Word Information Lost is: ', wil)
        print('MOSNet is: ', results['mosnet'][0][0])
        print('\n[Done]\n')
    else:
        print('\n[Done]\n')
