#%% Imports
from pathlib import Path
import subprocess as sp
import essentia.standard as es
from lag import config as cfg
import numpy as np
import librosa
from multipledispatch import dispatch
from torch import Tensor
import torchaudio

from lag.data.labels import GENRE_LABELS, MOOD_THEME_CLASSES, INSTRUMENT_CLASSES
from lag.utils import audio as audio_utils
import essentia

#%% Download models
if False:
    sp.call([
        "curl",
        "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
        "--output", "genre_discogs400-discogs-effnet-1.pb"
    ])
    sp.call([
        "curl",
        "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
        "--output", "discogs-effnet-bs64-1.pb"
    ])
    sp.call([
        "curl",
        "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
        "--output", "mtg_jamendo_moodtheme-discogs-effnet-1.pb"
    ])
    sp.call([
        "curl",
        "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb",
        "--output", "mtg_jamendo_instrument-discogs-effnet-1.pb"
    ])


def filter_predictions(predictions, class_list, threshold=0.1):
    predictions_mean = np.mean(predictions, axis=0)
    sorted_indices = np.argsort(predictions_mean)[::-1]
    filtered_indices = [
        i for i in sorted_indices if predictions_mean[i] > threshold
    ]
    filtered_labels = [class_list[i] for i in filtered_indices]
    filtered_values = [predictions_mean[i] for i in filtered_indices]
    return filtered_labels, filtered_values


def make_comma_separated_unique(tags):
    seen_tags = set()
    result = []
    for tag in ', '.join(tags).split(', '):
        if tag not in seen_tags:
            result.append(tag)
            seen_tags.add(tag)
    return ', '.join(result)


@dispatch(Path)
def get_audio_features(audio_filename: Path):  # type: ignore
    audio = audio_utils.load_audio(audio_filename, 16_000, False).squeeze()
    # audio = es.MonoLoader(filename=str(audio_filename),
    #                       sampleRate=16000,
    #                       resampleQuality=4)()

    return get_audio_features(audio, 16_000)


@dispatch(Tensor, int, Path)
def get_audio_features(audio: Tensor, sr: int,
                       models_dir: Path):  # type: ignore

    essentia.log.infoActive = False
    audio = audio_utils.to_mono(audio)
    audio = torchaudio.functional.resample(audio, sr, 16_000).squeeze()
    audio = audio.numpy()

    embedding_model = es.TensorflowPredictEffnetDiscogs(
        graphFilename=str(models_dir / "discogs-effnet-bs64-1.pb"),
        output="PartitionedCall:1")
    embeddings = embedding_model(audio)

    result_dict = {}

    # Predicting genres
    genre_model = es.TensorflowPredict2D(
        graphFilename=str(models_dir / "genre_discogs400-discogs-effnet-1.pb"),
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0")
    predictions = genre_model(embeddings)

    filtered_labels, _ = filter_predictions(predictions, GENRE_LABELS)
    filtered_labels = ', '.join(filtered_labels).replace("---",
                                                         ", ").split(', ')
    result_dict['genres'] = make_comma_separated_unique(filtered_labels)

    # Predicting mood/theme
    mood_model = es.TensorflowPredict2D(
        graphFilename=str(models_dir /
                          "mtg_jamendo_moodtheme-discogs-effnet-1.pb"))
    predictions = mood_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions,
                                            MOOD_THEME_CLASSES,
                                            threshold=0.05)
    result_dict['moods'] = make_comma_separated_unique(filtered_labels)

    bpm, key = get_bpm_key(audio, sr)

    result_dict["bpm"] = bpm
    result_dict["key"] = key

    # Predicting instruments
    # instrument_model = es.TensorflowPredict2D(
    #     graphFilename="mtg_jamendo_instrument-discogs-effnet-1.pb")
    # predictions = instrument_model(embeddings)
    # filtered_labels, _ = filter_predictions(predictions, INSTRUMENT_CLASSES)
    # result_dict['instruments'] = filtered_labels

    return result_dict


@dispatch(Path)
def get_bpm_key(audio_filename):
    y, sr = librosa.load(str(audio_filename))
    get_bpm_key(y, sr)


@dispatch(np.ndarray, int)
def get_bpm_key(audio: np.ndarray, sr: int):
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = round(tempo[0])
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    key = np.argmax(np.sum(chroma, axis=1))
    key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key]
    length = librosa.get_duration(y=audio, sr=sr)

    return tempo, key


#%% Test on demo audio
if __name__ == "__main__":
    audio_filename = cfg.AUDIO_DIR / "cake.wav"
    features = get_audio_features(audio_filename)

# %
