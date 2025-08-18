import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
import torch.utils.data
from pathlib import Path
import torch
import itertools

from lag.conditioning.beat_embedder import Beat
from lag.data.stem import Stem
from lag.utils import audio as audio_utils
import json
from torch import Tensor
import torchaudio
# from collections.abc import Sized
import librosa

# import pyrubberband as pyrb
# import pylibrb

N_VALID_SAMPLES = 24
# 014f37 is removed
# EXPECTED_N_SONGS = 240
# EXPECTED_N_SONGS = 150


class StemmedDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root_dir: Path,
                 stems: Set[Stem],
                 target_stem: Stem,
                 single_stem: bool,
                 min_context_seconds: int,
                 use_style_conditioning: bool,
                 use_beat_conditioning: bool,
                 type_of_context: str,
                 bpm_in_caption: bool,
                 add_click: bool,
                 sync_chunks: bool,
                 train: bool,
                 sample_rate: int,
                 chunk_size_samples: int,
                 speed_transform_p: float,
                 pitch_transform_p: float,
                 stereo: bool = False,
                 max_genres_in_description: int = 3,
                 max_moods_in_description: int = 3,
                 n_samples_per_epoch: Optional[int] = None,
                 verbose: bool = False):
        self.root_dir: Path = root_dir
        self.stems: Set[Stem] = stems
        self.single_stem: bool = single_stem
        self.min_context_seconds: int = min_context_seconds
        self.target_stem: Stem = target_stem
        self.use_style_conditioning: bool = use_style_conditioning
        self.use_beat_conditioning: bool = use_beat_conditioning
        if self.use_style_conditioning and not self.single_stem:
            raise ValueError("You can only use style conditioning if "
                             "the target is a single stem")

        if self.target_stem != Stem.ANY:
            assert self.single_stem

        self.add_click: bool = add_click
        self.bpm_in_caption: bool = bpm_in_caption
        self.sync_chunks: bool = sync_chunks
        self.stem_names: Set[str] = {s.getname() for s in self.stems}
        self.train: bool = train
        self.sample_rate: int = sample_rate
        self.chunk_size_samples: int = chunk_size_samples
        self.speed_transform_p: float = speed_transform_p
        self.pitch_transform_p: float = pitch_transform_p
        self.stereo: bool = stereo
        self.max_genres_in_description: int = max_genres_in_description
        self.max_moods_in_description: int = max_moods_in_description
        self.verbose: bool = verbose

        self.type_of_context: str = type_of_context
        assert self.type_of_context in [
            "stems", "beats", "stems or beats", "stems and beats"
        ]

        if self.add_click or self.sync_chunks:
            if not (self.root_dir / "sync.json").exists():
                raise FileNotFoundError(
                    "If you want click or sync, I need a 'sync.json' file in "
                    "the top-level dir of the dataset")

            with open(self.root_dir / "sync.json", "r") as f:
                self.syncdata: Dict[str, List[int]] = json.load(f)

        # load all song directories
        self.song_names: List[str] = sorted(
            [p.name for p in self.root_dir.iterdir() if p.is_dir()])
        # assert len(self.song_names) == EXPECTED_N_SONGS

        # if has a single target stem filter out songs that don't have that stem
        if self.target_stem != Stem.ANY:
            toremove: List[str] = []
            for song_name in self.song_names:
                if not (self.root_dir / song_name /
                        self.target_stem.getname()).exists():
                    toremove.append(song_name)
            self.song_names = [s for s in self.song_names if s not in toremove]

        # train/valid split
        if self.train:
            self.song_names = self.song_names[:-N_VALID_SAMPLES]
        else:
            self.song_names = self.song_names[-N_VALID_SAMPLES:]

        # if self.verbose:
        print(f"Loaded {len(self.song_names)} for "
              f"{'train' if self.train else 'valid'} dataset.")

        # WTF? TODO: remove
        # if self.target_stem != Stem.ANY:
        #     # if only interested in a stem, remove songs without it
        #     for songname in self.song_names:
        #         songdir = self.root_dir / songname
        #         if not (songdir / self.target_stem.getname()).exists():
        #             self.song_names.remove(songname)

        # create iterator to run n_sample times
        self.n_samples: int = n_samples_per_epoch or len(self.song_names)

        self.song_iterator: List[str] = list(
            itertools.islice(itertools.cycle(iter(self.song_names)),
                             self.n_samples))

    def __len__(self):
        return self.n_samples

    def save_sample(self, sample: Dict, path: Path):
        audio_utils.save_audio(sample["wav"], path / "input.wav",
                               self.sample_rate)
        audio_utils.save_audio(sample["conditioning"].wav, path / "cond.wav",
                               self.sample_rate)
        audio_utils.save_audio(sample["conditioning"].wav + sample["wav"],
                               path / "mix.wav", self.sample_rate)

    def get_description(self,
                        features: Dict[str, str | int],
                        instruments: Sequence[Stem],
                        speed_factor: Optional[float] = None) -> str:

        genres: List[str] = str(features["genres"]).split(",")
        moods: List[str] = str(features["moods"]).split(",")
        description = ""

        # Genre
        if len(genres) > 0:
            # if more than max number of genres, choose first few
            # if len(genres) > self.max_genres:
            #     genres = random.sample(genres, self.max_genres)
            genres = [
                s.strip() for s in genres[:self.max_genres_in_description]
            ]
            description += (f"Genre{'s' if len(genres) > 1 else ''}: "
                            f"{', '.join(genres)}. ")

        # Mood
        if len(moods) > 0:
            # if more than max number of moods, choose first few
            # if len(moods) > self.max_moods:
            #     moods = random.sample(moods, self.max_moods)
            moods = [s.strip() for s in moods[:self.max_moods_in_description]]
            description += (f"Mood{'s' if len(moods) > 1 else ''}: "
                            f"{', '.join(moods)}. ")

        # Instruments
        if not (self.single_stem and self.target_stem != Stem.ANY):
            instrument_names: List[str] = [s.name.lower() for s in instruments]
            random.shuffle(instrument_names)
            description += f"Instruments: {', '.join(instrument_names)}."

        # BPM
        if self.bpm_in_caption:
            bpm = features["bpm"]
            if speed_factor:
                bpm = round(int(bpm) / speed_factor)
            description += f" Bpm: {bpm}."

        # Key
        # key = features["key"]
        # description += f"Key: {key}."

        return description

    def _transform_chunk(self, t: Tensor, speed_factor: float,
                         pitch_factor: int, target_size: int):
        if self.stereo:
            raise NotImplementedError(
                "No augmentations for stereo audio implemented")

        stretched = audio_utils.stretch_with_timeout(
            t,
            self.sample_rate,
            speed_factor,
            pitch_factor,
            2,
        )

        if stretched.shape[-1] < target_size:
            stretched = torch.nn.functional.pad(
                stretched, (0, target_size - stretched.shape[-1]), "constant",
                0)
        elif stretched.shape[-1] > target_size:
            stretched = stretched[..., :target_size]

        return stretched

    def load_stems(
        self,
        song_path: Path,
        song_stems: Iterable[Stem],
        start_offset: int,
        n_frames: int,
    ) -> Dict[Stem, Tensor]:

        # load audio chunks for each stem
        stem_tensors: Dict[Stem, Tensor] = {}
        for stem in song_stems:
            stemdir = song_path / stem.getname()
            stem_tensor: Tensor = torch.zeros(2 if self.stereo else 1,
                                              n_frames,
                                              dtype=torch.float32)

            # for each track of a stem
            for trackpath in stemdir.iterdir():
                # load wav
                chunk = audio_utils.load_audio_chunk(trackpath,
                                                     start_offset,
                                                     n_frames,
                                                     stereo=self.stereo)
                stem_tensor += chunk

            # if not silent, include it in dictionary of stems
            if not audio_utils.is_silent(stem_tensor, threshold=0.01):
                stem_tensors[stem] = stem_tensor

        return stem_tensors

    # def choose_conditioning(
    #         self,
    #         stems: Sequence[Stem]) -> Tuple[Sequence[Stem], Sequence[Stem]]:
    #     n_stems: int = len(stems)
    #     if n_stems == 1:
    #         # if only one stem, use it as input with no conditioning
    #         return stems[:], []
    #     n_conditioning_stems: int = random.randint(1, n_stems - 1)
    #     conditioning_stems: Sequence[Stem] = random.sample(
    #         stems, n_conditioning_stems)
    #     input_stems = list(filter(lambda x: x not in conditioning_stems, stems))
    #     return input_stems, conditioning_stems

    def choose_input_and_conditioning(
            self,
            stems: Sequence[Stem]) -> Tuple[Sequence[Stem], Sequence[Stem]]:
        n_stems: int = len(stems)

        if n_stems == 1:
            raise RuntimeError("This song has only one stem")
            return stems[:], []

        # choose a random number of context stems,
        # leaving at least 1 for the input
        if self.target_stem != Stem.ANY:
            assert self.target_stem in stems
            possible_conditioning_stems = [
                s for s in stems if s != self.target_stem
            ]
            n_conditioning_stems: int = random.randint(
                1, len(possible_conditioning_stems))
            conditioning_stems: Sequence[Stem] = random.sample(
                possible_conditioning_stems, n_conditioning_stems)
        else:
            n_conditioning_stems: int = random.randint(1, n_stems - 1)
            conditioning_stems: Sequence[Stem] = random.sample(
                stems, n_conditioning_stems)

        # choose a random number of the remaining stems as input
        n_input_stems: int = 1 if self.single_stem else random.randint(
            1, n_stems - n_conditioning_stems)

        # choose input stem
        if self.target_stem != Stem.ANY:
            # if target stem != ANY, that HAS to be the input
            assert self.target_stem in stems
            input_stems: Sequence[Stem] = [self.target_stem]
        else:
            possible_input_stems: Sequence[Stem] = list(
                filter(lambda x: x not in conditioning_stems, stems))
            input_stems: Sequence[Stem] = random.sample(possible_input_stems,
                                                        n_input_stems)

        return input_stems, conditioning_stems

    def add_click_to_track(self, wav: Tensor, wav_sr: int,
                           click_frames: Sequence[int],
                           start_offset: int) -> Tensor:

        for idx, frame in enumerate(click_frames):
            if frame >= start_offset:
                first_relevant_index = idx
                break
        else:
            return wav

        shifted_click_frames: List[int] = [
            f - start_offset for f in click_frames[first_relevant_index:]
        ]

        click_track: Tensor = audio_utils.create_click(list(wav.shape), wav_sr,
                                                       shifted_click_frames)
        assert click_track.shape == wav.shape

        mix: Tensor = wav + click_track * 0.5
        return mix

    def mix_input_and_conditioning(self, stem_tensors: Dict[Stem, Tensor],
                                   input_stems: Sequence[Stem],
                                   condition_stems: Sequence[Stem]):

        assert len(input_stems) > 0
        input_tensor = torch.stack([stem_tensors[s] for s in input_stems
                                   ]).sum(dim=-0)

        if len(condition_stems) > 0:
            condition_tensor = torch.stack(
                [stem_tensors[s] for s in condition_stems]).sum(dim=-0)
        else:
            condition_tensor = None

        return input_tensor, condition_tensor

    def find_good_chunk(
            self, song_name: str, n_frames_to_take: int, song_n_frames: int,
            song_path: Path,
            song_stems: Iterable[Stem]) -> Tuple[Dict[Stem, Tensor], int]:
        found_good_chunk: bool = False
        attempts: int = 0
        while not found_good_chunk:
            attempts += 1
            if attempts > 10 and (attempts - 1) % 10 == 0:
                print(
                    f"Tried to find some non-silent chunk of song {song_name} "
                    f"for {attempts} times but it's so hard please master "
                    "I am tired let me rest")

            # choose random chunk
            if self.sync_chunks:
                choices: List[int] = self.syncdata[song_name]
                choices = list(
                    filter(lambda x: (x + n_frames_to_take) < song_n_frames,
                           choices))
                start_offset: int = random.choice(choices)

            else:
                start_offset: int = random.randint(
                    0, song_n_frames - n_frames_to_take)

            # load song stems, filter out silent ones
            stem_tensors: Dict[Stem, Tensor] = self.load_stems(
                song_path,
                song_stems,
                start_offset,
                n_frames_to_take,
            )
            nonsilent_stems: List[Stem] = list(stem_tensors.keys())

            if self.target_stem != Stem.ANY:
                if self.target_stem in nonsilent_stems and (
                        len(nonsilent_stems)
                        > (1 if len(list(song_stems)) > 1 else 0)):
                    found_good_chunk = True
            else:
                if len(nonsilent_stems) > 0:
                    found_good_chunk = True

        return stem_tensors, start_offset  # type: ignore

    def __getitem__(self, idx: int) -> Dict[str, Tensor | str]:
        # output = {
        #     "name": "",
        #     "target": torch.rand(1, 320_000),
        #     "description": "",
        #     "context": torch.rand(1, 320_000),
        #     "style": torch.rand(1, 320_000),
        #     "beat": Beat(beats, downbeats, seq_len)
        # }
        # return output

        song_name = self.song_iterator[idx]
        song_path: Path = self.root_dir / song_name

        # toss a coin to decide whether to augment data
        apply_speed_transform: bool = random.random() < self.speed_transform_p
        apply_pitch_transform: bool = random.random() < self.pitch_transform_p

        # choose random augmentation factors if needed
        speed_factor = round(random.random() * 0.4 +
                             0.80, 2) if apply_speed_transform else 1
        pitch_factor = (random.randint(-4, 4) if apply_pitch_transform else 0)

        # get song features
        with open(song_path / "features.json", "r") as f:
            features: Dict[str, str | int] = json.load(f)
        song_n_frames: int = int(features["num_frames"])
        song_sr: int = int(features["sample_rate"])
        n_frames_to_take_orig: int = int(self.chunk_size_samples /
                                         self.sample_rate * song_sr)
        n_frames_to_take = int(n_frames_to_take_orig / speed_factor)

        # get all song stems, including possibly silent ones
        song_stems: List[Stem] = []
        for stem in self.stems:
            if (song_path / stem.getname()).exists():
                song_stems.append(stem)

        if self.target_stem != Stem.ANY and self.target_stem not in song_stems:
            raise RuntimeError(f"Target stem is {self.target_stem} but song "
                               f"{song_name} has no interesting stems. "
                               "Maybe remove it from the dataset?")

        if len(song_stems) == 0:
            raise RuntimeError(f"Song {song_name} has no interesting stems. "
                               "Maybe remove it from the dataset?")
        if len(song_stems) == 1:
            raise RuntimeError(f"Song {song_name} has only one stem, which is"
                               f"{song_stems[0]}. What to do?")

        # find a good chunk of the song
        stem_tensors: Dict[Stem, Tensor]
        start_offset: int
        stem_tensors, start_offset = self.find_good_chunk(
            song_name, n_frames_to_take, song_n_frames, song_path, song_stems)
        nonsilent_stems: List[Stem] = list(stem_tensors.keys())

        # split stems between input and conditioning
        input_stems, condition_stems = self.choose_input_and_conditioning(
            nonsilent_stems)

        # mix input and condition tensors
        input_tensor, condition_tensor = self.mix_input_and_conditioning(
            stem_tensors, input_stems, condition_stems)

        # if applying style conditioning, find a good style conditioning chunk
        style_tensor: Optional[Tensor] = None
        if self.use_style_conditioning:
            assert self.single_stem
            assert len(input_stems) == 1
            inputstem = input_stems[0]
            style_tensor = self.find_good_chunk(song_name,
                                                n_frames_to_take_orig,
                                                song_n_frames, song_path,
                                                [inputstem])[0][inputstem]

        # if using beat conditioning, compute beats data for current chunk
        beats_conditioning: Optional[Beat] = None
        if self.use_beat_conditioning:
            beatfile = song_path / "beatthis.npz"
            if not beatfile.exists():
                raise FileNotFoundError(
                    f"Couldn't find beat annotations for song {song_path}")
            loaded = np.load(beatfile)
            beats_sec = torch.from_numpy(loaded["beats"])
            downbeats_sec = torch.from_numpy(loaded["downbeats"])

            # if using a speed augmentation, reposition beats
            beats_frames: Tensor = (beats_sec * speed_factor *
                                    self.sample_rate).round().long()
            downbeats_frames: Tensor = (downbeats_sec * speed_factor *
                                        self.sample_rate).round().long()
            start_offset = round(start_offset / song_sr * self.sample_rate *
                                 speed_factor)
            min_max_beat: Tensor = torch.tensor(
                [start_offset, start_offset + self.chunk_size_samples])

            beats_start_idx, beats_end_idx = torch.searchsorted(beats_frames,
                                                                min_max_beat,
                                                                right=False)
            beats_cut = beats_frames[beats_start_idx:beats_end_idx]

            downbeats_start_idx, downbeats_end_idx = torch.searchsorted(
                downbeats_frames, min_max_beat, right=False)
            downbeats_cut = downbeats_frames[
                downbeats_start_idx:downbeats_end_idx]

            assert (downbeats_end_idx + 1 >= len(downbeats_frames) or
                    downbeats_frames[downbeats_end_idx + 1]
                    >= start_offset + self.chunk_size_samples)
            assert (beats_end_idx + 1 >= len(beats_frames) or
                    beats_frames[beats_end_idx + 1]
                    >= start_offset + self.chunk_size_samples)

            beats_cut -= start_offset
            downbeats_cut -= start_offset

            beats_conditioning = Beat(beats_cut, downbeats_cut,
                                      self.chunk_size_samples)

            match self.type_of_context:
                case "beats":
                    beats_as_context = True
                    mix_context = False
                case "stems":
                    beats_as_context = False
                    mix_context = False
                case "stems or beats":
                    match random.randint(0, 2):
                        case 0:  # only beats
                            beats_as_context = True
                            mix_context = False
                        case 1:  # only stems
                            beats_as_context = False
                            mix_context = False
                        case 2:  # stems with beats
                            beats_as_context = False
                            mix_context = True
                case "stems and beats":
                    beats_as_context = False
                    mix_context = True

            beats_time = beats_cut.numpy() / self.sample_rate
            downbeats_time = downbeats_cut.numpy() / self.sample_rate
            clicks = librosa.clicks(times=beats_time,
                                    sr=self.sample_rate,
                                    click_freq=1000,
                                    length=self.chunk_size_samples)
            downbeat_clicks = librosa.clicks(times=downbeats_time,
                                             sr=self.sample_rate,
                                             click_freq=2000,
                                             length=self.chunk_size_samples)
            # Combine clicks (downbeats are stronger)
            audio_clicks = (clicks + downbeat_clicks) * 0.08
            audio_clicks = np.clip(audio_clicks, -1.0, 1.0)
            # mono audio
            if audio_clicks.ndim > 1:
                audio_clicks = audio_clicks.mean(axis=0)
            context_beats = torch.tensor(audio_clicks)
            context_beats = torch.unsqueeze(context_beats, 0)
            if beats_as_context:
                condition_tensor = context_beats

        # add click
        if self.add_click:
            raise NotImplementedError(
                "Add click is not implemented for new dataset with style")
            click_frames: List[int] = self.syncdata[song_name]
            input_tensor = self.add_click_to_track(input_tensor, song_sr,
                                                   click_frames, start_offset)
            if condition_tensor is not None:
                condition_tensor = self.add_click_to_track(
                    condition_tensor, song_sr, click_frames, start_offset)

        # get description of song input
        description: str = self.get_description(features, input_stems,
                                                speed_factor)

        # resample input and conditioning to desired sample rate
        input_tensor = torchaudio.functional.resample(input_tensor, song_sr,
                                                      self.sample_rate)
        if condition_tensor is not None and not beats_as_context:
            condition_tensor = torchaudio.functional.resample(
                condition_tensor, song_sr, self.sample_rate)

        if style_tensor is not None:
            style_tensor = torchaudio.functional.resample(
                style_tensor, song_sr, self.sample_rate)

        # data augmentation to input and conditioning
        if apply_speed_transform or apply_pitch_transform:
            input_tensor = self._transform_chunk(input_tensor, speed_factor,
                                                 pitch_factor,
                                                 self.chunk_size_samples)
            if condition_tensor is not None and not beats_as_context:
                condition_tensor = self._transform_chunk(
                    condition_tensor, speed_factor, pitch_factor,
                    self.chunk_size_samples)

        if mix_context:
            condition_tensor = condition_tensor + context_beats

        if condition_tensor is not None:
            # cut conditioning to a random length
            min_context_samples = self.min_context_seconds * self.sample_rate
            if min_context_samples < self.chunk_size_samples:
                if torch.rand(1).item() > 0.95:
                    index = self.chunk_size_samples
                else:
                    index = torch.randint(min_context_samples,
                                          self.chunk_size_samples + 1,
                                          (1,)).item()
                condition_tensor = condition_tensor[..., :index]

        output = {
            "name": song_name,
            "target": input_tensor,
            "description": description,
            "context": condition_tensor,
        }

        if self.use_beat_conditioning:
            output["beat_seconds"] = beats_time

        if style_tensor is not None:
            output["style"] = style_tensor

        if beats_conditioning is not None:
            output["beat"] = beats_conditioning

        return output


# def generate_sync_data(
#         output_path: Path) -> Dict[str, Dict[str, float | List[int]]]:
#     db = MoisesDB(data_path=str(cfg.DATA_DIR / "moisesdb"),
#                   sample_rate=32_000)
#     if not output_path.exists():
#         raise FileNotFoundError("output path doesn't seem to exist.")
#     sync_path = output_path / "sync.json"
#     if sync_path.exists():
#         raise FileExistsError()
#     data = {}
#     errors = 0
#     for song in tqdm(db, total=len(db)): # type: ignore
#         try:
#             songid = song.id
#             sr = song.sr
#             audio = librosa.to_mono(song.audio)
#             # utils.save_audio(audio)
#             tempo, beats = librosa.beat.beat_track(y=audio,
#                                                    sr=sr,
#                                                    units="samples")
#             beats = beats.tolist()
#             data[songid] = {
#                 "tempo": tempo,
#                 "beats": beats,
#             }
#         except:
#             errors += 1
#     with open(sync_path, "w") as fp:
#         json.dump(data, fp)
#     print(f"Saved sync data of {len(db) - errors} songs, with {errors} errors.")
#     return data


def prepare_data(root_dir: Path, save_mixed_drums: bool, save_mix: bool,
                 extract_features: bool, track_bpm: bool):
    from tqdm import tqdm
    import torchaudio
    from torch import Tensor
    from lag.data.auto_labelling import get_audio_features

    subdirs: List[Path] = sorted([p for p in root_dir.iterdir() if p.is_dir()],
                                 key=lambda x: x.name)
    # assert len(subdirs) == EXPECTED_N_SONGS

    if track_bpm:
        syncdata: Dict[str, List[int]] = {}

    for song in tqdm(subdirs):

        for stemdir in (p for p in song.iterdir() if p.is_dir()):
            if len(list(stemdir.iterdir())) == 0:
                raise FileNotFoundError(f"Song {song} contains no stems. WTF")

        # mix drums
        if (song / "drums").exists():
            drums_sample_rates: Set[int] = set()
            drums_audios: List[Tensor] = []
            for drum_stem in (song / "drums").iterdir():
                audio, sr = torchaudio.load(str(drum_stem))
                drums_sample_rates.add(sr)
                drums_audios.append(audio.permute(1, 0))

            if len(drums_sample_rates) != 1:
                raise ValueError(f"song {song} contains drums stems of "
                                 "different sample rates")
            drums_sr = drums_sample_rates.pop()

            # mix drums
            drums_tensor = torch.nn.utils.rnn.pad_sequence(
                drums_audios, batch_first=True,
                padding_value=0).permute(0, 2, 1).sum(dim=0)

            assert drums_tensor.ndim == 2

            if save_mixed_drums:
                target_dir = song / "drums_mixed"
                target_dir.mkdir(exist_ok=True)
                torchaudio.save(target_dir / "drums.wav", drums_tensor,
                                drums_sr)

        # load mixed song
        stem_subdirs = [p for p in song.iterdir() if p.is_dir()]

        # for each stem
        stem_tracks: List[Tensor] = []
        sample_rates: Set[int] = set()
        for stem_subdir in stem_subdirs:
            if stem_subdir.name == "drums":
                continue

            # for each track of that stem
            for audio_path in stem_subdir.iterdir():
                audio, sr = torchaudio.load(str(audio_path))
                assert audio.ndim == 2
                stem_tracks.append(audio.permute(1, 0))

                sample_rates.add(sr)
                if len(sample_rates) > 1:
                    raise ValueError(f"song {song} contains stems of "
                                     "different sample rates")
        sr = sample_rates.pop()

        # pad shorter tracks
        stems_tensor: Tensor = torch.nn.utils.rnn.pad_sequence(
            stem_tracks,
            batch_first=True,
            padding_value=0.,
        ).permute(0, 2, 1)
        assert stems_tensor.ndim == 3

        # mix song
        mixed_tensor = stems_tensor.sum(dim=0)
        num_frames = mixed_tensor.shape[-1]
        assert mixed_tensor.ndim == 2

        if save_mix:
            mix_out_path = song / "mixed.wav"
            audio_utils.save_audio(mixed_tensor, mix_out_path, sr)

        # neural classification to get metadata
        if extract_features:
            features = get_audio_features(mixed_tensor, sr, cfg.weights_dir())

            features["sample_rate"] = sr
            features["num_frames"] = num_frames
            out_file = song / "features.json"
            with open(out_file, "w") as f:
                json.dump(features, f)

        # bpm tracking
        if track_bpm:
            song_numpy = audio_utils.to_mono(mixed_tensor).squeeze().numpy()

            # audio = librosa.to_mono(song.audio)
            # utils.save_audio(audio)
            try:
                tempo, beats = librosa.beat.beat_track(y=song_numpy,
                                                       sr=sr,
                                                       units="samples")
            except Exception as e:
                print(f"Error tracking beats of song {song.name}")
                _waitforme = True
                raise e

            beats = beats.tolist()

            syncdata[song.name] = beats  # type: ignore

    if track_bpm:
        sync_path: Path = root_dir / "sync.json"
        with open(sync_path, "w") as f:
            json.dump(syncdata, f)  # type: ignore


if __name__ == "__main__":
    from lag import config as cfg
    from tqdm import tqdm

    # root_dir = Path("/home/tkol/dev/datasets") / "moisesdb" / "moisesdb_v0.1"
    # root_dir = Path("/home/tkol/dev/datasets") / "moisesdb" / "musdb"
    root_dir = cfg.moises_path()

    # prepare_data(root_dir,
    #              save_mixed_drums=False,
    #              save_mix=False,
    #              extract_features=True,
    #              track_bpm=False)

    stems = {
        Stem.DRUMS, Stem.GUITAR, Stem.BASS, Stem.PIANO, Stem.KEYBOARD,
        Stem.STRINGS
    }

    import lightning as L
    L.seed_everything(48)
    dataset = StemmedDataset(
        root_dir,
        stems,
        target_stem=Stem.DRUMS,
        single_stem=True,
        min_context_seconds=5,
        use_style_conditioning=True,
        use_beat_conditioning=True,
        add_click=False,
        bpm_in_caption=False,
        sync_chunks=False,
        train=False,
        sample_rate=32_000,
        chunk_size_samples=32_000 * 10,
        speed_transform_p=1,
        pitch_transform_p=1,
        stereo=False,
        n_samples_per_epoch=None,
        type_of_context="stems or beats",
    )

    sample = next(iter(dataset))

    from lag.utils.audio import play

    play(sample["context"], 32_000)

    # dataset_iterator = iter(dataset)
    # for i in tqdm(range(10)):
    #     sample = next(dataset_iterator)

    #     target: Tensor = sample["target"]  # type: ignore
    #     context: Tensor = sample["context"] if sample[
    #         "context"] is not None else sample["context"]

    #     audio_utils.save_audio(target,
    #                            cfg.AUDIO_DIR / "temp" / f"target{i}.wav")
    #     audio_utils.save_audio(context,
    #                            cfg.AUDIO_DIR / "temp" / f"context{i}.wav")
    #     mix = target + torch.nn.functional.pad(
    #         context, (0, target.shape[-1] - context.shape[-1]))
    #     audio_utils.save_audio(mix, cfg.AUDIO_DIR / "temp" / f"mix{i}.wav")
