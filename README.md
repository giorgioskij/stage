# Stage - Single Stem Accompaniment Generation
[![ArXiv](https://img.shields.io/badge/arXiv-2504.05690-b31b1b?logo=arxiv&labelColor=gray)](https://arxiv.org/abs/2504.05690)
[![demo](https://img.shields.io/badge/Demo-Samples-1997B5?labelColor=gray)](https://https://giorgioskij.github.io/stage-demo/)


### Disclaimer:
This is very much **research-oriented** code, it's not by any means production-ready. You'll see configurations for many failed experiments, and some seemingly overcomplicated structures that were necessary for our testing and experimentation workflow. 

If you are interested in making this code more usable, feel free to contribute or to ask questions! <br>Research is only beautiful when it's shared.


# Prerequisites
To run code in this repo, you need to:

- ## Set up the environment:
    This repo's environment is managed by [uv](https://docs.astral.sh/uv/).

    Setting everything up should be as easy as:
    - cloning the repo
    - `cd`-ing into it
    - running `uv sync`

- ## Download the weights for the pre-trained components of our models
    For ease of use, we entirely rewrote `MusicGen`'s architecture, so you'll need to download pre-trained weights that are compatible with our model.

    - Download the weights [here (coming soon)]() and place them in the `weights/` directory of this repo. You can place them anywhere else if you'd like, but modify `src/stage/config.py` accordingly if you do so.

    - If you want to run inference on our trained models, download the fine-tuned checkpoints [here (coming soon)]() and place them inside the `checkpoints/` directory.


# Inference

Follow the example in `src/stage/inference.py` to test inference with any model.


# Data:

The full fine-tuning, validation, and testing is performed on splits of the MoisesDB dataset.

The dataset should be kept in any of the folders listed in `stage.config.moises_path()`, such as `stage/datasets/moisesdb`.

Data preprocessing is needed to train the model. For each song:
- all the drums tracks should be mixed into a single track;
- a `features.json` file should be generated containing features extracted with *essentia*.
This can be done with the function `stage.data.StemmedDataset.prepare_data()`

The structure of the dataset directory should look something like:
```
moisesdb
| 014f3712-293b-42af-9f29-0ed1785be792 
    | features.json
    | bass
    |   | 47c825c0-1c9d-46ec-902c-0037fa45ec54.wav
    | drums_mixed
    |   | drums.wav
    | guitar/
    | ...
| ...
```

# Training: 

The model can be trained using the training script in `src/stage/train_stage_drums.py`
Here you can set all the hyperparameters for both the dataset and the model.

Runs are by default logged to Weights And Biases. Make sure to either:
- set your WandB entity/project name in `stage/config.py`; or
- set `log=False` in the `train(...)` function call
