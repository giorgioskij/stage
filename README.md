# Stage
Code and checkpoints for our paper "STAGE: Stemmed Accompaniment Generation through Prefix-Based Conditioning" will be released soon.


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

The model can be trained using the training script in `src.stage.train_stage_drums.py`
Here you can set all the hyperparameters for both the dataset and the model.

Runs are by default logged to Weights And Biases. Make sure to either:
- set your WandB entity/project name in `stage.config.py`
- set `log=False` in the `train(...)` function call


# Inference



