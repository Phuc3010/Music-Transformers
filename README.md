# Generating Vietnamese Folk Songs using Transformers
This repository contains code for generating Vietnamese folk songs using a transformer-based deep learning model, trained on MIDI files.
In this project, we use a transformer-based model to generate new Vietnamese folk songs, given a seed melody. We use a corpus of Vietnamese folk song MIDI files for training the model.

# Dataset
We used the Vietnamese Folk Song MIDI Dataset for training and evaluation. This dataset contains over 1000 Vietnamese folk song MIDI files, representing a wide range of styles and regions.

# Requirements
To run this project, you will need the following dependencies:
Python 3.x
PyTorch
NumPy
pretty_midi

Usage
To train the transformer model on the Vietnamese folk song MIDI dataset, run the train.py script. You can modify the hyperparameters of the model in the script's configuration file (config.json). The trained model will be saved to the models directory.

To generate new Vietnamese folk songs using the trained model, run the generate.py script.
