# NER_framework

## Introduction
This is a framework to train a custom `NER model`, with an interface to easily label the data and train the model.

## Installation
Install the requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Configs
Set up the labels in the file `labels.py`, should have a `O` label for label as empty, and for each entity X, create labels `B-X` and `I-X`, for label as begin and inside label respectively, for multi words entities. 

### Data Labeling
To label the data, create a text file with the sentences to be labeled with one sentence by line, then label it with the `labeling tool`, by running the following command:
```bash
python labeling_tool.py --input_file=<input_file> --output_file=<output_file>
```

### Training the model
To train the model, checkout the `train.ipynb` notebook

## Example
To see an example, checkout to `example_bert` branch
```bash
git checkout example_bert
```

## Spacy Model
To train a spacy model, checkout to `spacy_model` branch
```bash
git checkout spacy_model
```
