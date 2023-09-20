# NER_framework

## Introduction
This is a framework to train a custom `NER model` with spacy, with an interface to easily label the data and train the model.

## Installation
Install the requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Configs
Set up the configs in the file `config.py`:
1. `LABELS`: In a list format (If some label may have more than one word, create a B-label and a I-label, for the begining and intermediate words, respectively). 
2. `TRAIN_CONFIGS`: The training configs, such as the number of iterations, dropout, etc.

### Data Labeling
To label the data, create a text file with the sentences to be labeled with one sentence by line, then label it with the `labeling tool`, by running the following command:
```bash
python labeling_tool.py --input_file=<input_file> --output_file=<output_file>
```

### Training the model
To train the model, run the following command:
```bash
python train.py
```

### Validating the model
To validate the model, use the `validate.ipynb` notebook

## Example
To see an example, checkout to `example` branch
