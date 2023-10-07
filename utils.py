import os
from typing import Dict, List

import spacy
from spacy.training.example import Example

TEXT_KEY = "name"
TAGS_KEY = "tags"


def load_model(model_name: str, labels: List[str]) -> spacy.Language:
    """Load a model and add custom labels to the NER pipeline"""
    try:
        nlp = spacy.load(model_name)
    except:
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)

    # Set up the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add custom labels
    for label in labels:
        ner.add_label(label)

    return nlp


def check_data(data: List):
    """Check if the data is valid.
    Check if the start and end index of each tag is correct.
    Check if the end index of the previous tag is the same as the start index of the next tag.
    Check if the end index of the last tag is the same as the length of the product.
    """
    errors_in_data = False
    for product in data:
        name = product[TEXT_KEY]
        tags = product[TAGS_KEY]
        # Check if the first tag starts at 0 index
        if not tags[0][1] == 0:
            print(f"Start index not 0 on: {name}")
            errors_in_data = True
            continue

        end = tags[0][2]
        for tag in tags[1:]:
            start = tag[1]
            # Check if the tags are sequential
            try:
                assert start == end + 1
            except:
                print(f"Tags not sequential on: {name}")
                errors_in_data = True
                break
            end = tag[2]
        try:
            assert end == len(product[TEXT_KEY])
        except:
            errors_in_data = True
            print(f"Not all tags labelled on: {name}")
    if errors_in_data:
        raise ValueError("Errors in data")

def load_data(data_folder: str) -> List[Dict[str, List]]:
    """Load data from a folder containing json files"""
    data_files = os.listdir(data_folder)
    data = []
    for file in data_files:
        with open(f"{data_folder}/{file}", "r") as file:
            # Read the contents of the file as a string
            json_str = file.read()
        data.extend(eval(json_str))
    check_data(data)
    return data


def generate_train_examples(model: spacy.Language, data: List) -> List[Example]:
    """Generate train examples in the correct format"""
    examples = []
    for text, annotations in data:
        example = Example.from_dict(model.make_doc(text), annotations)
        examples.append(example)
    return examples
