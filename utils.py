import os
from typing import Dict, List

import spacy
from spacy.training.example import Example


def load_model(model_name: str, labels: List[str]) -> spacy.Language:
    """Load a model and add custom labels to the NER pipeline"""
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


def check_data(data):
    """Check if the data is in the correct format"""
    error = False
    for idx, elem in enumerate(data):
        tags = elem["tags"]
        # Check if the first tag starts at 0
        assert tags[0][1] == 0
        end = tags[0][2]
        for tag in tags[1:]:
            start = tag[1]
            try:
                # Check if tags are consecutive
                assert start == end + 1
            except:
                print(idx)
            end = tag[2]
        try:
            assert end == len(elem["name"])
        except:
            error = True
            print(end, len(elem["name"]))
            print(idx)
    if error:
        raise Exception("Data in wrong format")


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
    data = parse_data(data)
    return data


def parse_data(data: List) -> List[Dict[str, List]]:
    """Parse the data into the correct format"""
    for elem in data:
        tags = elem["tags"]
        for idx in range(len(tags)):
            if tags[idx][0].startswith("I-"):
                r_iter = 1
                while tags[idx - r_iter][0].startswith("I-"):
                    r_iter += 1
                tags[idx - r_iter][2] = tags[idx][2]
        elem["tags"] = [x for x in elem["tags"] if not x[0].startswith("I-")]
    check_data(data)

    train_data_parsed = []
    for elem in data:
        text = elem["name"]
        dic = {}
        dic["entities"] = [(x[1], x[2], x[0]) for x in elem["tags"]]
        train_data_parsed.append((text, dic))
    return train_data_parsed


def generate_train_examples(model: spacy.Language, data: List) -> List[Example]:
    """Generate train examples in the correct format"""
    examples = []
    for text, annotations in data:
        example = Example.from_dict(model.make_doc(text), annotations)
        examples.append(example)
    return examples
