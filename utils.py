import numpy as np
from functools import partial
from typing import Dict, List
from labels import LABELS
import evaluate
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

TEXT_KEY = "name"
TAGS_KEY = "tags"


def check_data(data: List) -> None:
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


def load_data(file: str) -> List[Dict[str, List]]:
    """Load data from a folder containing json files
    Args:
        file: Path to the data file (in the output format of the labelling tool)
    """
    data = []
    with open(file, "r") as file:
        # Read the contents of the file as a string
        json_str = file.read()
    data.extend(eval(json_str))
    data = [x for x in data if len(x[TAGS_KEY]) > 0]
    check_data(data)
    return data


def _create_dict(data_list: List[Dict]) -> Dict:
    """Create a dictionary with the data

    Args:
        data_list: List of dictionaries with the data (raw format)
    """
    ids = [x for x in range(len(data_list))]
    tokens = []
    ner_tags = []
    for elem in data_list:
        tokens.append(elem[TEXT_KEY].split())
        ner_tags.append([LABELS.index(x[0]) for x in elem[TAGS_KEY]])
    return {"id": ids, "ner_tags": ner_tags, "tokens": tokens}


def create_dataset_dict(data, test_size=0.2):
    """Create a dataset from a list of dictionaries

    Args:
        data_list: List of dictionaries with the data (raw format)
    """
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_dict = _create_dict(train_data)
    test_dict = _create_dict(test_data)

    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)
    final_dict = {"train": train_dataset, "test": test_dataset}
    return DatasetDict(final_dict)


def _align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            label_name = LABELS[label]
            # If the label is B-XXX we change it to I-XXX
            label_name = label_name.replace("B-", "I-")
            label = LABELS.index(label_name)
            new_labels.append(label)

    return new_labels


def _tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(_align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def generate_tokenized_datasets(tokenizer, dataset):
    return dataset.map(
        partial(
            _tokenize_and_align_labels,
            tokenizer,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )


def compute_metrics(eval_preds):
    metric = evaluate.load("seqeval")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[LABELS[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [LABELS[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
