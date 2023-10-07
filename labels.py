#################################################
#################################################
### Add the LABELS before labelling the data ####
#################################################
#################################################

LABELS = [
    "O",
]


def check_labels(labels):
    if len(labels) < 2:
        raise Exception("Add your labels in labels.py file")
    for label in labels:
        if label == "O":
            continue
        if not label.startswith("B-") and not label.startswith("I-"):
            raise Exception("Labels should start with B- or I- or be 'O'")

    entities = list(set([x[2:] for x in labels if x != "O"]))
    for entity in entities:
        if not f"B-{entity}" in labels:
            raise Exception(f"Missing label I-{entity[2:]}")
        if not f"I-{entity}" in labels:
            raise Exception(f"Missing label for entity B-{entity[2:]}")


check_labels(LABELS)
