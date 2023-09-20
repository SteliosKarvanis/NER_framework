import random

from configs import TRAIN_DATA_DIR, DROPOUT_RATE, LABELS, MODEL_NAME, MODEL_OUTPUT_DIR, TRAIN_ITER
from utils import generate_train_examples, load_data, load_model

######################################################
####### Set up the configs in configs.py first #######
######################################################

if __name__ == "__main__":
    model = load_model(model_name=MODEL_NAME, labels=LABELS)
    data = load_data(TRAIN_DATA_DIR)
    examples = generate_train_examples(model, data)

    other_pipes = [pipe for pipe in model.pipe_names if pipe != "ner"]
    with model.disable_pipes(*other_pipes):
        optimizer = model.create_optimizer()
        for itn in range(TRAIN_ITER):
            random.shuffle(data)
            losses = {}
            model.update(examples=examples, drop=DROPOUT_RATE, sgd=optimizer, losses=losses)
            print(losses)

    model.to_disk(MODEL_OUTPUT_DIR)
    print(f"Model saved to {MODEL_OUTPUT_DIR}")
