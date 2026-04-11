import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import torch
from seqeval.metrics import classification_report
from collections import defaultdict
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)

TRAIN_FILE = os.path.join("DATA", "df_train.csv")
TEST_FILE = os.path.join("DATA", "df_test.csv")

FIG_FILE = os.path.join("assets", "04_loss_f1_curves.png")

IGNORED_LABEL_ID = -100

LABEL_LIST = ["O", "B-Artist", "I-Artist", "B-WoA", "I-WoA"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

MODEL_NAME = "bert-base-cased"
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128

SAVE_MODEL_FILE = os.path.join("models", MODEL_NAME)


def normalize_token(token):
    token = token.replace("’", "'").replace("‘", "'")
    token = token.replace("“", '"').replace("”", '"')

    excluded_chars = set(string.punctuation)
    token = ''.join(ch for ch in token if ch not in excluded_chars)
    token = token.lower()

    return token


def group_entities_by_query(df):
    query_map = defaultdict(list)

    for _, row in df.iterrows():
        query = row["query"]
        entity = row["named_entity"]
        ent_type = row["type"]

        if pd.isna(query):
            continue

        if pd.isna(entity):
            query_map[query] = []
        else:
            query_map[query].append((entity.strip(), ent_type.strip()))

    return dict(query_map)


def tag_queries(query_entity_map):
    result = {}

    for query, entities in query_entity_map.items():
        tokens = query.strip().split()
        normalized_tokens = [normalize_token(t) for t in tokens]
        labels = ["O"] * len(tokens)

        normalized_entities = [([normalize_token(w)
                                 for w in ent.split()], ent_type)
                               for ent, ent_type in entities]

        for ent_words, ent_type in normalized_entities:
            for i in range(len(tokens) - len(ent_words) + 1):
                window = normalized_tokens[i:i + len(ent_words)]
                if window == ent_words:
                    labels[i] = f"B-{ent_type}"
                    for j in range(1, len(ent_words)):
                        labels[i + j] = f"I-{ent_type}"
                    break

        result[query] = (tokens, labels)

    return result


def prepare_dataset(tokenizer, tagged_data):
    data = {"tokens": [], "labels": []}

    for tokens, labels in tagged_data.values():
        data["tokens"].append(tokens)
        data["labels"].append([LABEL_TO_ID[label] for label in labels])

    ds = Dataset.from_dict(data)

    def tokenize_and_align_labels(example):
        tokenized = tokenizer(example["tokens"],
                              is_split_into_words=True,
                              truncation=True,
                              padding="max_length",
                              max_length=MAX_LENGTH)
        word_ids = tokenized.word_ids()
        new_labels = []

        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                new_labels.append(IGNORED_LABEL_ID)
            elif word_idx != prev_word_idx:
                new_labels.append(example["labels"][word_idx])
            else:
                label = example["labels"][word_idx]
                if label == LABEL_TO_ID["O"]:
                    new_labels.append(label)
                else:
                    new_labels.append(label + 1 if label % 2 == 1 else label)
            prev_word_idx = word_idx

        tokenized["labels"] = new_labels
        return tokenized

    return ds.map(tokenize_and_align_labels, batched=False)


def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)

    y_true, y_pred = [], []

    for pred_seq, label_seq in zip(preds, labels):
        true_tags, pred_tags = [], []
        for p_id, l_id in zip(pred_seq, label_seq):
            if l_id != IGNORED_LABEL_ID:
                true_tags.append(ID_TO_LABEL[l_id])
                pred_tags.append(ID_TO_LABEL[p_id])
        y_true.append(true_tags)
        y_pred.append(pred_tags)

    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "artist_precision": report.get("Artist", {}).get("precision", 0.0),
        "artist_recall": report.get("Artist", {}).get("recall", 0.0),
        "artist_f1": report.get("Artist", {}).get("f1-score", 0.0),
        "woa_precision": report.get("WoA", {}).get("precision", 0.0),
        "woa_recall": report.get("WoA", {}).get("recall", 0.0),
        "woa_f1": report.get("WoA", {}).get("f1-score", 0.0),
    }


def plot_training_curves(trainer):
    logs = trainer.state.log_history

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    artist_f1, woa_f1 = [], []

    for log in logs:
        if "loss" in log and "epoch" in log:
            train_steps.append(log["epoch"])
            train_loss.append(log["loss"])
        if "eval_loss" in log and "epoch" in log:
            eval_steps.append(log["epoch"])
            eval_loss.append(log["eval_loss"])
        if "eval_artist_f1" in log and "eval_woa_f1" in log and "epoch" in log:
            artist_f1.append(log["eval_artist_f1"])
            woa_f1.append(log["eval_woa_f1"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(train_steps, train_loss, label="Training Loss", color="blue")
    ax1.plot(eval_steps, eval_loss, label="Validation Loss", color="orange")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(eval_steps, artist_f1, label="Artist F1", color="green")
    ax2.plot(eval_steps, woa_f1, label="WoA F1", color="purple")
    ax2.set_title("F1 Scores by Class")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("Training Progress")
    plt.tight_layout()
    plt.savefig(FIG_FILE)
    plt.show()


def predict_entities(query, model, tokenizer):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokens = query.strip().split()

    inputs = tokenizer(tokens,
                       is_split_into_words=True,
                       return_tensors="pt",
                       truncation=True,
                       padding="max_length",
                       max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).squeeze().tolist()

    word_ids = tokenizer(tokens, is_split_into_words=True).word_ids()
    labels = []
    previous_word_idx = None

    for pred_id, word_idx in zip(preds, word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        labels.append(ID_TO_LABEL[pred_id])
        previous_word_idx = word_idx

    entities = []
    current_entity = []
    current_type = None

    for token, label in zip(tokens, labels):
        if label == "O":
            if current_entity:
                entities.append({
                    "named_entity": " ".join(current_entity),
                    "type": current_type
                })
                current_entity = []
                current_type = None
        elif label.startswith("B-"):
            if current_entity:
                entities.append({
                    "named_entity": " ".join(current_entity),
                    "type": current_type
                })
            current_entity = [token]
            current_type = label[2:]
        elif label.startswith("I-") and current_type == label[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append({
                    "named_entity": " ".join(current_entity),
                    "type": current_type
                })
            current_entity = []
            current_type = None

    if current_entity:
        entities.append({
            "named_entity": " ".join(current_entity),
            "type": current_type
        })

    return entities


def main():
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_map = tag_queries(group_entities_by_query(df_train))
    test_map = tag_queries(group_entities_by_query(df_test))

    train_dataset = prepare_dataset(tokenizer, train_map)
    test_dataset = prepare_dataset(tokenizer, test_map)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=test_dataset,
                      processing_class=tokenizer,
                      compute_metrics=compute_metrics)

    trainer.train()
    plot_training_curves(trainer)

    trainer.save_model("models/")

    model.eval()

    unique_queries = df_test["query"].dropna().unique()[:10]

    for query in unique_queries:
        entities = predict_entities(query, model, tokenizer)
        print(f"\nQuery: {query}")
        print("Predicted Entities:")
        for ent in entities:
            print(f"  - {ent['named_entity']} ({ent['type']})")


if __name__ == "__main__":
    main()
