import matplotlib.pyplot as plt
import pandas as pd
import os
import string
from collections import defaultdict
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

TRAIN_FILE = os.path.join("DATA", "df_train.csv")
TEST_FILE = os.path.join("DATA", "df_test.csv")


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

        result[query] = labels

    return result


# predicts everything as "O"
def stupid_baseliner(query):
    tokens = query.strip().split()
    return ["O"] * len(tokens)


def evaluate_on_test():
    df_test = pd.read_csv(TEST_FILE)

    query_map = group_entities_by_query(df_test)
    true_label_map = tag_queries(query_map)

    pred_label_map = {
        query: stupid_baseliner(query)
        for query in true_label_map
    }

    y_true = list(true_label_map.values())
    y_pred = list(pred_label_map.values())

    print("Evaluation using seqeval (default mode):\n")
    print(classification_report(y_true, y_pred))

    print("Evaluation using seqeval (strict mode):\n")
    print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))


def predict_entities(query, model):
    tokens = query.strip().split()
    labels = model(query)

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
    evaluate_on_test()

    entities = predict_entities(
        "I love Radiohead's Kid A. Something similar? Kinda a bit like Aphex Twin maybe.",
        stupid_baseliner)
    print(entities)


if __name__ == '__main__':
    main()
