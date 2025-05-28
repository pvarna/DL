import transformers
from transformers import AutoTokenizer
from datasets import Dataset


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_data = {
        'text': [
            "I'm really upset with the delays on delivering this item. Where is it?",
            "The support I've had on this issue has been terrible and really unhelpful. Why will no one help me?",
            'I have a question about how to use this product. Can you help me?',
            'This product is listed as out of stock. When will it be available again?'
        ],
        'label': [1, 1, 0, 0]
    }

    train_encodings = tokenizer(train_data['text'],
                                padding='max_length',
                                truncation=True,
                                padding=True,
                                max_length=64,
                                return_tensors='pt')

    test_data = {
        'text': [
            'You charged me twice for the one item. I need a refund.',
            'Very good - thank you!'
        ],
        'label': [1, 0]
    }

    print("Tokenized training data:")
    print({k: v for k, v in train_encodings.items()})

    test_rows = []
    for text, label in zip(test_data['text'], test_data['label']):
        encoding = tokenizer(text,
                             padding='max_length',
                             truncation=True,
                             max_length=20)
        encoding['text'] = text
        encoding['label'] = label
        test_rows.append(encoding)

    test_columns = {
        key: [row[key] for row in test_rows]
        for key in test_rows[0]
    }
    test_dataset = Dataset.from_dict(test_columns)

    print("\nTokenized test data:")
    print(test_dataset)


if __name__ == '__main__':
    main()
