import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch

MODEL_NAME = "bert-base-uncased"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                               num_labels=2)

    train_dict = {
        'text': [
            "I'm really upset with the delays on delivering this item. Where is it?",
            "The support I've had on this issue has been terrible and really unhelpful. Why will no one help me?",
            'I have a question about how to use this product. Can you help me?',
            'This product is listed as out of stock. When will it be available again?'
        ],
        'label': [1, 1, 0, 0]
    }

    train_data = Dataset.from_dict(train_dict)

    def tokenize_function(text_data):
        return tokenizer(text_data["text"],
                         return_tensors="pt",
                         padding=True,
                         truncation=True,
                         max_length=64)

    tokenized_train_data = train_data.map(tokenize_function, batched=True)

    training_args = TrainingArguments(output_dir='./finetuned',
                                      eval_strategy='no',
                                      num_train_epochs=3,
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=8,
                                      per_device_eval_batch_size=8,
                                      weight_decay=0.01)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=tokenized_train_data,
                      processing_class=tokenizer)

    trainer.train()

    model.save_pretrained("finetuned-customer-support-model")
    tokenizer.save_pretrained("finetuned-customer-support-model")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    input_text = [
        "I'd just like to say, I love the product! Thank you!",
        "Worst product I've used in my entire life!", "Disgusting..."
    ]
    new_input = tokenizer(input_text,
                          return_tensors="pt",
                          padding=True,
                          truncation=True,
                          max_length=64).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**new_input)

    predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    for i, predicted_label in enumerate(predicted_labels):
        sentiment = label_map[predicted_label]
        print(f"\nInput Text {i + 1}: {input_text[i]}")
        print(f"Predicted Label: {sentiment}")


if __name__ == '__main__':
    main()
