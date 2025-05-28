from transformers import pipeline


def main():
    generator = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english")
    input_text = """
    Classify the sentiment of this sentence as either positive or negative.
    Text: "The dinner we had was great and the service too." Sentiment: Positive
    Text: "The food was terrible!" Sentiment: Negative
    Text: "The waiter was very friendly and the food was delicious!" Sentiment:
    """

    result = generator(input_text, max_length=100)
    print(result[0]["label"])


if __name__ == '__main__':
    main()
