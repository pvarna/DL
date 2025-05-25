import transformers
import evaluate


def main():
    bleu = evaluate.load("bleu")
    translator = transformers.pipeline(task="translation_es_to_en",
                                       model="Helsinki-NLP/opus-mt-es-en")

    input_sentence_1 = "Hola, ¿cómo estás?"

    reference_1 = [["Hello, how are you?", "Hi, how are you?"]]

    input_sentences_2 = ["Hola, ¿cómo estás?", "Estoy genial, gracias."]

    references_2 = [["Hello, how are you?", "Hi, how are you?"],
                    ["I'm great, thanks.", "I'm great, thank you."]]

    generated_text_1 = translator(input_sentence_1,
                                  clean_up_tokenization_spaces=True)
    generated_text_1 = generated_text_1[0]["translation_text"]

    metrics_1 = bleu.compute(predictions=[generated_text_1],
                             references=reference_1)
    print(f"Translation for \"input_sentence_1\": {generated_text_1}")
    print(f"Metric for \"input_sentence_1\": {metrics_1}")

    generated_texts_2 = translator(input_sentences_2,
                                   clean_up_tokenization_spaces=True)
    predictions_2 = [
        result["translation_text"] for result in generated_texts_2
    ]

    metrics_2 = bleu.compute(predictions=predictions_2,
                             references=references_2)

    print(f'Translations for "input_sentence_2": {predictions_2}')
    print(f'Metric for "input_sentence_2": {metrics_2}')


if __name__ == '__main__':
    main()
