import transformers


def main():
    translator = transformers.pipeline(task="translation_es_to_bg",
                                       model="Helsinki-NLP/opus-mt-es-bg")

    spanish_text = "Este curso sobre LLMs se está poniendo muy interesante"
    output = translator(spanish_text, clean_up_tokenization_spaces=True)
    print(output[0]["translation_text"])


if __name__ == '__main__':
    main()
