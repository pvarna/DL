import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    question = "Who painted the Mona Lisa?"
    context = "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. Considered an archetypal masterpiece of the Italian Renaissance, it has been described as the most known, visited, talked about, and sung about work of art in the world. The painting's novel qualities include the subject's enigmatic expression, the monumentality of the composition, and the subtle modeling of forms."

    extractive_qa = transformers.pipeline("question-answering", model="deepset/roberta-base-squad2")
    output_extractive_qa = extractive_qa(question=question, context=context)
    output_extractive_qa = output_extractive_qa["answer"]

    tokenizer = AutoTokenizer.from_pretrained("consciousAI/question-answering-generative-t5-v1-base-s-q-c")
    generative_qa = AutoModelForSeq2SeqLM.from_pretrained("consciousAI/question-answering-generative-t5-v1-base-s-q-c")

    generative_qa_input_text = f"question: {question} </s> question_context: {context}"
    inputs = tokenizer.encode(generative_qa_input_text, return_tensors="pt", truncation=True)
    output_generative_qa = generative_qa.generate(inputs, max_length=50, num_beams=2, early_stopping=True)
    output_generative_qa = tokenizer.decode(output_generative_qa[0], skip_special_tokens=True)


    print(f"Extractive answer: {output_extractive_qa}")
    print(f"Generative answer: {output_generative_qa}")


if __name__ == '__main__':
    main()
