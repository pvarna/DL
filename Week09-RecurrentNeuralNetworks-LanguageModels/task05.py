import transformers


def main():
    generator = transformers.pipeline(task="text-generation", model="distilgpt2")
    
    review = "I had a wonderful stay at the Riverview Hotel! The staff were incredibly attentive and the amenities were top-notch. The only hiccup was a slight delay in room service, but that didn't overshadow the fantastic experience I had."
    response = "Dear valued customer, I am glad to hear you had a good stay with us."
    
    prompt = f"Customer review:\n{review}\n\nHotel reponse to the customer:\n{response}"
    output = generator(prompt, max_length=100, pad_token_id=generator.tokenizer.eos_token_id)
    print(output[0]["generated_text"])




if __name__ == '__main__':
    main()
