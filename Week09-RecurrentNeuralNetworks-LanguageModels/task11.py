from transformers import GPT2Tokenizer, GPT2LMHeadModel
import evaluate


def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompt = 'Current trends show that by 2030'
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt')

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    output = model.generate(prompt_ids,
                            max_length=20,
                            output_scores=True,
                            return_dict_in_generate=True)

    generated_text = tokenizer.decode(output.sequences[0],
                                      skip_special_tokens=True)
    print(f'Generated text: {generated_text}')

    perplexity = evaluate.load('perplexity', module_type='metric')
    results = perplexity.compute(predictions=[generated_text], model_id='gpt2')
    print(f"Mean perplexity: {results['mean_perplexity']}")

    # Each new token is, on average, as surprising as choosing 1 out of 9.09 possible options


if __name__ == '__main__':
    main()
