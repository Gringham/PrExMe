from transformers import AutoTokenizer

def apply_chat_template(model_name: str, prompts, llm=None, print_samples=False):
    # Custom method to apply chat template for different models
    print("Model name: ", model_name)

    if "openorca" in model_name.lower() or "nous-hermes" in model_name.lower():
        text = "### Instruction:\n\n{prompt}\n### Response:\n"
        res = [text.format(prompt=prompt) for prompt in prompts]
    elif "platypus2-70b" in model_name.lower():
        text = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
        res = [text.format(prompt=prompt) for prompt in prompts]
    else:
        print("using default template")
        res = [llm.llm_engine.tokenizer.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                                tokenize=False, add_generation_prompt=True)
         for prompt in prompts]

    if print_samples:
        print("Printing 3 first inputs:")
        for i in range(3):
            print(f"Prompt:\n{res[i]}\n")


    return res
