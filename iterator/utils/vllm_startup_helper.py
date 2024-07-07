from vllm import LLM, SamplingParams

def vllm_startup_helper(model, vllm_sampling_mode="greedy", max_tokens=180, parallel=1):
    print("Loading vllm with: ", model, vllm_sampling_mode, max_tokens, parallel)
    if vllm_sampling_mode == "greedy":
        sampling_params = SamplingParams(temperature=0, max_tokens=int(max_tokens))
    if "gptq" in model.lower():
        llm = LLM(model=model, quantization="marlin", tensor_parallel_size=int(parallel))
    else:
        llm = LLM(model=model, tensor_parallel_size=int(parallel))
    return llm, sampling_params
