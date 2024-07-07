import os

from iterator.utils.chatTemplates import apply_chat_template
from iterator.utils.vllm_startup_helper import vllm_startup_helper

import pandas as pd
import torch

import argparse
from torch.utils.data import Dataset

# Define a class ListDataset that inherits from Dataset
class ListDataset(Dataset):
    """
    A class used to represent a Dataset

    ...

    Attributes
    ----------
    prompt_dataset : list
        a list of prompts that the dataset holds

    Methods
    -------
    __len__(self):
        Returns the length of the dataset
    __getitem__(self, i):
        Returns the i-th item of the dataset
    """

    def __init__(self, p):
        """
        Parameters
        ----------
        p : list
            a list of prompts that the dataset will hold
        """
        self.prompt_dataset = p

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.prompt_dataset)

    def __getitem__(self, i):
        """Returns the i-th item of the dataset"""
        return self.prompt_dataset[i]


if __name__ == '__main__':
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description='Pass allowed models via command line.')
    parser.add_argument('--model', help='List of models to be allowed')
    parser.add_argument('--task', help='the language pair to consider')
    parser.add_argument('--fr', help='Starting sample')
    parser.add_argument('--to', help='Ending sample')
    parser.add_argument('--max_tokens', help='Max tokens')
    parser.add_argument('--prompt_df_path', help='File with the prompts for the current model', required=True)
    parser.add_argument('--out_dir', help='output directory', required=True)
    parser.add_argument('--tag', help='tag for the output file', default="")
    parser.add_argument('--parallel', help='parallel size for the model', default=1)
    parser.add_argument('--vllm_sampling_mode', help='vllm sampling mode', default="greedy")
    parser.add_argument('--hf_home', help='hf home', required=True)

    # Parse the arguments
    args = parser.parse_args()

    # Set environment variables
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['HF_HOME'] = args.hf_home

    # Set the Hugging Face token
    # os.environ["HF_TOKEN"]= TOKEN

    # Load the data from the json file
    raw_df = pd.read_json(args.prompt_df_path)
    raw_samples = len(raw_df)
    raw_df = raw_df[raw_df["task"] == args.task]
    df = raw_df[int(args.fr):int(args.to)]
    del raw_df

    # Unpack and repack the prompts
    prompt_list = df["prompts"].tolist()
    prompt_list_unpacked = [d["base_prompt"]["prompt"] for e in prompt_list for d in e]

    # Initialize the model and generate the responses
    llm, sampling_params = vllm_startup_helper(args.model, max_tokens=args.max_tokens, vllm_sampling_mode=args.vllm_sampling_mode, parallel=args.parallel)
    input = apply_chat_template(args.model, prompt_list_unpacked, llm, print_samples=True)
    outputs = llm.generate(input, sampling_params)

    # Process the results
    processed_results = [m.outputs[0].text for i, m in enumerate(outputs)]
    processed_results += ['MISSING'] * (len(prompt_list_unpacked) - len(processed_results))

    # Reformat the results to match the original dataframe
    new_res = [processed_results[x:x + len(prompt_list[0])] for x in
               range(0, len(processed_results), len(prompt_list[0]))]

    # Add the generated text to the dataframe
    df["generated_text"] = new_res

    # Clear the GPU memory
    torch.cuda.empty_cache()

    # Save the dataframe to a json file
    df.to_json(
        os.path.join(args.out_dir, f"slurm_pool_{args.model.replace('/', '_')}_{args.fr}_{args.to}_of_{raw_samples}"
                                   f"_{args.tag}_{args.task}.json"),
        orient="records", force_ascii=False)
