# *<span style="color:orange">Pr<span style="color:red">Ex</span><span style="color:green">Me</span><span style="color:pink">!*
## *Large Scale Prompt Exploration of Open Source LLMs for Machine Translation and Summarization Evaluation*

[![arXiv](https://img.shields.io/badge/View%20on%20arXiv-B31B1B?logo=arxiv&labelColor=gray)](https://arxiv.org/abs/2406.18528)

This repository contains code for our paper - PrExME!: Large Scale Prompt Exploration of Open Source LLMs for Machine Translation and Summarization Evaluation . This Readme describes the basic usage. More details can be found in the respective folders and files.

### Setup
To setup PrExMe!, please clone this repository and install the requirements: 

```sh
git clone https://github.com/Gringham/PrExMe.git
cd PrExMe
conda create --name PrExMe
conda activate PrExMe
conda install pip

pip install -r requirements.txt
```

### Applying PrExMe! Prompts
To apply the PrExMe! prompts, follow these steps:

1. First, we need to produce the prompts for the dataset that should be evaluated. To do so, we use the scripts in `iterator/prompts`. In `data` you can find datasets from our experiments or you can add your own datasets. As an example, we generate the zero shot prompts for the phase 2 dev set:
    ```python
    # In iterator/prompts/zero_shot.py

    # Load the highest correlating prompts of phase 1
    zero_shot_file = join_with_root("outputs/corr_results/zero_shot_train_sign.json")
    x_best_zero_shot = get_x_best(zero_shot_file, 1)

    # Load the dev set and apply the prompt types.
    # Save generated prompts into a new directory (here "data/prompts")
    prepare_prompts_from_df(load_dev_df(), outname=join_with_root("data/prompts/zero_shot_prompts_dev.json"),
    prompt_combinations=x_best_zero_shot)
    ```

2. Next, we can apply the prompts to let an LLM grade the quality of the generated hypothesis. To do so, we use the `iterator/singleStepIterator.py` cli tool. Create a new directory `outputs/raw/dev` to save the outputs to.
    ```sh
    # Example 1: Apply Orca-Platypus-13B on sample 0 to 12000 (upper bound) of the dev set for en_de
    # The --tag should be unique for each dataset
    python3 iterator/Iterator.py --model Open-Orca/OpenOrca-Platypus2-13B --fr 0 --to 12000 --max_tokens 180 --prompt_df_path data/prompts/zero_shot_prompts_dev.json --tag zs_dev --task en_de --parallel 1 --out_dir outputs/raw/dev --hf_home <PATH_TO_HF_CACHE>

    # Example 2: Apply TheBloke/Platypus2-70B-Instruct-GPTQ on sample 4000 to 8000 (upper bound) of the dev set for summarization. 
    # As this model requires more vram, we set --parallel=2
    # The --tag should be unique for each dataset
    python3 iterator/Iterator.py --model  --fr 4000 --to 8000 --max_tokens 180 --prompt_df_path data/prompts/zero_shot_prompts_dev.json --tag zs_dev --task summarization --parallel 2 --out_dir outputs/raw/dev --hf_home <PATH_TO_HF_CACHE>
    ```

    The resulting file will have a new column `generated_text` that contains the computed metric outputs. During evaluation, we extract scores from this column.

### Evaluation
To evaluate our computed output, we will use the script `evaluation/evaluation/evaluate.py`. Make sure that each dataset has its own output directory for raw files. Files that cover different ranges of a dataset will automatically be concatenated. We continue our example. First create two new directories `outputs/cleaned` and `outputs/example_correlations`. Then follow these steps:

```python
# In evaluation/evaluation/evaluate.py

# Create a dictionary of the model tags and their corresponding file paths
zero_shot_dev = load_dir("outputs/raw/dev")

# Concatenate all raw results, extract the scores from generated text and save+return the dataframe
# The dataframe will be saved as a json file
df = reformat_df(zero_shot_dev, outpath="outputs/cleaned/cleaned_zs_dev", force=True)

# Compute the correlations and save the results as an excel and json file
compute_correlation(df, outpath="outputs/example_correlarions/zs_dev_correlations", no_tie=False)
```

The results will be written to an excel table and json file for further usage.

### Folder Structure and Overview
Many folders of this project contain README files with further instructions. Also, the main section of the code often contains an example. Here is an overview of the folder structure:

```
/baselines             -> Wrappers and scripts to apply baselines and other non-hierarchical approaches
/data                  -> Datasets used in our evaluation
/distributed_execution -> Scripts to create slurm task arrays for distributed execution
/evaluation            -> Scripts to compute correlations, significance tests and to plot the results
/iterator              -> Scripts to build and execute prompts in an efficient manner with vllm
/mt_metrics_eval       -> Selected scripts of mt_metrics_eval for tie calibrated accuracy
/outputs               -> The output directory. 
```

### Citation
If you use the code of PrExMe!, please cite:
```bibtex
@misc{leiter2024prexmelargescaleprompt,
      title={PrExMe! Large Scale Prompt Exploration of Open Source LLMs for Machine Translation and Summarization Evaluation}, 
      author={Christoph Leiter and Steffen Eger},
      year={2024},
      eprint={2406.18528},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18528}, 
}
```
