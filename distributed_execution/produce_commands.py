import argparse
import pandas as pd

from project_root import join_with_root

# For iterator
p1_default = "p1_commands_seahorse_lingua.txt"
p2_default = "p2_commands_seahorse_lingua.txt"
program_default = "iterator/Iterator.py"

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Pass command line arguments.')
parser.add_argument('--p1_file', help='Filename for p1', default=p1_default)
parser.add_argument('--p2_file', help='Filename for p2', default=p2_default)
parser.add_argument('--python_prog', help='Python program name', default=program_default)

# Parse the arguments
args = parser.parse_args()

prompt_files = [
    (join_with_root("second_round_prompts/zero_shot_prompts_test2_sh_lingua.json"), "zs_test2_sh_lingua")
    ,]

models = [
    {"name": "Open-Orca/OpenOrca-Platypus2-13B",
     "split": 12000,
     "parallel": 1},
    {"name": "NousResearch/Nous-Hermes-13b",
     "split": 12000,
     "parallel": 1},
    {"name": "Unbabel/TowerInstruct-13B-v0.1",
     "split": 12000,
     "parallel": 1},
    {"name": "meta-llama/Meta-Llama-3-8B-Instruct",
     "split": 12000,
     "parallel": 1},
    {"name": "TheBloke/Platypus2-70B-Instruct-GPTQ",
     "split": 4000,
     "parallel": 2},
    {"name": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
     "split": 4000,
     "parallel": 2},
    {"name": "MaziyarPanahi/Meta-Llama-3-70B-Instruct-GPTQ",
     "split": 4000,
     "parallel": 2},
]

base = ("python3 iterator/Iterator.py --model {model} --fr {fr} --to {to} --max_tokens 180 --prompt_df_path {"
        "prompt_file} --mode vllm-instruct --tag {tag} --task {task} --parallel {parallel} --hf_home <HF_HOME>"
        "--out_dir <OUTDIR>").replace(program_default, args.python_prog)

commands_p1 = ""
commands_p2 = ""
for model in models:
    for prompt_file in prompt_files:
        df = pd.read_json(prompt_file[0])
        tasks = df["task"].unique()
        for task in tasks:
            selected_df = df[df["task"] == task]
            end = 0
            while end < len(selected_df):
                end += model["split"]
            print(len(selected_df), end, model["split"])
            for x in range(0, end, model["split"]):
                line = base.format(model=model["name"], fr=str(x), to=model["split"]+x,
                                             prompt_file=prompt_file[0],
                                      tag=prompt_file[1], task=task, parallel=model["parallel"]) + "\n"
                if model["parallel"] == 1:
                    commands_p1 += line
                else:
                    commands_p2 += line

with open(args.p1_file, "w") as f:
    f.write(commands_p1)

with open(args.p2_file, "w") as f:
    f.write(commands_p2)