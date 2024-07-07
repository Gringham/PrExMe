from project_root import join_with_root

import time

if __name__ == "__main__":
    with open(join_with_root(f"<PATH_TO_COMMAND_SCRIPT>")) as f:
        commands = f.read().split("\n")

    script_head = f'''#!/bin/bash

#SBATCH --job-name=PrExMe
#SBATCH --output=logs/PrExMe_%A_%a.log
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-{len(commands)-1}%50
#SBATCH --export=ALL,PYTHONPATH=<PYTHON_PATH>,HF_HOME=<HF_HOME>

echo "=============================="
echo "Running on $(hostname) at $(date) in $(pwd)"

# Report relevant slurm variables
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_TIME_LIMIT: $SLURM_TIME_LIMIT"
echo "=============================="

<ACTIVATE CONDA>
'''

    distrib = [c.split("--")[1:] for c in commands]

    lon = 0
    for d in distrib:
        if len(d) > lon:
            lon = len(d)

    args = {}
    for d in distrib:
        if len(d) == lon:
            for element in d:
                args[element.split(" ")[0]] = []
            break

    for d in distrib:
        params = {}
        for element in d:
            params[element.split(" ")[0]] = element.split(" ")[1]
        for k,v in args.items():
            if k in params:
                args[k].append(params[k])
            else:
                args[k].append("")


    for k, v in args.items():
        args[k] = [f'"{i}"' for i in v]
        args[k] = "\t\n".join(args[k])
        args[k] = f'{k}=(\n{args[k]}\n)'

    script_head += "\n".join(args.values())

    script_head+='\nsrun python iterator/singleStepIterator.py '

    for k, v in args.items():
        script_head += f'--{k} ${{{k}[$SLURM_ARRAY_TASK_ID]}} '

    # Start the pool runner
    with open(join_with_root(f"slurm_array_p2_baselines.sh"), "w") as f:
        f.write(script_head)
