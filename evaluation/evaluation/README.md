### Evaluation scripts
The scripts in this folder are used to (1) compute correlations for the hierarchical prompts and baselines and (2) compute the significance values among them. 
First, the scripts `evaluate.py` and `evaluate_baselines.py` should be used to preprocess the raw score outputs and to compure correlation scores. Then the script `compute_significance.py` will calculate the significance tests between the best hierarchical prompts and baselines for each task/dataset.  For more details, please view the comments in each file.

#### Evaluate hierachical prompts
Go to `evaluate_baselines.py` follow the comments in the main section and specify all the file paths that are currently abbreviated with `PATH_TO_...`. Then run the script. First, raw files will be cleaned, concatenated and scores will be extracted from generated texts. Then json/xlsx files with correlations will be written.

#### Evaluate baselines
Go to `evaluate_baselines.py` follow the comments in the main section and specify all the file paths that are currently abbreviated with `PATH_TO_...`. Then run the script. The result is a json/xlsx file with correlations for every baseline per task.

#### Compute significance tests
Go to `compute_significance.py` follow the comments in the main section and specify all the file paths that are currently abbreviated with `PATH_TO_...`. Then run the script.