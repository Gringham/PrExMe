from calendar import c
from codecs import ignore_errors
import csv
import os.path

import pandas as pd
from tqdm import tqdm

from multiprocessing import pool, Pool


from data.load_eval_df import load_train_df, load_dev_df, load_test_df
from evaluation.evaluate import save_corr
from project_root import join_with_root
from os import listdir, path

def pool_method(dataset, task, baseline_path, gt):
    """
    Computes correlations for one baseline output file. It is designed to be used in a parallel pool.

    Args:
        dataset (str): The dataset name, e.g. "train" or "test".
        task (str): The task name, e.g. "en_de" or "en_zh".
        baseline_path (str): The path to the baseline output file.
        gt (list): The list of ground truth scores.

    Returns:
        dict: A dictionary containing the computed correlations, dataset name, task name, and approach name.
    """
    baseline_df = pd.read_json(baseline_path)
    baseline_name = baseline_df.columns[0]
    baseline_scores = baseline_df[baseline_name].fillna(baseline_df[baseline_name].mean()).tolist()

    if dataset == "train":
        # Train samples are limited to 500
        gt = gt[:500]

    assert len(gt) == len(baseline_scores)

    gt_clean = []
    baseline_clean = []
    non_count = 0

    # For WMT 23, some samples have no human GT score and are filtered out
    for g, b in zip(gt, baseline_scores):
        if g != "None":
            gt_clean.append(float(g))
            baseline_clean.append(b)
        else:
            non_count += 1
    print(f"Non-Count: {non_count} for task {task} in dataset {dataset}")

    # Apply the correlation computation and save details on the baseline in the result dictionary
    correlations = save_corr(a=gt_clean, b=baseline_clean, no_tie=False, ignore_errors=False)
    correlations["dataset"] = dataset
    correlations["task"] = task
    correlations["approach"] = baseline_name.split("___")[1]

    # These two baselines include a model name and an approach name
    if "DSBA" in correlations["approach"] or "MQM" in correlations["approach"]:
        correlations["model"] = baseline_name.split("___")[2].split(".")[0]
        correlations["approach"] = baseline_name.split("___")[1].split("___")[0]
    else:
        correlations["model"] = ""

    return correlations

if __name__ == '__main__':
    # Load the datasets for ground truth scores
    datasets = {
        "train": load_train_df(),
        "dev": load_dev_df(),
        "test": load_test_df(),
        "test2": load_test_df("wmt_23_seahorse")
    }

    # Get a list of baseline files
    baseline_path_head = "<PATH_TO_RAW_BASELINE_OUTPUT>"
    baseline_paths = listdir(baseline_path_head) 
    baseline_paths_full = [path.join(baseline_path_head, file) for file in baseline_paths]

    # Create a list of score, human-score tuples to be used in the parallel pool
    items = []
    for dataset, df in datasets.items():
        for task, sub_df in df.groupby("task"):
            for i, baseline in tqdm(enumerate(baseline_paths)):
                if f"{dataset}_" in baseline and task in baseline and not "generated_text" in baseline:
                    items.append((dataset, task, baseline_paths_full[i], sub_df["GT_Score"].to_list()))
    
    # Compute the correlations in parallel
    correlations_list = []
    with Pool(processes = 50) as pool:
        for idx, result in enumerate(pool.starmap(pool_method, tqdm(items, desc="Computing Correlations", total=len(items)))):
            correlations_list.append(result)

    # Concatenate the results and save them into the desired output formats
    baseline_correlations = pd.DataFrame(correlations_list)
    baseline_correlations.to_csv("<PATH_TO_SAVE_CORRELATIONS>.tsv", sep="\t")
    #baseline_correlations.to_json("<PATH_TO_SAVE_CORRELATIONS>.json")
    #baseline_correlations.to_excel("<PATH_TO_SAVE_CORRELATIONS>.xlsx")
