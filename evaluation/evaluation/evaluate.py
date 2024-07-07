import math
import os.path
import re
from multiprocessing import pool, Pool

import nlpstats.correlations.correlations
import pandas as pd
import scipy
import tqdm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

from data.load_eval_df import load_train_df, load_dev_df, load_test_df
from project_root import join_with_root

from mt_metrics_eval import tau_optimization
from mt_metrics_eval import stats

def read_df(k, v):
    # Read and concatenate dataframes from files in v and add a column with the model tag in k
    print(f"Reading {k}, {v}")
    df = pd.concat([pd.read_json(f) for f in tqdm.tqdm(v)])
    df["model_tag"] = [k] * len(df)
    return df


def filter_score(score, format_prompt):
    # Filters each row for the last valid match of a number or label. For MQM, the MQM parser method is used instead
    found = None
    try:
        if num_there(format_prompt):
            found = re.findall(r"[-+]?(?:\d*\.*\d+)", score)
            return float(found[-1])
        else:
            return re.findall(format_prompt, score.lower())[-1]

    except Exception as e:
        return np.NaN


def num_there(s):
    # Checks if a string contains a number
    return any(i.isdigit() for i in s)


def apply_tie_correction(x, y):
    # Applies the tie correction to the Kendall tau correlation by Deutsch, et.al. (2023), Ties Matter, EMNLP
    # (https://aclanthology.org/2023.emnlp-main.798/)
    # (https://github.com/google-research/mt-metrics-eval/blob/main/mt_metrics_eval/ties_matter.ipynb)
    result = tau_optimization.tau_optimization(
        np.array([np.array(x)]), np.array([np.array(y)]), tau_optimization.TauSufficientStats.acc_23, sample_rate=0.1
    )
    accuracy, _ = stats.KendallVariants(
        y, x, variant="acc23", epsilon=result.best_tau
    )
    return accuracy


def save_corr(name=None, p=None, a=None, b=None, no_tie=False, significances = None, ignore_errors=True):
    # Compute correlations with fallback to 0 if an error occurs and ignore_errors is set to True
    # If computed, significance values can be passed through to appear in the final results
    try:
        kendall = scipy.stats.kendalltau(a, b, nan_policy="raise").statistic
    except Exception as e:
        kendall = 0
        if not ignore_errors:
            print(f"Error in {name} with exception: {e}")
    if not no_tie:
        try:
            kendall_tie_corrected = apply_tie_correction(a, b)
        except Exception as e:
            kendall_tie_corrected = 0
            if not ignore_errors:
                print(f"Error in {name} with exception: {e}")
    else:
        kendall_tie_corrected = np.nan
    try:
        pearson = scipy.stats.pearsonr(a, b).statistic
    except Exception as e:
        pearson = 0
        if not ignore_errors:
            print(f"Error in {name} with exception: {e}")
    try:
        spearman = scipy.stats.spearmanr(a, b, nan_policy="raise").statistic
    except Exception as e:
        spearman = 0
        if not ignore_errors:
            print(f"Error in {name} with exception: {e}")

    if p and name:
        result = {
            "name": name,
            "regex": p[0]["format_prompt"],
            "task_description": p[0]["task_description"],
            "task": name[0],
            "prompt": p[0]["base_prompt"]["name"],
            "model": name[1],
            "count": len(a),
            "kendall": kendall,
            "kendall_tie_corrected": kendall_tie_corrected,
            "pearson": pearson,
            "spearman": spearman,
        }
    else:
        result = {
            "kendall": kendall,
            "kendall_tie_corrected": kendall_tie_corrected,
            "pearson": pearson,
            "spearman": spearman
        }

    if significances:
        result["significances"] = significances

    return result


def scores_to_float(row):
    # Translate text labels to scores
    score_dict1 = {"bad": 1, "neutral": 3, "good": 5}
    score_dict2 = {"catastrophic": 1, "indifferent": 3, "marvelous": 5}

    reformatted_scores = []
    for prompt, score in zip(row["prompts"], row["generated_text"]):
        new_score = filter_score(score, prompt["format_prompt"]["regex"])
        new_score = score_dict1[new_score] if new_score in score_dict1 else new_score
        new_score = score_dict2[new_score] if new_score in score_dict2 else new_score
        reformatted_scores.append(new_score)
    return reformatted_scores


def reformat_df(file_paths, outname=None, force=False, outpath=None):
    '''
    Reads a files with raw prompt outputs, concatenates them into a single large dataframe and extracts the scores into a new column.
    File paths should be provided as a dictionary with a tag as key and a list of file paths as value. By doing so, we can keep track of the model tag 
    in a new column.
    Each row is a list of scores, one for each prompt for one sample
    Either specify outname or outpath, not both, as output location
    If the file exists, it will be loaded and returned, otherwise it will be created
    If force is set to True, the file will be overwritten
    '''
    
    if outname and os.path.isfile(join_with_root(f"outputs/cleaned/{outname}.json")) and not force:
        return pd.read_json(join_with_root(f"outputs/cleaned/{outname}.json"))
    if outpath and os.path.isfile(outpath) and not force:
        return pd.read_json(outpath)
    
    # Load a concatenated dataframe with all the results written in file_paths
    df = pd.concat([read_df(k, v) for k, v in file_paths.items()])

    # apply the regex filters for the score and replace the results of text-based approaches
    df["score"] = df.apply(lambda row: scores_to_float(row), axis=1)

    print(f"There are {df['score'].explode().isna().sum()} na values at the moment. They will be replaced with their "
          f"avg")

    # Backup the original scores for later use
    df["score_unfilled"] = df["score"]

    def f(x):
        # Method to replace NaN values with the average of the other scores for the same prompt
        try:
            scores = np.array(x.tolist())
            avgs = np.nanmean(scores, axis=0)
            inds = np.where(np.isnan(scores))
            scores[inds] = np.take(avgs, inds[1])
            return scores.tolist()
        except Exception as e:
            print(e)
            # In case all scores for a prompt are NaN, return a list of 0s
            return [0] * len(x)

    df['score'] = df.groupby(["task", "model_tag"])["score"].transform(f)

    # Save the file that combines all raw scores with the extracted scores
    if outname:
        df.to_json(join_with_root(f"outputs/cleaned/{outname}.json"), orient="records")
    if outpath:
        df.to_json(outpath + ".json", orient="records")

    return df

def compute_correlation(df, outname=None, no_tie=False, outpath=None):
    '''
    Computes the correlations on the provided dataframe
    @param outname: The name of the file the results are written to. Do not specify the extension.
    @param no_tie: As the tie corrected kendall tau is computationally expensive, it can be disabled
    @param outpath: The path to the file the results are written to. Do not specify the extension. 
    @return: Nothing, things get written into evalution/outputs_v2
    '''

    # Compute the correlations for each dimension and write them into a list of dicts
    results = []
    for name, group in tqdm.tqdm(df.groupby(["task", "model_tag"], dropna=False)):
        group = group[group["GT_Score"] != "None"]
        scores = np.array(group["score"].tolist()).T.tolist()

        # Fallback
        scores = [[float(a) if a != None else 0 for a in s] for s in scores]

        # Ensure Formats
        prompts = np.array(group["prompts"].tolist()).T.tolist()
        gt_scores = group["GT_Score"].tolist()
        gt_scores = [float(g) for g in gt_scores]

        # Compute with multiprocessing to ensure evaluation speed
        with Pool(processes = 50) as pool:

            # Create a list of all score - human score combinations that should be evaluated
            items = [(name, p, [float(s1) if s1 else 0 for s1 in s], gt_scores, no_tie) for s, p in zip(scores, prompts)]

            # Parallely compute the correlations
            for idx, result in enumerate(pool.starmap(save_corr, tqdm.tqdm(items, desc="Computing correlations"))):
                results.append(result)
            


        print(results[-1])

    out = pd.DataFrame(results)

    # Save output files as excel and json files
    if outname:
        out.to_excel(join_with_root(f"outputs/evaluation/{outname}.xlsx"))
        out.to_json(join_with_root(f"outputs/evaluation/{outname}.json"))

    if outpath:
        out.to_excel(outpath + ".xlsx")
        out.to_json(outpath + ".json")

def extract_range_and_task(filename):
    range_match = re.search(r'_(\d+)_(\d+)_', filename)
    start_range = int(range_match.group(1)) if range_match else float('inf')
    task = "_".join(filename.split(".json")[0].split("_")[-2:])
    print(start_range, task)
    return start_range, task

def load_dir(dir, keys=None):
    if keys is None:
        keys = {
            "Maziyar": "LLaMA3-70B",
            "TheBloke_Platypus2": "Platypus2-70B",
            "Llama-3-8B": "LLaMA3-8B",
            "Tower": "Tower-13B",
            "Nous": "NousHermes-13B",
            "Open-Orca": "OpenOrca-13B"
        }

    if not os.path.isdir(dir):
        raise ValueError(f"The provided directory '{dir}' does not exist.")

    paths = [os.path.join(dir, d) for d in os.listdir(dir)]
    paths.sort(key=extract_range_and_task)

    path_dict = {v: [] for v in keys.values()}
    for p in paths:
        for k, v in keys.items():
            if k in p:
                path_dict[v].append(p)

    return path_dict

if __name__ == '__main__':
    # 1. Make sure that a directory only contains the raw experiment outputs for one dataset
    # Alternatively, you can manually map files to tags, e.g. {"Tag1": ["file1", "file2", "file5"], "Tag2": ["file3", "file4"], ...}
    # The latter allows to put files from different experiments into one directory. However make sure to use consistent tags to ensure 
    # correct plotting of the results at a later stage

    # 2. Create a dictionary of the model tags and their corresponding file paths
    zero_shot_test = load_dir("<PATH_TO_RAW_OUTPUTS>")

    # 3. Concatenate all raw results, extract the scores from generated text and save+return the dataframe
    # The dataframe will be saved as a json file
    df = reformat_df(zero_shot_test, outpath="<PATH_TO_CLEANED_OUTPUTS>", force=True)

    # 4. Compute the correlations and save the results as an excel and json file
    compute_correlation(df, outpath="<PATH_TO_CORRELATION_TABLES>", no_tie=False)
