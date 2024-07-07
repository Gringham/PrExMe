from ast import mod
from math import e
import os, tqdm
from tkinter import E

import pandas as pd
import numpy as np
import nlpstats.correlations.permutation
import re

from data.load_eval_df import load_train_df, load_dev_df, load_test_df
from multiprocessing import pool, Pool
from project_root import join_with_root

def extract_info(text):
    # To save space, our result files only save a prompt. As an assert of the prompt order, 
    # we match back the source and translation to our evaluation datasets
    # This function extracts the source and hypothesis from the provided text

    # Define the regex patterns
    source_pattern = r'Source Text: (.*?) \n'
    hypothesis_pattern = r'Translation: (.*?) \n'
    
    # Find all matches
    source_match = re.search(source_pattern, text, re.DOTALL)
    hypothesis_match = re.search(hypothesis_pattern, text, re.DOTALL)

    # Extract the matched groups
    source = source_match.group(1) if source_match else None
    hypothesis = hypothesis_match.group(1) if hypothesis_match else None

    if not hypothesis:
        hypothesis_pattern = r'Summary: (.*?) \n'
    
        # Find all matches
        hypothesis_match = re.search(hypothesis_pattern, text, re.DOTALL)

        # Extract the matched groups
        hypothesis = hypothesis_match.group(1) if hypothesis_match else None

    if not source or not hypothesis:
        raise Exception(f"Could not extract source or hypothesis from {text}")
    
    return source, hypothesis


def sign(id1, s1, id2, s2, h):
    # Compute the significance that s1 correlates better with ground truth h than s2 and returns the p-value
    # The ids describe the experiment setting for s1 and s2
    # We are using the nlpstats library: https://github.com/danieldeutsch/nlpstats
    if id1 != id2:
        try:
            print("lengths: ", len(s1), len(s2))
            sig = (id1, id2, nlpstats.correlations.permutation.permutation_test(s1, s2,
                                                                  h,
                                                                  "global",
                                                                  "kendall", "both",
                                                                  alternative="greater",
                                                                  n_resamples=9999).pvalue)
        except Exception as e:
            l1 = len(s1.flatten().tolist())
            l2 = len(s2.flatten().tolist())
            l3 = len(h.flatten().tolist())
            print(f"Error in {id1} and {id2}, with lens {l1}, {l2}, {l3}: {e}")
            if l1 != l3 or l2 != l3:
                raise Exception(f"Lengths do not match: {l1}, {l2}, {l3}")
            sig = (id1, id2, 1)
        return sig
    else:
        return (id1, id2, 1)

if __name__ == '__main__':
    # Specify the paths to the cleaned output data. I.e. the results of reformat_df in evaluate.py these files are
    # used to retrieve the scores of the best prompts
    cleaned_files = {
        "train": {"zs": "<PATH_TO_CLEANED_ZS_TRAIN>",
                "fs": "<PATH_TO_CLEANED_FS_TRAIN>"},
        "dev": {"zs": "<PATH_TO_CLEANED_ZS_DEV>",
                "fs": "<PATH_TO_CLEANED_FS_DEV>"
                },
        "test": {"zs": "<PATH_TO_CLEANED_ZS_TEST>",
                "fs": "<PATH_TO_CLEANED_FS_TEST>"
                },
        "test2": {"zs": "<PATH_TO_CLEANED_ZS_TEST2>"}
    }

    # Specify the paths to the files with correlation scores. I.e. the results of compute_correlation in evaluate.py.
    result_files = {
        "train": {"zs": "<PATH_TO_CORRELATIONS_ZS_TRAIN>",
                "fs": "<PATH_TO_CORRELATIONS_FS_TRAIN>"},
        "dev": {"zs": "<PATH_TO_CORRELATIONS_ZS_DEV>",
                "fs": "<PATH_TO_CORRELATIONS_FS_DEV>",
                },
        "test": {"zs": "<PATH_TO_CORRELATIONS_ZS_TEST>",
                "fs": "<PATH_TO_CORRELATIONS_FS_TEST>",
                },
        "test2": {"zs": "<PATH_TO_CORRELATIONS_ZS_TEST2>"}
    }

    # Specify the paths to the baseline outputs.
    baselines = {
        "train" : [os.path.join("<PATH_TO_RAW_BASELINE_OUTPUTS_TRAIN>", file) for file in os.listdir("<PATH_TO_RAW_BASELINE_OUTPUTS_TRAIN>") if "data_train" in file and "generated_text" not in file],
        "dev" : [os.path.join("<PATH_TO_RAW_BASELINE_OUTPUTS_DEV>", file) for file in os.listdir("<PATH_TO_RAW_BASELINE_OUTPUTS_DEV>") if "data_dev" in file and "generated_text" not in file],
        "test" : [os.path.join("<PATH_TO_RAW_BASELINE_OUTPUTS_TEST>", file) for file in os.listdir("<PATH_TO_RAW_BASELINE_OUTPUTS_TEST>") if "data_test_" in file and "generated_text" not in file],
        "test2" : [os.path.join("<PATH_TO_RAW_BASELINE_OUTPUTS_TEST2>", file) for file in os.listdir("<PATH_TO_RAW_BASELINE_OUTPUTS_TEST2>") if "data_test2" in file and "generated_text" not in file]
    }

    for dataset, paths in tqdm.tqdm(baselines.items()):
        # For each dataset, load the specified files
        print("Loading dataframes")

        cleaned_dfs = []
        for key, value in cleaned_files[dataset].items():
            df = pd.read_json(value)
            df["mode"] = key
            cleaned_dfs.append(df)
        if len(cleaned_dfs) == 1:
            cleaned_df = cleaned_dfs[0]
        else:
            cleaned_df = pd.concat(cleaned_dfs, axis=0)

        result_dfs = []
        for key, value in result_files[dataset].items():
            df = pd.read_json(value)
            df["mode"] = key
            result_dfs.append(df)
        if len(result_dfs) == 1:
            result_df = result_dfs[0]
        else:   
            result_df = pd.concat(result_dfs, axis=0)

        significance_matrix = {}
        # iterate over tasks
        for name_outer, group_outer in tqdm.tqdm(result_df.groupby(["task"], dropna=False)):
            task = name_outer[0]
            model_scores = {}

            baseline_dfs = [pd.read_json(file) for file in paths if task in file]
            baselin_dfs_shapes = [df.shape for df in baseline_dfs]

            # iterate over models
            for name, group in tqdm.tqdm(group_outer.groupby(["model"], dropna=False)):
                if len(baseline_dfs) == 1:
                    baseline_df = baseline_dfs[0]
                else:
                    baseline_df = pd.concat(baseline_dfs, axis=1)
                model = name[0]
                print(f"Task: {task}, Model: {model}")

                # find the best prompt for the current model-task pair
                best_prompt = group.loc[group['kendall'].idxmax()].iloc[0].to_dict()

                task_description = best_prompt["task_description"]
                regex = best_prompt["regex"]["name"]
                prompt = best_prompt["prompt"]
                mode = best_prompt["mode"]

                print("Filtering cleaned_df")

                # find the scores of the best prompt
                cleaned_df_group = cleaned_df[(cleaned_df["model_tag"] == model) & (cleaned_df["task"] == task) & (cleaned_df["mode"] == mode)]
                
                baseline_df = baseline_df.reset_index(drop=True)
                cleaned_df_group = cleaned_df_group.reset_index(drop=True)

                if dataset == "train":
                    # Assert that the GT_Scores match between the cleaned_df and original df
                    control = load_train_df()
                    control = control[control["task"] == task]
                    assert sum([abs(f1 - f2) <= 0.0001 for f1, f2 in zip(cleaned_df_group['GT_Score'].to_list(),control['GT_Score'].to_list())]) == len(cleaned_df_group['GT_Score'].to_list())
                    print("Assert passed")

                if dataset == "dev":
                    # Assert that the GT_Scores match between the cleaned_df and original df
                    control = load_dev_df()
                    control = control[control["task"] == task]
                    assert sum([abs(f1 - f2) <= 0.0001 for f1, f2 in zip(cleaned_df_group['GT_Score'].to_list(),control['GT_Score'].to_list())]) == len(cleaned_df_group['GT_Score'].to_list())
                    print("Assert passed")

                if dataset == "test":
                    # Assert that the GT_Scores match between the cleaned_df and original df
                    control = load_test_df("eval4nlp23")
                    control = control[control["task"] == task]
                    assert sum([abs(f1 - f2) <= 0.0001 for f1, f2 in zip(cleaned_df_group['GT_Score'].to_list(),control['GT_Score'].to_list())]) == len(cleaned_df_group['GT_Score'].to_list())
                    print("Assert passed")

                if dataset == "test2":
                    # The GT_Scores do not fully match between the cleaned_df and original df
                    # Here we match them back together based on an id we build from all available information
                    control = load_test_df("wmt_23_seahorse")
                    control = control[control["task"] == task]
                    
                    baseline_df_filtered = baseline_df[control["GT_Score"] != "None"]
                    cleaned_df_group = cleaned_df_group[cleaned_df_group["GT_Score"] != "None"]
                    control = control[control["GT_Score"] != "None"]

                    cleaned_df_group["SRC"] = cleaned_df_group.apply(lambda row: extract_info(row["prompts"][0]["base_prompt"]["prompt"])[0], axis=1)
                    cleaned_df_group["HYP"] = cleaned_df_group.apply(lambda row: extract_info(row["prompts"][0]["base_prompt"]["prompt"])[1], axis=1)
                    cleaned_df_group["SRC"] = cleaned_df_group['SRC'].str.split('\n').str[0].str.strip()
                    cleaned_df_group["HYP"] = cleaned_df_group['HYP'].str.split('\n').str[0].str.strip()
                    control["SRC"] = control["SRC"].str.split('\n').str[0].str.strip()
                    control["HYP"] = control["HYP"].str.split('\n').str[0].str.strip()
                    control["GT_Score_for_id"] = control["GT_Score"].astype(str).str.replace(r'(\.\d{3})\d+', r'\1', regex=True)
                    cleaned_df_group["GT_Score_for_id"] = cleaned_df_group["GT_Score"].astype(str).str.replace(r'(\.\d{3})\d+', r'\1', regex=True)


                    control["sub_id"] = control[['task', 'GT_Score_for_id', 'DOC','system-name', "HYP", "SRC"]].astype(str).agg('-'.join, axis=1)
                    cleaned_df_group["sub_id"] = cleaned_df_group[['task', 'GT_Score_for_id', 'DOC', 'system-name', "HYP", "SRC"]].astype(str).agg('-'.join, axis=1)

                    if(len(control) != len(baseline_df_filtered)):
                        print(f"Warning!: Lengths do not match for {model} and {task}")
                    control = pd.concat([control, baseline_df_filtered], axis=1)
                    control = control.drop_duplicates(subset='sub_id', keep='first')
                    
                    cleaned_df_group = cleaned_df_group.drop_duplicates(subset='sub_id', keep='first')

                    unknown_id = []
                    unknown_id2 = []
                    for a1, b1 in zip(cleaned_df_group["sub_id"].to_list(), control["sub_id"].to_list()):
                        if a1 not in control["sub_id"].to_list():
                            unknown_id.append(a1)

                    for a1, b1 in zip(cleaned_df_group["sub_id"].to_list(), control["sub_id"].to_list()):
                        if b1 not in cleaned_df_group["sub_id"].to_list():
                            unknown_id2.append(b1)

                    cleaned_df_group_backup = cleaned_df_group.copy()
                    cleaned_df_group = pd.merge(control, cleaned_df_group, on="sub_id", suffixes=('_df1', ''), how='left',)
                    baseline_df_filtered = control

                    if pd.isna(cleaned_df_group.iloc[-1]['GT_Score']):
                        # Drop the last row
                        cleaned_df_group = cleaned_df_group[:-1]

                    if pd.isna(baseline_df_filtered.iloc[-1]['GT_Score']):
                        # Drop the last row
                        baseline_df_filtered = baseline_df_filtered[:-1]

                    assert cleaned_df_group['GT_Score'].to_list() == baseline_df_filtered['GT_Score'].to_list()

                else:
                    baseline_df_filtered = baseline_df
                    
                if cleaned_df_group.empty:
                    print(f"Warning!: Empty group for {model} and {task}")
                    continue

                # Create unique ids for every best prompt scores
                for idx, prompt_dict in enumerate(cleaned_df_group["prompts"].to_list()[0]):
                    if prompt_dict["task_description"] == task_description and prompt_dict["format_prompt"]["name"] == regex and prompt_dict["base_prompt"]["name"] == prompt:
                        try:
                            prompt_scores = [scores[idx] for scores in cleaned_df_group["score"].to_list()]
                        except:
                            print(f"Error in {model}, {idx}, {task}, {task_description}, {regex}, {prompt}, {cleaned_df_group['score'].to_list()}")
                            raise Exception
                        id = model + "___" + task_description + "___" + regex + "___" + prompt
                        model_scores[id] = prompt_scores
                        break

            # Build lists of all evaluation settings
            all_names = list(model_scores.keys()) + [column for column in baseline_df_filtered.columns if task in column]
            all_scores = list(model_scores.values()) + [baseline_df_filtered[column].to_list() for column in baseline_df_filtered.columns if task in column]

            all_lens1 = list(set([len(scores) for scores in list(model_scores.values())]))
            all_lens2 = list(set([len(scores) for scores in  [baseline_df_filtered[column].to_list() for column in baseline_df_filtered.columns if task in column]]))

            # for each list in all_scores, replace nan values with the average. This is important for the baselines. 
            for i, scores in enumerate(all_scores):
                all_scores[i] = [np.nanmean(scores) if np.isnan(score) else score for score in scores]

            all_scores = [np.array([np.array(scores, dtype=np.float64)], dtype=np.float64).T for scores in all_scores]

            h = np.array([np.array(cleaned_df_group["GT_Score"].to_list(), dtype=np.float64)], dtype=np.float64).T

            # Create a list of all possible combinations of bestprompts and baselines for parallel computation
            items = [(all_names[i], s, all_names[i2], s2, h)
                            for i2, s2 in enumerate(all_scores)
                            for i, s in enumerate(all_scores)]

            significance_matrix = {p : {} for p in all_names}

            print("Computing Kendall significance")
            # Compute the significance of the correlation between the best prompts and the baselines
            with Pool(processes = 50) as pool:
                for idx, result in enumerate(pool.starmap(sign, tqdm.tqdm(items, desc="Computing Kendall significance", total=len(items)))):
                    significance_matrix[result[0]][result[1]] = result[2]
                    
            significance_df = pd.DataFrame(significance_matrix)
            significance_df.to_json(f"<PATH_TO_SAVE_SIGNIFICANCE_MATRIX_{dataset}_{task}.json")
