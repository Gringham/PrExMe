import itertools
import json

import numpy as np
import pandas as pd
import torch
from scipy import spatial
from tqdm import tqdm

from data.load_eval_df import load_train_df, load_dev_df, load_test_df, load_retrieval_df
from iterator.prompts.zero_shot import TASK_DESCRIPTIONS
from project_root import join_with_root

from sentence_transformers import SentenceTransformer

def easy_token_sum(x):
    if x <= 0.333:
        return "bad"
    elif x >= 0.666:
        return "good"
    else:
        return "neutral"
def comp_token_sum(x):
    if x <= 0.333:
        return "catastrophic"
    elif x >= 0.666:
        return "marvelous"
    else:
        return "indifferent"

FORMAT_PROMPTS = [
    # Normalizations are used to normalize the scores in order to present them as examples
    {'name': '0 or 1',
     'regex': '(?=.*\d)\d{0,2}(?:\.\d{0,2})?',
     'format_prompt': 'Return a discrete score of 0 if the {result_type} has flaws and 1 if it is perfect.',
     'normalization_sum': lambda x: round(x, ndigits=0),
     'normalization_mt': lambda x: round((x+25)/25, ndigits=0)},
    {'name': '-1 or 0 or 1',
     'regex': '-?(?=.*\d)\d{0,2}(?:\.\d{0,2})?',
     'format_prompt': 'Return a discrete score of -1 if the {result_type} has flaws, 0 if you are indecisive and 1 if '
                      'it is perfect.',
     'normalization_sum': lambda x: round((x*2)-1, ndigits=0),
     'normalization_mt': lambda x: round((((x+25)/25)*2)-1, ndigits=0)
     },
    {'name': '0 to 5',
     'regex': '(?=.*\d)\d{0,2}(?:\.\d{0,2})?',
     'format_prompt': 'Return a score on a scale from 0 to 5 where 0 indicates that the {result_type} is very bad and 5 is assigned to a perfect {result_type}.',
     'normalization_sum': lambda x: x*5,
     'normalization_mt': lambda x: ((x+25)/25)*5
     },
    {'name': '-5 to 5',
     'regex': '-?(?=.*\d)\d{0,2}(?:\.\d{0,2})?',
     'format_prompt': 'Return a score on a scale from -5 to 5 where 0 indicates that the {result_type} is very bad and 5 is assigned to a perfect {result_type}.',
     'normalization_sum': lambda x: (x*10)-5,
     'normalization_mt': lambda x:(((x+25)/25)*10)-5
     },
    {'name': '0 to 100',
     'regex': '(?=.*\d)\d{0,2}(?:\.\d{0,2})?|100',
     'format_prompt': 'Return a score on a scale from 0 to 100 where 0 indicates that the {result_type} is very bad and 100 is assigned to a perfect {result_type}.',
     'normalization_sum': lambda x: x*100,
     'normalization_mt': lambda x: ((x+25)/25)*100
     },
    {'name': '-100 to 100',
     'regex': '-?(?=.*\d)\d{0,2}(?:\.\d{0,2})?|100',
     'format_prompt': 'Return a score on a scale from -100 to 100 where -100 indicates that the {result_type} is very bad and 100 is assigned to a perfect {result_type}.',
     'normalization_sum': lambda x: (x*200)-100,
     'normalization_mt': lambda x: (((x+25)/25)*200)-100
     },
    {'name': '0.0 to 1.0',
     'regex': '(?=.*\d)\d{0,2}(?:\.\d{0,2})?',
     'format_prompt': 'Return a score on a scale from 0.0 to 1.0 where 0.0 indicates that the {result_type} is very bad and 1.0 is assigned to a perfect {result_type}.',
     'normalization_sum': lambda x: x,
     'normalization_mt': lambda x: (x+25)/25
     },
    {'name': '-1.0 to 1.0',
     'regex': '(?=.*\d)\d{0,2}(?:\.\d{0,2})?',
     'format_prompt': 'Return a score on a scale from -1.0 to 1.0 where -1.0 indicates that the {result_type} is very bad and 1.0 is assigned to a perfect {result_type}.',
     'normalization_sum': lambda x: (x*2)-1,
     'normalization_mt': lambda x: ((x+25)/25)*2-1
     },
    {'name': 'easy token labels',
     'regex': '(bad|neutral|good)',
     'format_prompt': 'Choose, whether the {result_type} is either "bad", "neutral" or "good".',
     'normalization_sum': easy_token_sum,
     'normalization_mt': lambda x: easy_token_sum((x+25)/25)
     },
    {'name': 'complex token labels',
     'regex': '(catastrophic|indifferent|marvelous)',
     'format_prompt': 'Choose, whether the {result_type} is either "catastrophic", "indifferent" or "marvelous".',
     'normalization_sum': comp_token_sum,
     'normalization_mt': lambda x: comp_token_sum((x+25)/25)},
]

BASE_PROMPTS = [
    {"name": "Zero-Shot",
     "prompt": "{task_description} \nHere is an example:\nSource Text: {ex1_src} \n{result_type}: {ex1_hyp}\nScore: {"
               "ex1_score}\n\nNow it is your turn to grade the {result_type}. \nSource Text: {src} \n{result_type}: {hyp} \n{"
               "format_prompt} \nScore: ", },
    {"name": "Zero-Shot-Cot",
     "prompt": "{task_description} \nHere is an example:\nSource Text: {ex1_src} \n{result_type}: {"
               "ex1_hyp}\nJudgement: <Description of reasons>. Therefore the score is {"
               "ex1_score}\n\nNow it is your turn to grade the {result_type}.\nSource Text: {src} \n{result_type}: {hyp} \n{format_prompt} \nFirst, think step by step and explain your thought process, then return your judgment in the format 'Judgment: '.", },
    {"name": "Zero-Shot-Cot-Emotion",
     "prompt": "{task_description} \nHere is an example:\nSource Text: {ex1_src} \n{result_type}: {"
               "ex1_hyp}\nJudgement: <Description of emotions and reasons>. Therefore the score is  {"
               "ex1_score}\n\nNow it is your turn to grade the {result_type}.\nSource Text: {src} \n{result_type}: {hyp} \n{format_prompt} \nFirst describe "
               "your emotions, then think step by step and explain your thought process, finally return your judgment in the format 'Judgment: '.", },
]

def generate_dataset_embeddings(df, out_path = None):
    model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')

    src_embeddings = model.encode(df["SRC"].tolist())
    hyp_embeddings = model.encode(df["HYP"].tolist())

    df["embedding"] = list(np.hstack((src_embeddings, hyp_embeddings)))

    if out_path:
        df.to_json(out_path, orient="records")

    del model
    torch.cuda.empty_cache()
    return df




COMBINED_PROMPTS = list(itertools.product(*[BASE_PROMPTS, TASK_DESCRIPTIONS, FORMAT_PROMPTS]))

TASK_INSERTS = {"en_es": "translation from English to Spanish",
                "en_zh": "translation from English to Chinese",
                "en_de": "translation from English to German",
                "zh_en": "translation from Chinese to English",
                "summarization": "summary of a given source text"}


def fill_prompt(SRC, HYP, base_prompt, task_description, format_prompt, task_insert, result_type, examples):
    format = format_prompt["format_prompt"].format(result_type=result_type.lower())
    description = task_description["description"].format(task_specific_insert=task_insert)

    if result_type =="summary":
        score = str(format_prompt["normalization_sum"](examples[0]["GT_Score"]))
    else:
        score = str(format_prompt["normalization_mt"](examples[0]["GT_Score"]))
    prompt = base_prompt["prompt"].format(task_description=description,
                                          format_prompt=format,
                                          result_type=result_type,
                                          src=SRC, hyp=HYP,
                                          ex1_src=examples[0]["SRC"],
                                          ex1_hyp=examples[0]["HYP"],
                                          ex1_score=score)

    return {"format_prompt": {"name": format_prompt["name"], "regex": format_prompt["regex"]},
            "task_description": task_description["name"],
            "base_prompt": {"name": base_prompt["name"], "prompt": prompt}}


def fill_all_prompts(SRC, HYP, task_insert, result_type, examples, prompt_combinations=None):
    if not prompt_combinations:
        return [fill_prompt(SRC, HYP, base_prompt, format_prompt, task_description, task_insert, result_type, examples) for
            base_prompt,
            format_prompt,
            task_description in COMBINED_PROMPTS]
    else:
        return [fill_prompt(SRC, HYP, base_prompt, format_prompt, task_description, task_insert, result_type, examples)
                for
                base_prompt,
                format_prompt,
                task_description in prompt_combinations]


def prepare_prompts_from_df(df, outname=None, prompt_combinations=None):
    """
    Prepares a dataframe of prompts from the provided dataset. The dataset should at least have the columns:
    "task", "SRC", "HYP"

    :param df: The input dataframe containing the dataset.
    :param outname: Optional. The name of the output file to save the resulting dataframe.
    :param prompt_combinations: Optional. A list of prompt combinations to apply to the dataset samples.

    :return: A dataframe with all prompts that should be run.
    """
    tqdm.pandas()

    # Apply task specific fillers in new dataframe columns
    df["result_type"] = df.apply(lambda row: "Translation" if row["task"] != "summarization" else "Summary", axis=1)
    df["task_specific_insert"] = df.progress_apply(lambda row: TASK_INSERTS[row["task"]], axis=1)

    if not prompt_combinations:
        # Apply all prompt combinations to all dataset samples
        df["prompts"] = df.apply(lambda row: fill_all_prompts(row["SRC"], row["HYP"], row["task_specific_insert"],
                                                                   row["result_type"], row["retrieval_examples"]),
                                      axis=1)

    else:
        df["prompts"] = df.apply(lambda row: fill_all_prompts(row["SRC"], row["HYP"], row["task_specific_insert"],
                                                              row["result_type"], row["retrieval_examples"],
                                                              prompt_combinations),
                                 axis=1)

    df = df.drop(labels=["SRC", "HYP", "task_specific_insert", "result_type"], axis=1)

    if outname:
        df.to_json(outname, orient="records", force_ascii=False)

    return df

def find_n_closest(in_df, ex_df, n=5):
    best_samples = []
    for row in tqdm(in_df.iterrows(), total=len(in_df)):
        ex_df["cos_sim"] = ex_df.apply(lambda ex_row: 1 - spatial.distance.cosine(row[1]["embedding"],
                                                                                  ex_row["embedding"]), axis=1)
        ex_df = ex_df.sort_values(by=['cos_sim'], ascending=False)
        best_samples.append(ex_df[:n].to_dict('records'))
    in_df["retrieval_examples"] = best_samples

    return in_df

def find_by_name(name, d):
    for p in d:
        if p["name"] == name:
            return p
def get_x_best2(path, x=3):
    df = pd.read_json(path)
    df = df.sort_values(by=["kendall"], ascending=False)
    prompts = []
    for task in df["task"].unique().tolist():
        selected = df[df["task"] == task][:x].values.tolist()

        prompts += [(find_by_name(s[4], BASE_PROMPTS),
                    find_by_name(s[2], TASK_DESCRIPTIONS),
                    find_by_name(s[1]["name"],FORMAT_PROMPTS))
                    for s in selected]

    return prompts

def get_x_best(path, x=3):
    df = pd.read_json(path)
    df = df.sort_values(by=["kendall"], ascending=False)

    # The ID's have the task in them, therefore we first need to replace the name to remove duplicate prompts
    df['ID'] = df['ID'].str.replace('summarization', 'task')
    df['ID'] = df['ID'].str.replace('en_de', 'task')
    df['ID'] = df['ID'].str.replace('en_zh', 'task')
    df = df.drop_duplicates(subset="ID")

    prompts = []
    for task in df["task"].unique().tolist():
        for prompt in ["Zero-Shot", "Zero-Shot-Cot", "Zero-Shot-Cot-Emotion"]:
            selected = df[(df["task"] == task) & (df["prompt"] == prompt)][:x].values.tolist()
            prompts += [(find_by_name(s[4], BASE_PROMPTS),
                         find_by_name(s[2], TASK_DESCRIPTIONS),
                         find_by_name(s[1]["name"], FORMAT_PROMPTS))
                        for s in selected]

    return prompts

def escape(s):
    return "``" + s.replace("\n", "\\n").replace("{", "\{").replace("}", "\}").replace("_", "\_") + "''"

if __name__ == '__main__':
    # Generate embeddings for the new dataset for which prompts should be built
    test_df2 = load_dev_df()
    test_df2 = generate_dataset_embeddings(test_df2, out_path="<PATH_TO_NEW_TEST_EMBEDDINGS>")

    # Generate embeddings for the retrieval dataset
    retrieval_df = load_retrieval_df()
    retrieval_df = generate_dataset_embeddings(retrieval_df, out_path="<PATH_TO_RETRIEVAL_EMBEDDINGS>")

    # Find the examples to be presented
    test_embeddings2 = pd.read_json("<PATH_TO_NEW_TEST_EMBEDDINGS>", orient="records")
    retrieval_embeddings = pd.read_json("<PATH_TO_RETRIEVAL_EMBEDDINGS>", orient="records")
    df_with_samples = find_n_closest(test_embeddings2, retrieval_embeddings)
    df_with_samples.to_json("<PATH_TO_FEW_SHOT_AUGMENTED_TEST2>")

    # Prepare the prompts
    prepare_prompts = True
    if prepare_prompts:
        zero_shot_file = "<PATH_TO_ZERO_SHOT_FILE>"
        x_best_zero_shot = get_x_best(zero_shot_file, 1)
        few_shot_samples_test2 = pd.read_json("<PATH_TO_FEW_SHOT_AUGMENTED_TEST2>")

        prompts = prepare_prompts_from_df(few_shot_samples_test2, "<PATH_TO_PROMPTS_OUTPUT>", x_best_zero_shot)
