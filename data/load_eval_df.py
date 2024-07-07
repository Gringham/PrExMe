import csv
import pandas as pd

from project_root import join_with_root

def load_retrieval_df():
    '''
    @return: A dataframe with the retrieval data, featuring the columns:  Score, model, SRC, HYP, task
    '''
    file_dict = {
        "en_de": join_with_root("data/dict/mqm21_en-de.tsv"),
        "zh_en": join_with_root("data/dict/mqm21_zh-en.tsv"),
        "summarization": join_with_root("data/dict/rose_summarization.tsv")
    }

    # Create a dictionary of dataframes
    df_dict = {k: pd.read_csv(v, sep="\t", quoting=csv.QUOTE_NONE) for k, v in file_dict.items()}

    # Create a column containing the task name
    for task, data in df_dict.items():
        df_dict[task]["task"] = [task] * len(df_dict[task])

    # Rename certain columns to make the dataframes comparable
    df_dict["summarization"]["LP"] = [None] * len(df_dict["summarization"])
    df_dict["summarization"]["seg-id"] = [None] * len(df_dict["summarization"])

    df_dict["en_de"] = df_dict["en_de"].rename(columns={"DA": "GT_Score", "seg-id": "id"})
    df_dict["zh_en"] = df_dict["zh_en"].rename(columns={"DA": "GT_Score", "seg-id": "id"})
    df_dict["summarization"] = df_dict["summarization"].rename(columns={"model": "system", "score":"GT_Score"})

    # concatenate the dataframes of the different tasks
    df = pd.concat([df_dict[k] for k in df_dict], ignore_index=True)
    return df

def load_train_df():
    '''
    @return: A dataframe with the training data, featuring the columns: domain, DOC, Score, id, system-name, SRC, HYP
    '''
    file_dict = {
        "en_de": join_with_root("data/train/train_en_de.tsv"),
        "zh_en": join_with_root("data/train/train_zh_en.tsv"),
        "summarization": join_with_root("data/train/train_summarization.tsv")
    }

    # Create a dictionary of dataframes
    df_dict = {k: pd.read_csv(v, sep="\t", quoting=csv.QUOTE_NONE) for k, v in file_dict.items()}

    # Create a column containing the task name
    for task, data in df_dict.items():
        df_dict[task]["task"] = [task] * len(df_dict[task])

    # Rename certain columns to make the dataframes comparable
    df_dict["summarization"]["domain"] = [None] * len(df_dict["summarization"])
    df_dict["summarization"]["DOC"] = [None] * len(df_dict["summarization"])
    df_dict["en_de"] = df_dict["en_de"].rename(columns={"mqm": "GT_Score", "seg-id": "id"})
    df_dict["zh_en"] = df_dict["zh_en"].rename(columns={"mqm": "GT_Score", "seg-id": "id"})
    df_dict["summarization"] = df_dict["summarization"].rename(columns={"model_id": "system-name", "Score":"GT_Score"})

    # concatenate the dataframes of the different tasks
    df = pd.concat([df_dict[k] for k in df_dict], ignore_index=True)
    return df


def load_dev_df():
    """
    @return: Loads a dataframe of the dev set, following the same layout as the train set 
    """

    # We need tp separate dictionaries as the scores are saved separately for the dev sets
    file_dict = {
        "en_de": join_with_root("data/dev/dev_en_de.tsv"),
        "zh_en": join_with_root("data/dev/dev_zh_en.tsv"),
        "summarization": join_with_root("data/dev/dev_summarization.tsv")
    }

    separate_score_dict = {
        "en_de": join_with_root("data/dev/seg.scores_en_de"),
        "zh_en": join_with_root("data/dev/seg.scores_zh_en"),
        "summarization": join_with_root("data/dev/seg.scores_summarization")
    }

    # Load both file types as pd dfs and add the scores to the first dict
    df_dict = {k: pd.read_csv(v, sep="\t", quoting=csv.QUOTE_NONE) for k, v in file_dict.items()}
    df_dict_scores = {k: pd.read_csv(v, sep="\t", header=None, names=["Score"]) for k, v in
                           separate_score_dict.items()}
    for k in df_dict:
        df_dict[k]["Score"] = df_dict_scores[k]["Score"]

    # Create a column containing the task name
    for task, data in df_dict.items():
        df_dict[task]["task"] = [task] * len(df_dict[task])

    df_dict["summarization"]["domain"] = [None] * len(df_dict["summarization"])
    df_dict["summarization"]["DOC"] = [None] * len(df_dict["summarization"])
    df_dict["en_de"] = df_dict["en_de"].rename(columns={"Score": "GT_Score", "seg-id": "id"})
    df_dict["zh_en"] = df_dict["zh_en"].rename(columns={"Score": "GT_Score", "seg-id": "id"})
    df_dict["summarization"] = df_dict["summarization"].rename(columns={"model_id": "system-name", "Score":"GT_Score"})

    # concatenate the dataframes of the different tasks
    df = pd.concat([df_dict[k] for k in df_dict], ignore_index=True)
    return df

def load_test_df(label="eval4nlp23", to_files=None):
    df = None

    if label == "eval4nlp23":
        file_dict = {
            "en_de": join_with_root("data/test/mt_en_de_ground_truth_cleaned_l.tsv"),
            "en_es": join_with_root("data/test/mt_en_es_ground_truth_cleaned_l.tsv"),
            "en_zh": join_with_root("data/test/mt_en_zh_ground_truth_cleaned_l.tsv"),
            "summarization": join_with_root("data/test/summarization_ground_truth_cleaned_l.tsv")
        }

        # Create a dictionary of dataframes
        df_dict = {k: pd.read_csv(v, sep="\t", quoting=csv.QUOTE_NONE) for k, v in file_dict.items()}

        # Create a column containing the task name
        for task, data in df_dict.items():
            df_dict[task]["task"] = [task] * len(df_dict[task])

        # Rename score columns to GT_Score
        df_dict["summarization"]  = df_dict["summarization"].rename(columns={"summary_score":"GT_Score"})
        df_dict["en_de"] = df_dict["en_de"].rename(columns={"mqm": "GT_Score"})
        df_dict["en_es"] = df_dict["en_es"].rename(columns={"mqm": "GT_Score"})
        df_dict["en_zh"] = df_dict["en_zh"].rename(columns={"mqm": "GT_Score"})

        # concatenate the dataframes of the different tasks
        df = pd.concat([df_dict[k] for k in df_dict], ignore_index=True)

        # Add dummy columns and add up names
        df = df.rename(columns={"TGT": "HYP"})
        df["DOC"] = [None]*len(df)
        df["system-name"] = [None] * len(df)
        df["id"] = list(range(len(df)))
        df["domain"] = [None] * len(df)
        df = df.drop(labels=["Unnamed: 0"], axis = 1)

    elif label == "wmt_23_seahorse":
        # Loads the (google) mqm annotations of WMT2023 and the summarization annotations of seahorse as reference free datasets
        df_dict = {
            "en_de": pd.read_json(join_with_root("data/new_test/en-de_wmt_23_test.json"),orient="records"),
            "zh_en": pd.read_json(join_with_root("data/new_test/zh-en_wmt_23_test.json"),orient="records"),
            "he_en": pd.read_json(join_with_root("data/new_test/he-en_wmt_23_test.json"),orient="records"),
            "summarization": pd.read_json(join_with_root("data/new_test/seahorse_test.json"), orient="records", lines=True)
        }

        # Create a column containing the task name
        for task, data in df_dict.items():
            df_dict[task]["task"] = [task] * len(df_dict[task])

        df_dict["en_de"] = df_dict["en_de"].rename(columns={"SYS": "system-name"})
        df_dict["he_en"] = df_dict["he_en"].rename(columns={"SYS": "system-name"})
        df_dict["zh_en"] = df_dict["zh_en"].rename(columns={"SYS": "system-name"})
        df_dict["summarization"] = df_dict["summarization"].rename(columns={"model": "system-name", "gem_id":"DOC", "summary":"HYP"})

        for k, v in df_dict.items():
            df_dict[k] = v[["GT_Score", "system-name", "SRC", "HYP", "task", "DOC"]]
            df_dict[k] = df_dict[k].reset_index(drop=True)
            print(df_dict[k]["system-name"].unique())

        # concatenate the dataframes of the different tasks
        df = pd.concat([df_dict[k] for k in df_dict])

        print("Full lengths", [(v, len(p)) for v, p in df_dict.items()])
        print([(v, len(p[p["GT_Score"]!="None"])) for v, p in df_dict.items()])

    if to_files:
        for task, data in df_dict.items():
            data.to_json(to_files + f"_{task}.json", orient="records")
#
    return df

if __name__ == '__main__':
    a = load_train_df()
    print(a)
    b = load_dev_df()
    print(b)
    d = load_retrieval_df()
    print(d)
    e = load_test_df("wmt_23_seahorse")
    print(e)