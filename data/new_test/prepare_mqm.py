# Create valid scorefile for MQM evaluation. References are excluded from the evaluation.
import os

import pandas as pd

def read_newline_sep(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [l.strip() for l in lines]

for lp in ["en-de", "he-en", "zh-en"]:
    src = f"~/.mt-metrics-eval/mt-metrics-eval-v2/wmt23/sources/{lp}.txt"
    score_file = f"~/.mt-metrics-eval/mt-metrics-eval-v2/wmt23/human-scores/{lp}.mqm.seg.score"
    hyp_base = f"~/.mt-metrics-eval/mt-metrics-eval-v2/wmt23/system-outputs/{lp}"
    SYS = os.listdir(hyp_base)
    hyp = [hyp_base +"/"+ b for b in SYS]
    docs = f"~/.mt-metrics-eval/mt-metrics-eval-v2/wmt23/documents/{lp}.docs"

    src_l = read_newline_sep(src)
    score_df = pd.read_csv(score_file, sep="\t", header=None, names=["SYS", "GT_Score"])
    docs_l = read_newline_sep(docs)
    hyp_dict = {}
    for h, s in zip(hyp, SYS):
        hyp_l = read_newline_sep(h)
        hyp_dict[s.replace(".txt", "")] = pd.DataFrame({"HYP": hyp_l, "SYS": [s.replace(".txt", "")] * len(hyp_l)})


    hyp_dfs = []
    for sys in score_df["SYS"].unique():
        hyp_dfs.append(hyp_dict[sys])

    hyp_df = pd.concat(hyp_dfs, ignore_index=True)
    hyp_df = hyp_df[hyp_df["SYS"] != "synthetic_ref"]
    try:
        hyp_df = hyp_df[hyp_df["SYS"] != "refA"]
        score_df = score_df[score_df["SYS"] != "refA"]
    except:
        pass

    try:
        hyp_df = hyp_df[hyp_df["SYS"] != "refB"]
        score_df = score_df[score_df["SYS"] != "refB"]
    except:
        pass


    hyp_df["SYS_Comp"] = score_df["SYS"].tolist()
    hyp_df["GT_Score"] = score_df["GT_Score"].tolist()
    hyp_df["SRC"] = src_l * len(hyp_df["SYS"].unique())
    hyp_df["DOC"] = docs_l * len(hyp_df["SYS"].unique())

    hyp_df.to_csv(f"{lp}_wmt_23_test.tsv", sep="\t", index=False)
