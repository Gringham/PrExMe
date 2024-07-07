from datasets import load_dataset
import pandas as pd

from project_root import join_with_root

ds = load_dataset("Salesforce/rose", "xsum")
hyp, src, model, score, task = ([], [], [], [], [])
for line in ds["data"]:
    hyp += list(line["system_outputs"].values())
    src += [line["source"]] * len(line["system_outputs"])
    model_list = list(line["system_outputs"].keys())
    model += model_list
    score += list(line["annotations"][m]["normalized_acu"] for m in model_list)
    task += ["summarization"] * len(line["system_outputs"])

hyp = [h.replace("\t", " ").replace("\n", " ") for h in hyp]
src = [s.replace("\t", " ").replace("\n", " ") for s in src]
df = pd.DataFrame([hyp, src, model, score, task]).T
df.columns = ["HYP", "SRC", "model", "score", "task"]

df.to_csv(join_with_root("data/dict/rose_summarization.tsv"), sep="\t")