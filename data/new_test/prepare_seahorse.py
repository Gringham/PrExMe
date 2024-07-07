import csv

import datasets
import tensorflow_datasets as tfds
import pandas as pd

from project_root import join_with_root

seahorse = pd.read_csv(join_with_root("data/new_test/seahorse_test.tsv"), sep="\t", quoting=csv.QUOTE_NONE)

mlsum_de = datasets.load_dataset('GEM/mlsum', "de")["test"].to_pandas()
mlsum_es = datasets.load_dataset('GEM/mlsum', "es")["test"].to_pandas()

corpora = seahorse["gem_id"].tolist()
corpora = list(set([c.split("-test-")[0] for c in corpora]))
xlsum_names = [c.split("_")[1] for c in corpora if "xlsum" in c]

xlsum = {l:datasets.load_dataset('GEM/xlsum', l)["test"].to_pandas() for l in xlsum_names}
xsum = datasets.load_dataset('GEM/xsum')["test"].to_pandas()

df_with_lingua = datasets.load_dataset("mtc/seahorse_dataset_with_articles")["test"].to_pandas()
df_with_lingua = df_with_lingua[df_with_lingua['gem_id'].str.contains("wiki_lingua")]


def filter_seahorse(row):
    try:
        if "mlsum_de" in row["gem_id"]:
            return mlsum_de[mlsum_de["gem_id"] == row["gem_id"]]["text"].tolist()[0]
        elif "mlsum_es" in row["gem_id"]:
            return mlsum_es[mlsum_es["gem_id"] == row["gem_id"]]["text"].tolist()[0]
        elif "xsum" in row["gem_id"]:
            return xsum[xsum["gem_id"] == row["gem_id"]]["document"].tolist()[0]
        elif "xlsum" in row["gem_id"]:
            for name in xlsum_names:
                if name in row["gem_id"]:
                    return xlsum[name][xlsum[name]["gem_id"] == row["gem_id"]]["text"].tolist()[0]
        elif "wiki_lingua" in row["gem_id"]:
            return df_with_lingua[df_with_lingua["gem_id"] == row["gem_id"]]["article"].tolist()[0]
    except Exception as e:
        print(e)
        return None

seahorse["SRC"] = seahorse.apply(filter_seahorse, axis=1)

def construct_score(row):
    if row["question1"] == "No":
        return 0
    score = 0
    for i in range(1, 6):
        if row[f"question{i}"] == "Yes":
            score += 0.2
    return score

seahorse["GT_Score"] = seahorse.apply(construct_score, axis=1)
seahorse.to_json(join_with_root("data/new_test/seahorse_test.json"), orient="records", lines=True)

# save split of wikilingua to a different file
lingua = seahorse[seahorse['gem_id'].str.contains("wiki_lingua")]
lingua.to_json(join_with_root("data/new_test/seahorse_test_lingua.json"), orient="records", lines=True)