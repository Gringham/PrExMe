from re import T
import numpy as np
import pandas as pd
from sympy import besseli

from project_root import join_with_root

BASE_TABLE = '''\\begin{{table*}}[htb]
    \\centering
    \\begin{{tabular}}{{|l|l|l|l|l|l|}}
        \\hline
        \\textbf{{Model}} & \\textbf{{Prompt}} & \\textbf{{KD}} & \\textbf{{PE}} & \\textbf{{SP}} & \\textbf{{ACC}}\\\\
        {CONTENT}\\hline
    \\end{{tabular}}
    \\caption{{Best performing promts of the phase 2 evaluation on the Eval4NLP dev set. We present the \\textbf{{K}}en\\textbf{{D}}all, \\textbf{{SP}}earman and \\textbf{{PE}}arson, as well as the tie calibrated pair-wise \\textbf{{ACC}}uracy. We bold the two largest correlations per column. Baselines are indicated with a \\textit{{B:}}. The middle column shows the prompt combination for which the correlations are reported. For the Baselines, it instead shows the model that was used for the reported correlations. The asterisk indicates all metrics that are in the best significance cluster (not including BARTScore and XComet) according to a permute-input test (p<=0.075).}}
    \\label{{tab:bestPhase2}}
\\end{{table*}}'''

BASE_TABLE_DENSE = '''\\begin{{table}}[htb]
\\small
    \\centering
    \\begin{{tabular}}{{|l|{LS}|}}
        \\hline
        \\textbf{{Model}} & {TASKS}\\\\
        \\hline
        {CONTENT}\\hline
    \\end{{tabular}}
    \\caption{{Best performing promts of the phase 1 evaluation on the Eval4NLP train set. We present the Kendall correlation across tasks. Bold values indicate the two highest correlations. The asterisk shows which models are in the best significance cluster (ignoring BARTScore and XComet). More details can be found in Appendix \\ref{{phase2perf}}.}}
    \\label{{tab:dense_train}}
\\end{{table}}'''

TASK_BLOCK = '''\\hline
\\textbf|||{TASKNAME}::: & & & & & \\\\
{SCORES}'''

def add_0(string):
    star = False
    string = str(string)
    if "*" in string:
        star = True
        string = string.replace("*", "")
    if len(string) == 0:
        return string
    if len(string) == 1:
        string += ".0"
    while len(string) < 5:
        string += "0"
    if star:
        string += "*"
    return string

def shortname(model):
    if "Platypus2-70B" in model or "Platypus70B" in model:
        return "\\textsc{Platypus2-70B}"
    if "OpenOrca" in model:
        return "\\textsc{OrcaPlt-13B}"
    if "Nous" in model:
        return "\\textsc{NousHermes-13B}"
    else:
        return "\\textsc{" + model + "}"
    
def significance_cluster(matrix, bonferonni=False, p=0.075):
    # Compute significance clusters based on a significance p value matrix
    significances = {model:0 for model in matrix}
    if bonferonni:
        p = p/len(matrix)
    for model in matrix:
        for other_model in matrix:
            if model != other_model:
                if matrix[model][other_model] < p:
                    significances[other_model] += 1

    significance_clusters = {}
    for model, cluster in significances.items():
        significance_clusters[cluster] = [model] if cluster not in significance_clusters else significance_clusters[cluster] + [model]

    return significances, significance_clusters

def get_significance(dataset, task):
    path = "<PATH_TO_SIGNIFICANCE_MATRICES>/___{}___{}___significance.json".format(dataset, task)
    sign_dict = pd.read_json(path).to_dict()
    return sign_dict

def model_in_list(model, model_list, approach=None):
    for m in model_list:
        if approach == None:
            if model in m and "DSBA" not in m and "LocalGembaMQM" not in m:
                return True
        elif approach in m and model in m:
            return True
    return False

def add_star(value, sign_clusters, model, approach=None):
    # Exclude XComet from the significance clusters
    if len(sign_clusters[0]) == 1 and model_in_list("XComet", sign_clusters[0]):
        if model_in_list(model, sign_clusters[1], approach):
            return str(value) + "*"
    elif model_in_list(model, sign_clusters[0], approach):
        return str(value) + "*"
    return value

def print_table_dense(prex_df, baseline_df, dataset):
    rows = []
    tasks = prex_df["task"].unique()
    order = ["en_de", "en_es", "en_zh", "he_en", "zh_en", "summarization", ]
    order_dict = {task: index for index, task in enumerate(order)}
    tasks = sorted(tasks, key=lambda task: order_dict[task])

    TASKS = " & ".join(["\\textbf{" + task.replace("_", "\\_") + "}" if task != "summarization" else "\\textbf{sum}" for task in tasks])
    for model, model_group in prex_df.groupby(["model"]):
        line = [model[0]]
        for task in tasks:
            task_group = model_group[model_group["task"] == task]
            matrix = get_significance(dataset, task)
            _, sign_clusters = significance_cluster(matrix)

            best_prompts = dict(task_group.loc[task_group['kendall'].idxmax()])
            kd = round(best_prompts["kendall"], 3)

            line += [add_star(kd, sign_clusters, model[0])]
            #if "FS" == best_prompts["mode"]:
            #    line[1] = line[1].replace("ZS", "OS")
        rows.append(line)

    baseline_df_dataset = baseline_df[baseline_df["dataset"]==dataset]
    for approach, approach_group in baseline_df_dataset.groupby(["approach"]):
        line = ["B:" + approach[0][:10]]
        for task in tasks:
            task_group = approach_group[approach_group["task"] == task]
            matrix = get_significance(dataset, task)
            _, sign_clusters = significance_cluster(matrix)
            added = False
            for idx, row in task_group.iterrows():
                if "Platypus2-70B" in row["model"] and "DSBA" in row["approach"] or "DSBA" not in row["approach"]:
                    if "Platypus2-70B" in row["model"] and "Gemba" in row["approach"] or "Gemba" not in row["approach"]:
                        line += [add_star(round(row["kendall"],3), sign_clusters, row["model"], row["approach"])]
                        added = True
            if not added:
                line += [np.nan]
        rows.append(line)
     
                


    for x in range(-len(tasks),0):
        top = np.argmax([float(str(r[x]).replace("*","")) for r in rows])
        top2 = np.argsort([np.max(float(str(r[x]).replace("*",""))) for r in rows])[-2]
        for i in range(len(rows)):
            rows[i][x] = add_0(str(rows[i][x]))
            if i == top or i == top2:
                rows[i][x] = "\\textbf{" + rows[i][x] + "}"
    rows = [" & ".join([str(r) for r in row]) for row in rows]        
    rows = "\\\\\n".join(rows)+"\\\\\n"

    table = BASE_TABLE_DENSE.format(CONTENT=rows, TASKS=TASKS, LS="|".join(["l"]*len(tasks)))
    print(table)

    return table

def print_table(prex_df, baseline_df, dataset):
    order = ["en_de", "en_es", "en_zh", "he_en", "zh_en", "summarization", ]
    content_blocks = []
    for task, group in prex_df.groupby(["task"]):
        rows = []

        matrix = get_significance(dataset, task[0])
        significances, sign_clusters = significance_cluster(matrix)
        for inner_name, inner_group in group.groupby(["model"]):
            best_prompts = dict(inner_group.loc[inner_group['kendall'].idxmax()])

            kd = round(best_prompts["kendall"], 3)
            pe = round(best_prompts["pearson"], 3)
            if type(best_prompts["spearman"]) == list:
                sp = round(list(best_prompts["spearman"])[0], 3)
            else:
                sp = round(best_prompts["spearman"], 3)
            acc = round(best_prompts["kendall_tie_corrected"], 3)


            line = [shortname(inner_name[0]), best_prompts["prompt"].replace("Zero-Shot", "ZS").replace(
                "One-Shot", "OS").replace("emotion", "EM") + ", " + best_prompts[
                "task_description"]
                    +  ", " + best_prompts["regex"]["name"], add_star(kd, sign_clusters, inner_name[0]), pe, sp, acc]
            if "FS" == best_prompts["mode"]:
                line[1] = line[1].replace("ZS", "OS")

            rows.append(line)

        for idx, row in baseline_df[(baseline_df["task"]==task[0]) & (baseline_df["dataset"]==dataset)].iterrows():
            # Filter for best baselines
            if "Platypus2-70B" in row["model"] and "DSBA" in row["approach"] or "DSBA" not in row["approach"]:
                if "Platypus2-70B" in row["model"] and "Gemba" in row["approach"] or "Gemba" not in row["approach"]:
                    model_block = ""
                    if "DSBA" in row["approach"] or "Gemba" in row["approach"]:
                        model_block = "Model:" + shortname(row["model"])
                    line = ["B:" + row["approach"][:10], model_block,add_star(round(row["kendall"],3), sign_clusters,row["model"], row["approach"]),
                            round(row["pearson"],3), round(row["spearman"],3), round(
                                row["kendall_tie_corrected"],3)]

                    rows.append(line)

        for x in range(-4,0):
            top = np.argmax([float(str(r[x]).replace("*","")) for r in rows])
            top2 = np.argsort([np.max(float(str(r[x]).replace("*",""))) for r in rows])[-2]
            for i in range(len(rows)):
                rows[i][x] = add_0(str(rows[i][x]))
                if i == top or i == top2:
                    rows[i][x] = "\\textbf{" + rows[i][x] + "}"

        # Bold the best ones
        rows = [" & ".join(row) for row in rows]
        rows = "\\\\\n".join(rows)+"\\\\\n"


        content_blocks.append((task[0],TASK_BLOCK.format(TASKNAME=task[0].replace("_", "\\_"), SCORES=rows).replace(
            "|||","{").replace(":::","}")))

    content_blocks = sorted(content_blocks, key=lambda x: order.index(x[0]))
    table = BASE_TABLE.format(CONTENT="".join([c[1] for c in content_blocks]))
    table = table.replace("complex token labels", "complex l.").replace("Cot-Emotion", "CoT-EM").replace("easy token labels", "simple labels")
    print(table)

    return table


if __name__ == '__main__':
    baseline_df = pd.read_json("<PATH_TO_CORRELATION_FILES>/baseline_correlations.json")

    prex_df_zero_shot = pd.read_json("<PATH_TO_CORRELATION_FILES>/zero_shot_train_sign.json")
    prex_df_zero_shot["mode"] = "ZS"
    prex_df_few_shot = pd.read_json("<PATH_TO_CORRELATION_FILES>/few_shot_train_sign.json")
    prex_df_few_shot["mode"] = "FS"
    prex_df = pd.concat([prex_df_zero_shot, prex_df_few_shot]).reset_index()
    print_table(prex_df, baseline_df, "train")#

    print("\n\n\n----------------------------------\n\n\n")

    print_table_dense(prex_df, baseline_df, "train")

    print("\n\n\n----------------------------------\n\n\n")

    prex_df_zero_shot = pd.read_json("<PATH_TO_CORRELATION_FILES>/zero_shot_dev_sign.json")
    prex_df_zero_shot["mode"] = "ZS"
    prex_df_few_shot = pd.read_json("<PATH_TO_CORRELATION_FILES>/few_shot_dev_sign.json")
    prex_df_few_shot["mode"] = "FS"
    prex_df = pd.concat([prex_df_zero_shot]).reset_index()
    print_table(prex_df, baseline_df, "dev")

    print("\n\n\n----------------------------------\n\n\n")

    print_table_dense(prex_df, baseline_df, "dev")

    print("\n\n\n----------------------------------\n\n\n")

    prex_df_zero_shot = pd.read_json("<PATH_TO_CORRELATION_FILES>/zero_shot_test_sign.json")
    prex_df_zero_shot["mode"] = "ZS"
    prex_df_few_shot = pd.read_json("<PATH_TO_CORRELATION_FILES>/few_shot_test_sign.json")
    prex_df_few_shot["mode"] = "FS"
    prex_df = pd.concat([prex_df_zero_shot, prex_df_few_shot]).reset_index()
    prex_df = prex_df.drop_duplicates(subset=['kendall'])

    print_table(prex_df, baseline_df, "test")

    print("\n\n\n----------------------------------\n\n\n")

    print_table_dense(prex_df, baseline_df, "test")

    print("\n\n\n----------------------------------\n\n\n")
    

    prex_df_zero_shot = pd.read_json("<PATH_TO_CORRELATION_FILES>/zero_shot_test2_sign.json")
    prex_df_zero_shot["mode"] = "ZS"
    #prex_df_few_shot = pd.read_json(join_with_root("outputs/evaluation/corr_few_shot_test_avg.json"))
    #prex_df_few_shot["mode"] = "FS"
    #prex_df = pd.concat([prex_df_zero_shot, prex_df_few_shot]).reset_index()
    #prex_df = prex_df.drop_duplicates(subset=['kendall'])

    print_table(prex_df_zero_shot, baseline_df, "test2")

    print("\n\n\n----------------------------------\n\n\n")

    print_table_dense(prex_df_zero_shot, baseline_df, "test2")

