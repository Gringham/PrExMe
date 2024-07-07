import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
from project_root import join_with_root

def abbreviate_labels(labels):
    abbreviations = {
        "easy token labels": "simple l.",
        "complex token labels": "complex l."
    }
    return [abbreviations.get(label, label) for label in labels]

def gen_corr_heatmaps_2(df, main_sub_axes_list, corr_measure="kendall", measure=np.median):
    sns.set(rc={'figure.figsize': (3.15, 3)})  # Keeping the original figure size
    sns.set_theme(style="dark")
    sns.set_palette("colorblind")

    # Determine the number of subplots needed
    num_axes = len(main_sub_axes_list)
    fig, axes = plt.subplots(1, num_axes, gridspec_kw={'width_ratios': [1.016]*num_axes, 'wspace': 0.1})  # Adjusted width ratios

    if num_axes == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one subplot

    for idx, (main_axis, sub_axis) in enumerate(main_sub_axes_list):
        ax = axes[idx]

        order = ["en_de", "en_es", "en_zh", "zh_en", "summ"]
        df["task"] = [d.replace("summarization", "summ") for d in df["task"].tolist()]

        if isinstance(df[main_axis].tolist()[0], dict):
            df[main_axis] = [u["name"] for u in df[main_axis].tolist()]
        if isinstance(df[sub_axis].tolist()[0], dict):
            df[sub_axis] = [u["name"] for u in df[sub_axis].tolist()]

        u1 = df[main_axis].unique()
        u2 = df[sub_axis].unique()
        try:
            u2 = sorted(u2, key=lambda x: order.index(x))
        except ValueError:
            pass

        res_dict = {u: [] for u in u2}

        for s1 in u2:
            for s2 in u1:
                subset = df[(df[sub_axis] == s1) & (df[main_axis] == s2)]
                if subset.empty:
                    print(f"Empty subset for {sub_axis}={s1} and {main_axis}={s2}")
                num = np.array(subset[corr_measure].tolist())
                if len(num) == 0:
                    print(f"Empty num array for {sub_axis}={s1} and {main_axis}={s2}")
                m = measure(num)
                m = 0 if np.isnan(m) else m
                res_dict[s1].append(m)

        corr_dict = {u: [] for u in u2}

        for s1 in u2:
            for s2 in u2:
                if not res_dict[s1] or not res_dict[s2]:
                    print(f"Empty list in res_dict for {s1} or {s2}")
                corr = scipy.stats.kendalltau(res_dict[s1], res_dict[s2], nan_policy="omit")
                corr_dict[s1].append(corr.statistic)

        corr_df = pd.DataFrame(corr_dict).T
        corr_df.columns = [u for u in corr_df.T.columns]

        corr_df = corr_df.rename(columns={
            "Zero-Shot": "PZS", 
            "Zero-Shot-Cot": "ZSC", 
            "Zero-Shot-Cot-Emotion": "ZSCE"
        }).T

        corr_df = corr_df.rename(columns={
            "Zero-Shot": "PZS", 
            "Zero-Shot-Cot": "ZSC", 
            "Zero-Shot-Cot-Emotion": "ZSCE"
        }).T

        matrix = np.triu(np.ones_like(corr_df))
        np.fill_diagonal(matrix, False)

        heatmap = sns.heatmap(
            corr_df.round(2), annot=True, fmt=".2f", annot_kws={"fontsize": 4, 'fontweight': 'bold'}, cbar=idx == num_axes - 1,
            linewidths=.03, mask=matrix, square=True, ax=ax,
            cbar_kws={"shrink": 0.5, "aspect": 10, 'pad': 0.01, "fraction": 0.05} if idx == num_axes - 1 else None, vmin=-0.01, vmax=1
        )

        ax.set_aspect('equal', 'box')  # Ensure the cells are square

        ax.set_xticklabels(abbreviate_labels([label.get_text() for label in ax.get_xticklabels()]), rotation=90, fontsize=6, rotation_mode='anchor')
        ax.set_yticklabels(abbreviate_labels([label.get_text() for label in ax.get_yticklabels()]), rotation=0, fontsize=6, rotation_mode='anchor')
        ax.tick_params(axis='both', which='major', labelsize=6, pad=-3)
        plt.setp(ax.get_xticklabels(), fontsize=4, ha='right')  # Adjusted for less spacing
        plt.setp(ax.get_yticklabels(), fontsize=4)

        if idx == num_axes - 1:  # Adjust colorbar fontsize only for the last subplot
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=6)

        main_axis_dir = {
            "task_description": "Task Desc.",
            "regex2": "Format Req.",
            "model": "Model",
            "filename": "Dataset"
        }
        #ax.set_title(f'{main_axis_dir[main_axis]}', fontsize=6, fontweight='bold')

        if idx != 0:
            ax.set_yticklabels([])

    # Adjust margins manually to ensure labels are visible and cells are square
    fig.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.25, wspace=0.05)  # Reduced wspace
    plt.savefig(join_with_root(f"outputs/plots/heatmaps2/corr_task_description_format.pdf"), bbox_inches='tight', pad_inches=0.01)
    plt.show()
    plt.clf()


# Example usage
if __name__ == '__main__':
    folder = "<PATH_TO_FOLDER_WITH_CORRELATION_JSON_FILES>"

    # Read and concatenate all json files to pandas dataframe
    df_list = []
    for file in os.listdir(folder):
        if file.endswith(".json") and not "baseline" in file.lower():
            df = pd.read_json(os.path.join(folder, file))
            df["filename"] = file
            df["mode"] = "fs" if "few-shot" in file.lower() else "zero-shot"
            df_list.append(df)
    df1 = pd.concat(df_list, ignore_index=True)

    # Filter out summarization tasks and handle regex column
    df1["regex2"] = [str(r["name"]) if isinstance(r, dict) else r for r in df1["regex"].tolist()]

    main_sub_axes_list = [("task_description", "regex2")]
    gen_corr_heatmaps_2(df1, main_sub_axes_list, corr_measure="kendall", measure=np.median)
