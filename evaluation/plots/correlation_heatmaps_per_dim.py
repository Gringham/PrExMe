import time

import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


from nlpstats.correlations import bootstrap

from project_root import join_with_root


def gen_corr_heatmap(df, corr_measure="kendall", main_axis="regex", sub_axis="task", measure=np.median):
    sns.set(rc={'figure.figsize': (2, 2)})

    # How robust is the ranking of prompts given by the main axis across the sub-axis when the other dimensions are
    # aggregated by measure
    order = ["en_de", "en_es", "en_zh", "zh_en", "summ"]
    df["task"] = [d.replace("summarization", "summ") for d in df["task"].tolist()]

    # Accomodate different dataformats
    if type(df[main_axis].tolist()[0]) == dict:
        df[main_axis] = [u["name"] for u in df[main_axis].tolist()]
    if type(df[sub_axis].tolist()[0]) == dict:
        df[sub_axis] = [u["name"] for u in df[sub_axis].tolist()]

    # Order df for nicer heatmaps
    u1 = df[main_axis].unique()
    u2 = df[sub_axis].unique()
    try:
        u2 = sorted(u2, key=lambda x: order.index(x))
    except:
        pass

    res_dict = {u:[] for u in u2}

    # Iterate through each combination of main_axis and sub_axis
    for i, s1 in enumerate(u2):
        for j, s2 in enumerate(u1):
            # Filter data for the current model and task
            subset = df[(df[sub_axis] == s1) & (df[main_axis] == s2)]
            num = np.array(subset[corr_measure].tolist())
            m = measure(num)
            m = 0 if np.isnan(m) else m
            res_dict[s1].append(m)

    matrix_dim1 = []
    matrix_dim2 = []

    corr_dict = {u:[] for u in u2}
    p_values = {u:[] for u in u2}

    corr_list = []
    for i, s1 in enumerate(u2):
        for j, s2 in enumerate(u2):
            corr = scipy.stats.kendalltau(res_dict[s1], res_dict[s2], nan_policy="raise")
            corr_dict[s1].append(corr.statistic)
            p_values[s1].append(corr.pvalue)

            if i < j:
                corr_list.append(corr)
                matrix_dim1.append(res_dict[s1])
                matrix_dim2.append(res_dict[s2])

    corr_df = pd.DataFrame(corr_dict)
    corr_df = corr_df.rename(columns={"Nous":"N", "OpenOrca":"O", "Platypus70B":"P"})
    corr_df = corr_df.rename(columns={"Zero-Shot":"ZS", "Zero-Shot-Cot":"ZSC", "Zero-Shot-Cot-Emotion":"ZSEC"}).T
    corr_df.columns = [u[:4] for u in corr_df.T.columns]

    matrix = np.triu(np.ones_like(corr_df))
    np.fill_diagonal(matrix, False)
    a = sns.heatmap(corr_df, annot=True, annot_kws={"fontsize": 9}, cbar=False, linewidths=.03, mask=matrix)
    a.set_xticklabels(a.get_xticklabels(), verticalalignment='center',
                      horizontalalignment='center', rotation=80)
    a.set_yticklabels(a.get_yticklabels(), rotation=0)
    a.tick_params(axis='both', which='major', labelsize=9)
    a.tick_params(axis='x', which='major', pad = 10)


    #plt.tight_layout()
    #plt.savefig(join_with_root(f"outputs/plots/heatmaps/{main_axis}_along_{sub_axis}_{measure.__name__}_corr.pdf"))
    #plt.show()


    return corr_df.values + matrix*9999, matrix_dim1, matrix_dim2, corr_list

def agg_10p(arr):
    return np.percentile(arr, 90).mean()

def saturation(arr):
    return (1-(np.max(arr)-np.mean(arr)))*np.max(arr)

if __name__ == '__main__':
    train_files = [#join_with_root("outputs/evaluation/corr_zero_shot_train_avg_recombined.json"),
                   join_with_root("outputs/evaluation/new_compute_expanded.json")]

    df = pd.concat([pd.read_json(t) for t in train_files]).reset_index()

    primary_dimensions = ["task_description"]
    #primary_dimensions = ["regex"]
    secondary_dimensions = ["prompt", "task", "model"]

    agg = [np.median, np.mean, np.max, np.min, agg_10p, np.std, saturation]

    all_values = []
    all_corr = []
    all_conf_int = []
    avg_corrs = []
    all_dim1 = []
    all_dim2 = []
    for a in agg:
        pro_cnt = 0
        n_cnt = 0
        dim1 = []
        dim2 = []
        vals = []
        corrs = []
        for p in primary_dimensions:
            for s in secondary_dimensions:
                m, d1, d2, corr = gen_corr_heatmap(df, corr_measure="kendall", main_axis=p, sub_axis=s, measure=a)
                pro_cnt += np.sum((m > 0.3) & (m < 0.99))
                n_cnt += np.sum(m < 0)
                vals += m[m < 0.99].flatten().tolist()
                corrs += corr
                dim1 += d1
                dim2 += d2

        all_values.append(np.array(vals))
        all_corr.append(corrs)
        avg_corrs.append(np.mean([c.statistic for c in corrs]))

        # The bootstrapping method is normally used on correlations of summarization metrics with human judgements
        # and samples in and out the systems and documents. Instead, we have the main and sub dimensions of the heatmaps
        b = bootstrap(np.array(dim1), np.array(dim2), "global", "kendall", "both")

        # lower, upper, mean, median
        all_conf_int.append((b.lower, b.upper, sum(b.samples)/len(b.samples), np.median(b.samples)))
        all_dim1.append(dim1)
        all_dim2.append(dim2)

        print(f"For aggregation type {a} along {s} we have {pro_cnt} strong positive and {n_cnt} negative correlations")

    for a in zip(agg, all_values):
        print(f"Aggregation type {a[0]}: {a[1]}")



    best_methods = []
    for x in range(len(all_conf_int)):
        print(f"Aggregation type {a} confidence interval {x}: {all_conf_int[x]}")


    def correlation_difference(data1, data2, data3, data4):
        a1 = scipy.stats.kendalltau(data1, data2, nan_policy="raise")
        a2 = scipy.stats.kendalltau(data3, data4, nan_policy="raise")
        return a1.statistic - a2.statistic

    def standardize(X: np.ndarray) -> np.ndarray:
        return (X - np.nanmean(X)) / np.nanstd(X)

    def permutation_test(data1, data2, data3, data4, n_permutations=10000):
        # Based on Daniel Deutsch's implementation (https://github.com/danieldeutsch/nlpstats/blob/main/nlpstats/correlations/permutation.py#L107)
        cnt = 0
        data1 = standardize(data1)
        data2 = standardize(data2)
        data3 = standardize(data3)
        data4 = standardize(data4)

        inital_delta = correlation_difference(data1, data2, data3, data4)

        for k in range(n_permutations):
            mod_data1 = data1.copy()
            mod_data2 = data2.copy()
            mod_data3 = data3.copy()
            mod_data4 = data4.copy()

            mask = (np.random.rand(len(data1)) > 0.5)
            mod_data1[mask], mod_data2[mask] = data3[mask], data4[mask]
            mod_data3[mask], mod_data4[mask] = data1[mask], data2[mask]

            delta = correlation_difference(mod_data1, mod_data2, mod_data3, mod_data4)

            if delta > inital_delta:
                cnt += 1
        p_value = cnt/n_permutations
        return p_value


    agg_comp_dict = {a.__name__:{} for a in agg}
    agg_comp_dict_corrected = {a.__name__: {} for a in agg}

    n_comparisons = sum([1 for i in range(len(all_dim1)-1)]) #for j in range(len(all_dim2)) if i != j])

    global_sign = 0.05
    local_sign_corrected = global_sign / n_comparisons

    #Compute c for all combinations of agg in all_dim1 and all_dim2
    for i in range(len(all_dim1)):
        for j in range(len(all_dim2)):
            if i != j:
                agg_1a = np.array(all_dim1[i]).flatten()
                agg_1b = np.array(all_dim2[i]).flatten()
                agg_2a = np.array(all_dim1[j]).flatten()
                agg_2b = np.array(all_dim2[j]).flatten()
                c = permutation_test(agg_1a, agg_1b, agg_2a, agg_2b)
                #c = stats.permutation_test(data, correlation_difference, permutation_type="pairings", alternative="greater")
                agg_comp_dict[agg[i].__name__][agg[j].__name__] = c < global_sign
                agg_comp_dict_corrected[agg[i].__name__][agg[j].__name__] = c < local_sign_corrected
                print(f"p-value for {agg[i].__name__} and {agg[j].__name__}: {c}, significance: {c < global_sign}, corrected (p={local_sign_corrected}): {c < local_sign_corrected}")
            else:
                agg_comp_dict[agg[i].__name__][agg[j].__name__] = False

    # Convert the dictionary to a DataFrame
    agg_comp_df = pd.DataFrame(agg_comp_dict)

    # Convert the dictionary to a DataFrame
    agg_comp_dict_corrected_df = pd.DataFrame(agg_comp_dict_corrected)

    # Sort agg_comp_df by column names and row index
    agg_comp_df = agg_comp_df.sort_index(axis=1)
    agg_comp_df = agg_comp_df.sort_index(axis=0)

    # Sort agg_comp_dict_corrected_df by column names and row index
    agg_comp_dict_corrected_df = agg_comp_dict_corrected_df.sort_index(axis=1)
    agg_comp_dict_corrected_df = agg_comp_dict_corrected_df.sort_index(axis=0)

    # Transform agg_comp_dict_corrected_df to a numpy array and replace all True values with 1 and everything else with 0
    mask = agg_comp_dict_corrected_df.to_numpy()
    mask = np.nan_to_num(mask, nan=1)
    mask = np.where(mask, 1, 0)

    # Set middle diagonal of mask to 0
    mask[np.diag_indices_from(mask)] = 0

    # Plot the DataFrame using seaborn's heatmap
    plt.figure(figsize=(10, 8))
    cmap = mpl.colormaps.get_cmap('coolwarm')
    cmap.set_bad('orange')
    sns.heatmap(agg_comp_df, cmap=cmap, cbar=False, mask=mask, linewidths=0.5)


    # Rotate y-axis labels
    plt.yticks(rotation=90)

    # Add a title
    plt.title(f"Significance for {primary_dimensions[0]} (p < {global_sign})")

    # Position x-axis labels on top
    plt.gca().xaxis.tick_top()

    # Rotate y-axis labels
    plt.yticks(rotation=0)

    # Add a diagonal line
    plt.plot([0, len(agg_comp_df)], [0, len(agg_comp_df)], color='black')
    plt.savefig(join_with_root(f'<PATH_TO_SIGNIFICANCE_HEATMAP_{primary_dimensions[0]}>'))

    # Show the plot
    plt.show()
