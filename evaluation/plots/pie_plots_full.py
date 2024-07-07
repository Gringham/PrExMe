import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

# Set seaborn style for a more polished look
sns.set(style="white")
sns.set(rc={'figure.figsize': (6.3, 3)})

# Mapping for label replacements
label_replacements = {
    "complex token labels": "complex labels",
    "easy token labels": "simple labels",
    "Zero-Shot": "PZS",
    "Zero-Shot-Cot": "ZS-CoT",
    "Zero-Shot-Cot-Emotion": "ZS-CoT-EM"
}

def generate_combined_pie_charts(df, group_column, value_column, filter_column, task_column):
    # Get unique groups
    groups = df[group_column].unique()
    
    # Create a color mapping for the value_column using a more varied seaborn color palette
    unique_labels = df[value_column].unique()
    colors = sns.color_palette("tab20", len(unique_labels))
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # Calculate number of rows and columns needed for subplots
    n_groups = len(groups)
    n_cols = 4
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    all_wedges = []
    all_labels = []

    title_dict = {
        "zero_shot_train_sign.json": "ZS - Eval4NLP Train",
        "zero_shot_dev_sign.json": "ZS - Eval4NLP Dev",
        "zero_shot_test_sign.json": "ZS - Eval4NLP Test",
        "zero_shot_test2_sign.json": "ZS - WMT23/Seahorse",
        "few_shot_train_sign.json": "OS - Eval4NLP Train",
        "few_shot_dev_sign.json": "OS - Eval4NLP Dev",
        "few_shot_test_sign.json": "OS - Eval4NLP Test",
    }

    if group_column == "filename":
        groups = title_dict.keys()

    for group, ax in zip(groups, axes):
        group_df = df[df[group_column] == group]

        if group == '' or "baseline" in group:
            fig.delaxes(ax)
            continue
        
        # Combine top 2% Kendall rows
        top_kendall_df = pd.DataFrame()
        unique_tasks = group_df[task_column].unique()
        
        for task in unique_tasks:
            task_df = group_df[group_df[task_column] == task]
            threshold = task_df[filter_column].quantile(0.98)
            top_kendall_df = pd.concat([top_kendall_df, task_df[task_df[filter_column] >= threshold]])
        
        counts = top_kendall_df[value_column].value_counts()
        
        # Replace labels as per the mapping
        labels = [label_replacements.get(label, label) for label in counts.index]
        
        # Map the colors using the color_map
        pie_colors = [color_map[label] for label in counts.index]
        
        wedges, texts, autotexts = ax.pie(counts, labels=None, autopct='%1.1f%%', colors=pie_colors,
                                          textprops={'fontsize': 9, 'fontweight': 'bold'})
        
        circle_border = patches.Circle((0, 0), 0.99, transform=ax.transData, edgecolor='black', linewidth=0.8, facecolor='none', zorder=10)
        ax.add_patch(circle_border)
        
        # Rotate the percentages to match the direction of the wedges
        for autotext, wedge in zip(autotexts, wedges):
            angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
            
            if wedge == wedges[0]:  # The largest wedge is always the first one
                autotext.set_rotation(0)  # Ensure the largest wedge text is not rotated
                x, y = autotext.get_position()
                autotext.set_position((x, y - 0.1))  # Move the text down slightly
            else:
                if 90 < angle < 270:  # Wedges on the left side
                    autotext.set_rotation(angle + 180)
                else:
                    autotext.set_rotation(angle)
            
            autotext.set_rotation_mode('anchor')

        # Add title below the pie chart
        if group in title_dict:
            ax.annotate(f"{title_dict[group]}", xy=(0.50, 1), xycoords='axes fraction', 
                        ha='center', fontsize=11, weight='bold')
        else:
            ax.annotate(f"{group}", xy=(0.50, 0.926), xycoords='axes fraction', 
                        ha='center', fontsize=11, weight='bold')

        all_wedges.extend(wedges)
        all_labels.extend(labels)
        ax.grid(False)
        ax.axis('equal')  # Ensure pie chart is circular
        ax.axis('off')    # Hide the axes

    # Hide any unused subplots
    for ax in axes[len(groups):]:
        fig.delaxes(ax)
    
    # Create a common legend
    unique_labels = list(dict.fromkeys(all_labels))  # Ensure the labels are unique while preserving order
    unique_wedges = [all_wedges[all_labels.index(label)] for label in unique_labels]
    legend = fig.legend(unique_wedges, unique_labels, title="Base Prompt", fontsize=9, edgecolor='black', bbox_to_anchor=(0.56, 0.03), ncol=5)

    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.8))
    legend._legend_box.align = "left"

    plt.subplots_adjust(wspace=-0.75, hspace=-0.2)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"<PATH_TO_PIE_CHART_OUTPUTS>/combined_pie_plots_{group_column}_{value_column}_{filter_column}_all.pdf", bbox_inches='tight', transparent=True)
    plt.close()

# Example usage:
folder = "<PATH_TO_FOLDER_WITH_CORRELATION_JSON_FILES>"

# Read and concatenate all json files to pandas dataframe
df_list = []
for file in os.listdir(folder):
    if file.endswith(".json"):
        df = pd.read_json(os.path.join(folder, file))
        df["filename"] = file
        df["mode"] = "fs" if "few-shot" in file.lower() else "zero-shot"
        df_list.append(df)
df1 = pd.concat(df_list, ignore_index=True)

# Filter out summarization tasks and handle regex column
df1 = df1[(df1["task"] != "summarization")]
df1["regex2"] = [str(r["name"]) if isinstance(r, dict) else r for r in df1["regex"].tolist()]

# Generate combined pie charts
generate_combined_pie_charts(df1, group_column="filename", value_column="task_description", filter_column="kendall", task_column="model")
