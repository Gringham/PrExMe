import os
from numpy import pad
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

# Set seaborn style for a more polished look
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize': (3.15, 3)})

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
    
    # Create a color mapping for the value_column using a seaborn color palette
    unique_labels = df[value_column].unique()
    colors = sns.color_palette("colorblind", len(unique_labels))
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    fig, axes = plt.subplots(1, len(groups), figsize=(8, 4))
    
    for ax, group in zip(axes, groups):
        group_df = df[df[group_column] == group]
        
        # Combine top 10% Kendall rows for each unique task within the group
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
        ax.annotate(f"{group}", xy=(0.50, 0.926), xycoords='axes fraction', 
                    ha='center', fontsize=11, weight='bold')

    # Create a common legend
    legend = fig.legend(wedges, labels, title="Format Requirement", fontsize=9, edgecolor='black', bbox_to_anchor=(0.755, 0.255), ncol=3)
    #legend = fig.legend(wedges, labels, title="Base Prompt", fontsize=9, edgecolor='black', bbox_to_anchor=(0.70, 0.265), ncol=3)

    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 0.8))
    legend._legend_box.align = "left"

    #rect = plt.Rectangle(
    #    (0.14, 0.18), 0.72, 0.79, transform=fig.transFigure, linewidth=2, edgecolor='black', facecolor='none'
    #)
    #fig.patches.append(rect)

    
    plt.subplots_adjust(wspace=-0.52)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"<PATH_TO_OUTPUT_DIR>/combined_pie_plots_{group_column}_{value_column}_{filter_column}.pdf", bbox_inches='tight', transparent=True)
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

df1 = df1[(df1["model"]=="OpenOrca-13B")|(df1["model"]=="Tower-13B")]
df1["regex2"] = [str(r["name"]) if type(r)==dict else r for r in df1["regex"].tolist()]

print(df1.columns)
# Generate combined pie charts
generate_combined_pie_charts(df1, group_column="model", value_column="regex2", filter_column="kendall", task_column="task")
