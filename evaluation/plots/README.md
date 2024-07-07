### Plotting
Here we describe the files we use to create the plots of our paper. To run them, please replace the paths with `<PATH_TO_...>` in the respective files

#### Heatmaps
`single_column_heatmaps.py` produces a latex optimized heatmap of change in the ranking of dimension A, when dimension B is changed.  
`double_column_heatmaps.py` does the same, but places two heatmaps beneath each other.  
`correlation_heatmaps_per_dim.py` additionally includes significance tests to find the best aggregation measure for the heatmaps.  


#### Pie Charts
`pie_plots_single_optimized.py` and `pie_plots_full.py` both create pie charts. One is more optimized towards generating single figures, while the other is more optimized towards sub figures

#### Latex Tables
`produceLatexTablesOverallScores.py` produces the latex tables presented in our paper