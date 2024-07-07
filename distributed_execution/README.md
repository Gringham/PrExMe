### Distributed Execution
The scripts in this folder aid the distributed execution of the hierarchical prompts and baselines. 
`produce_commands.py` will write a shell script that lists python commands that execute prompts for a specific model, task and dataset. 
It will automatically split the prompts, if the range of samples becomes too large. 

The resulting shell scripts can be built into a a slurm task array script using the `generate_pool_script2.py` script.
The paths with `<PATH_...>` need to be adapted. Depending on your cluster, further changes may be necessary