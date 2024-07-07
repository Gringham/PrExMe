## Baselines

This folder contains wrappers for the baseline metrics (and `LocalGembaMQM`). These ensure that each metric can be called in the same way. Each wrapper contains a usage example in its main section. For example, `BARTScore` can be called as follows:

```python
b = BARTScore()

# b(<List of ground truth texts>, <List of hyporthesis texts>)
print(b(["A test sentence", "Sentence B"],["So Cummings was told that these units must be preserved in their entirety.", "Satz B"]))
```

### Applying all baselines
The cli tool `apply_baselines.py` can be used to apply the baselines to datasets that are presented in *.tsv* or *.json* formats. The datasets should have a *SRC* and a *HYP* column for reference-free evaluation. Here are some usage examples: 

```sh
# These scripts will save the computed scores into the --out_dir folder. For LLM based baselines, the generated text can be saved when setting the --save_generated flag to True

# Apply BARTScore to the en_de dev set. The parameters max_tokens, mode, parallel and save_generated have no effect for baselines that are not based on LLMs
python3 baselines/apply_baselines.py --model None --to None --max_tokens 180 --dataset data/dev/dev_en_de.tsv --mode vllm-instruct --tag dev_en_de --parallel None --hf_home <HF_HOME> --out_dir outputs/baselines --src_lang English --tgt_lang German --baseline BARTScore --save_generated True

# Apply LocalGembaMQM to the en_de dev set
python3 baselines/apply_baselines.py --model TheBloke/Platypus2-70B-Instruct-GPTQ --to None --max_tokens 180 --dataset data/dev/dev_en_de.tsv --mode vllm-instruct --tag dev_en_de --parallel 2 --hf_home <HF_HOME> --out_dir outputs/baselines --src_lang English --tgt_lang German --baseline LocalGembaMQM --save_generated True
```

The descriptions of each parameter can be found in the script. 