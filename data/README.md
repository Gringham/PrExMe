### Datasets

For each phase of our experiments, this folder contains the datasets that we have used or instructions on building them. Where applicable, the associated Licenses are provided within the respective folders. All datasets can be loaded in a uniform format by using the `load_eval_df.py` script. Below, we present usage examples of the script. Each dataframe (at least) has the columns *SRC*, *HYP*, *GT_Score* and *task*, where GT_Score is the human ground truth score. The evaluation scripts work task-wise. Data was acquired using [mt-metrics-eval](https://github.com/google-research/mt-metrics-eval/tree/main/mt_metrics_eval) and [wmt-mqm-human-evaluation](https://github.com/google/wmt-mqm-human-evaluation), [SummEval](https://github.com/Yale-LILY/SummEval), [ROSE](https://github.com/Yale-LILY/ROSE), [Seahorse](https://github.com/google-research-datasets/seahorse), [Eval4NLP 2023](https://github.com/eval4nlp/SharedTask2023). More details below:
#### Phase 1: Eval4NLP 2023 Train
The "train" set of the Eval4NLP 2023 shared task (`data/train`). It is a split of:

* **MT**: The MQM en-de and zh-en language pairs of the WMT 22 metrics shared task, a work by `Freitag et. al. (2022), Results of WMT22 Metrics Shared Task: Stop Using BLEU -- Neural Metrics Are Better and More Robust. In: WMT22`
* **Summarization**: The average aspect score of SummEval, a work by `Fabbri et. al. (2020), SummEval: Re-evaluating Summarization Evaluation. In: Transactions of the Association for Computational Linguistics`

Loading with `load_eval_df.py`:
```python
df = load_train_df()
print(df)

#                                                     SRC                                                HYP  ...  GT_Score           task
#0      Then Dominic Cummings, once Johnson's closest ...  Dann versprach Dominic Cummings, einst Johnson...  ... -0.000000          en_de
#...                                                  ...                                                ...  ...       ...            ...
#27111  New Zealand police are appealing to the public...  new zealand police are appealing to the public...  ...  4.916667  summarization
```


#### Phase 2: Eval4NLP 2023 Dev
The "dev" set of the Eval4NLP 2023 shared task (`data/dev`). It is a split of:

* **MT**: The MQM en-de and zh-en language pairs of the WMT 22 metrics shared task, a work by `Freitag et. al. (2022), Results of WMT22 Metrics Shared Task: Stop Using BLEU -- Neural Metrics Are Better and More Robust. In: WMT22`
* **Summarization**: The average aspect score of SummEval, a work by `Fabbri et. al. (2020), SummEval: Re-evaluating Summarization Evaluation. In: Transactions of the Association for Computational Linguistics`

Loading with `load_eval_df.py`:
```python
df = load_dev_df()
print(df)

#    SRC                                                HYP  ...  GT_Score           task
#0      Once this chat has ended you will be sent a 'r...  Sobald dieser Chat beendet ist, erhalten Sie e...  ... -0.000000          en_de
#...                                                  ...                                                ...  ...       ...            ...
#19139  Nathan Hughes on Friday night had his ban for ...  nathan hughes 's knee collided with george nor...  ...  3.666667  summarization
```

#### Phase 2: Eval4NLP 2023 Test
Please contact the authors if you are interested in the dataset.

#### Phase 2: WMT23 & Seahorse
A dataset (`data/new_test`) constructed from:

* **MT**: The MQM he-en, en-de and zh-en language pairs of the WMT 23 metrics shared task, a work by `Freitag et. al. (2023), Results of WMT23 Metrics Shared Task: Metrics Might Be Guilty but References Are Not Innocent, In: WMT23`
* **Summarization**: The ratio of positive/negative answered questions in the Seahorse summarization dataset by `Elizabeth Clark et. al. (2023), SEAHORSE: A Multilingual, Multifaceted Dataset for Summarization Evaluation, In: EMNLP 2023`

Loading with `load_eval_df.py`:
```python
df = load_test_df("wmt_23_seahorse")
print(df)

#                    GT_Score system-name  ...           task                      DOC
#0                     -1.0        AIRC  ...          en_de   news\taj-english.33941
#...                    ...         ...  ...            ...                      ...
#18325                  0.8   finetuned  ...  summarization  xlsum_english-test-7423  
```



#### Datasets for retrieval augmented generation (RAG) - WMT 2021 & ROSE: 
The retrieval dataset (`data/dict`) is constructed from:

* **MT**: The MQM en-de and zh-en language pairs of the 2021 MQM annotations of `Freitag et. al. (2021), Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation, In: TACL 2021`
* **Summarization**: The overlap of atomic content units (ACUs) in ROSE by `Yixin Liu et. al. (2023, Revisiting the Gold Standard: Grounding Summarization Evaluation with Robust Human Evaluation, In: ACL 2023)`

Loading with `load_eval_df.py`:
```python
df = load_retrieval_df()
print(df)

#       Unnamed: 0              system  GT_Score  seg_id                                                SRC  ...     LP    REF SYSTEM           task seg-id
#0               0         Facebook-AI      -5.0     1.0  Couple MACED at California dog park for not we...  ...  en-de  dummy  dummy          en_de    NaN
#...           ...                 ...       ...     ...                                                ...  ...    ...    ...    ...            ...    ...
#23128        3995               cliff       0.0     NaN  "On a visit to Japan, the prime minister welco...  ...   None    NaN    NaN  summarization   None
```