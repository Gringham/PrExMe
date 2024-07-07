# Prompts by
#@inproceedings{kim-etal-2023-better,
#    title = "Which is better? Exploring Prompting Strategy For {LLM}-based Metrics",
#    author = "Kim, JoongHoon  and
#      Lee, Sangmin  and
#      Hun Han, Seung  and
#      Park, Saeran  and
#      Lee, Jiyoon  and
#      Jeong, Kiyoon  and
#      Kang, Pilsung",
#    editor = {Deutsch, Daniel  and
#      Dror, Rotem  and
#      Eger, Steffen  and
#      Gao, Yang  and
#      Leiter, Christoph  and
#      Opitz, Juri  and
#      R{\"u}ckl{\'e}, Andreas},
#    booktitle = "Proceedings of the 4th Workshop on Evaluation and Comparison of NLP Systems",
#    month = nov,
#    year = "2023",
#    address = "Bali, Indonesia",
#    publisher = "Association for Computational Linguistics",
#    url = "https://aclanthology.org/2023.eval4nlp-1.14",
#    doi = "10.18653/v1/2023.eval4nlp-1.14",
#    pages = "164--183",
#}
# We thank Juri Opitz for the implementation

HG_SUMMARY_RELEVANCE = """Instruction:
In this task you will evaluate the quality of a summary written for a document.

To correctly solve this task, follow these steps:
1. Carefully read the document, be aware of the information it contains.
2. Read the proposed summary.
3. Rate each summary on a scale from 0 (worst) to 100 (best) by its Relevance.

# Definition:
Relevance: The rating measures how well the summary captures the key points of the article.
Consider whether all and only the important aspects are contained in the summary.

Source text: {src}

Summary: {hyp}

Score: """


HG_SUMMARY_FACTUALITY= """Instruction:
In this task you will evaluate the quality of a summary written for a document.

To correctly solve this task, follow these steps:
1. Carefully read the document, be aware of the information it contains.
2. Read the proposed summary.
3. Rate each summary on a scale from 0 (worst) to 100 (best) by its Factuality.

# Definition:
Factuality: This rating gauges the accuracy and truthfulness of the information presented
in the summary compared to the original article.
Scrutinize the summary to ensure it presents facts without distortion or misrepresentation,
staying true to the source content’s details and intent.

Source text: {src}

Summary: {hyp}

Score: """


HG_SUMMARY_FLUENCY = """Instruction:
In this task you will evaluate the quality of a summary written for a document.

To correctly solve this task, follow these steps:
1. Carefully read the document, be aware of the information it contains.
2. Read the proposed summary.
3. Rate each summary on a scale from 0 (worst) to 100 (best) by its Fluency.

# Definition:
Fluency: This rating evaluates the clarity and grammatical integrity of each sentence in the summary.
Examine each sentence for its structural soundness and linguistic clarity.

Source text: {src}

Summary: {hyp}

Score: """

HG_SUMMARY_COHERENCE = """Instruction:
In this task you will evaluate the quality of a summary written for a document.

To correctly solve this task, follow these steps:
1. Carefully read the document, be aware of the information it contains.
2. Read the proposed summary.
3. Rate each summary on a scale from 0 (worst) to 100 (best) by its Coherence.

# Definition:
Coherence: This rating evaluates how well the parts of the summary  (phrases, sentences, etc.) fit together in a natural or reasonable way.

Source text: {src}

Summary: {hyp}

Score: """