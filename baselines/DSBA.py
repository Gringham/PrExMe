# Wrapper for the DSBA metric
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

import os, torch, re

import numpy as np

from baselines.MetricClass import MetricClass
from vllm import LLM, SamplingParams
from iterator.utils.vllm_startup_helper import vllm_startup_helper
from iterator.utils.chatTemplates import apply_chat_template


from utils.dsba import HG_SUMMARY_RELEVANCE, HG_SUMMARY_FACTUALITY, HG_SUMMARY_FLUENCY, HG_SUMMARY_COHERENCE


class DSBA(MetricClass):
    name = 'DSBA'

    def __init__(self, model, max_tokens=180, parallel=1):
        self.model = model
        self.llm, self.sampling_params = vllm_startup_helper(model, max_tokens=max_tokens, parallel=parallel)

    def get_score(self, text):
        # Return last number as score
        try:
            found = re.findall("[-+]?(?:\d*\.*\d+)", text)
            return float(found[-1])

        except Exception as e:
             print(e)
             return np.NaN

    def order_scores(self, scores):
        ordered_scores = []
        for i in range(0, len(scores), 4):
            ordered_scores.append({
                "relevance": scores[i],
                "factuality": scores[i+1],
                "fluency": scores[i+2],
                "coherence": scores[i+3]
            })
            ordered_scores[-1]["avg"] = sum(ordered_scores[-1].values()) / 4
        return ordered_scores


    def __call__(self, gt, hyp):
        prompt_list = []
        for g, h in zip(gt, hyp):
            prompt_list.append(HG_SUMMARY_RELEVANCE.format(src=g, hyp=h))
            prompt_list.append(HG_SUMMARY_FACTUALITY.format(src=g, hyp=h))
            prompt_list.append(HG_SUMMARY_FLUENCY.format(src=g, hyp=h))
            prompt_list.append(HG_SUMMARY_COHERENCE.format(src=g, hyp=h))

        input = apply_chat_template(self.model, prompt_list, self.llm, print_samples=True)
        outputs = self.llm.generate(input, self.sampling_params)
        processed_results = [m.outputs[0].text for i, m in enumerate(outputs)]
        processed_results += ['MISSING'] * (len(prompt_list) - len(processed_results))
        scores = [self.get_score(p) for p in processed_results]
        o = self.order_scores(scores)

        return {"scores": [a["avg"] for a in o], "all_scores": o, "texts": processed_results}



if __name__ == '__main__':
    #b = DSBA(model="Open-Orca/OpenOrca-Platypus2-13B")
    #print(b(["A test sentence", "Sentence B"],["So Cummings was told that these units must be preserved in their entirety.", "Satz B"]))#

    #del b.llm
    #del b
    #torch.cuda.empty_cache()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
    b = DSBA(model="TheBloke/Platypus2-70B-Instruct-GPTQ", parallel=2)
    print(b(["A test sentence", "Sentence B"],["So Cummings was told that these units must be preserved in their entirety.", "Satz B"]))

    del b.llm
    del b
    torch.cuda.empty_cache()

    b = DSBA(model="NousResearch/Nous-Hermes-13b")
    print(b(["A test sentence", "Sentence B"],["So Cummings was told that these units must be preserved in their entirety.", "Satz B"]))