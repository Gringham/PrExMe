# Wrapper for the local Gemba MQM metric
# This is not fully a baseline metric, as it uses local/HF models instead of GPT4
# @inproceedings{kocmi-federmann-2023-gemba-mqm,
#    title = {GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4},
#    author = {Kocmi, Tom  and Federmann, Christian},
#    booktitle = "Proceedings of the Eighth Conference on Machine Translation",
#    month = dec,
#    year = "2023",
#    address = "Singapore",
#    publisher = "Association for Computational Linguistics",
#}

import os
import torch

from baselines.MetricClass import MetricClass
from vllm import LLM, SamplingParams

from iterator.utils.chatTemplates import apply_chat_template
from iterator.utils.vllm_startup_helper import vllm_startup_helper
from utils.mqm_gemba_utils import parse_mqm_answer, apply_template_shared_task_like_mt


class LocalGembaMQM(MetricClass):
    name = 'localGembaMqm'

    def __init__(self, model, parallel, max_tokens=180):
        self.model = model
        self.llm, self.sampling_params = vllm_startup_helper(model, max_tokens=max_tokens, parallel=parallel)

    def __call__(self, gt, hyp, src_lang, hyp_lang):
        prompt_list = [apply_template_shared_task_like_mt(src_lang, hyp_lang, g, h) for g, h in zip(gt, hyp)]
        input = apply_chat_template(self.model, prompt_list, self.llm, print_samples=True)
        outputs = self.llm.generate(input, self.sampling_params)
        processed_results = [m.outputs[0].text for i, m in enumerate(outputs)]
        processed_results += ['MISSING'] * (len(prompt_list) - len(processed_results))
        return ([parse_mqm_answer(p) for p in processed_results], processed_results)

    def evaluate_df(self, df, src_lang, tgt_lang):
        return self.__call__(df['SRC'], df['HYP'], src_lang, tgt_lang)



if __name__ == '__main__':
    b = LocalGembaMQM(model="Open-Orca/OpenOrca-Platypus2-13B")
    print(b(["A test sentence", "Sentence B"],["So Cummings was told that these units must be preserved in their entirety.", "Satz B"], "English", "German"))

    del b
    torch.cuda.empty_cache()

    b = LocalGembaMQM(model="TheBloke/Platypus2-70B-Instruct-GPTQ", parallel=2)
    print(b(["A test sentence", "Sentence B"],["So Cummings was told that these units must be preserved in their entirety.", "Satz B"], "English", "German"))

    del b
    torch.cuda.empty_cache()

    b = LocalGembaMQM(model="NousResearch/Nous-Hermes-13b")
    print(b(["A test sentence", "Sentence B"],["So Cummings was told that these units must be preserved in their entirety.", "Satz B"], "English", "German"))
