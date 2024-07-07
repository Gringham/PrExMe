# Wrapper for BARTScore:
# %%
#@inproceedings{NEURIPS2021_e4d2b6e6,
# author = {Yuan, Weizhe and Neubig, Graham and Liu, Pengfei},
# booktitle = {Advances in Neural Information Processing Systems},
# editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
# pages = {27263--27277},
# publisher = {Curran Associates, Inc.},
# title = {BARTScore: Evaluating Generated Text as Text Generation},
# url = {https://proceedings.neurips.cc/paper/2021/file/e4d2b6e6fdeca3e60e0f1a62fee3d9dd-Paper.pdf},
# volume = {34},
# year = {2021}
#}

from utils.bart_score import BARTScorer
from baselines.MetricClass import MetricClass



class BARTScore(MetricClass):
    name = 'BARTSCORE'

    def __init__(self, batch_size=8, lang='en', *args, **kwargs):
        self.bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        self.batch_size = batch_size


    def __call__(self, gt, hyp):
        return self.bart_scorer.score(hyp, gt, batch_size=self.batch_size)



if __name__ == '__main__':
    b = BARTScore()

    print(sum(p.numel() for p in b.bart_scorer.model.parameters()))
    print(b(["A test sentence", "Sentence B"],["So Cummings was told that these units must be preserved in their entirety.", "Satz B"]))
