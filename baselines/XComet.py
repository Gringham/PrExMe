#A wrapper for XComet-XXL https://huggingface.co/Unbabel/XCOMET-XXL
#@misc{guerreiro2023xcomettransparentmachinetranslation,
#      title={xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection}, 
#      author={Nuno M. Guerreiro and Ricardo Rei and Daan van Stigt and Luisa Coheur and Pierre Colombo and André F. T. Martins},
#      year={2023},
#      eprint={2310.10482},
#      archivePrefix={arXiv},
#      primaryClass={cs.CL},
#      url={https://arxiv.org/abs/2310.10482}, 
#}


import os, sys

from project_root import ROOT_DIR
from baselines.MetricClass import MetricClass

from comet import download_model, load_from_checkpoint

from huggingface_hub import snapshot_download, login

class XComet(MetricClass):
    name = 'XComet'

    def __init__(self, access_token, batch_size=8, cache_dir=None, xcomet_path="<PATH_TO_XCOMET>" *args, **kwargs):
        # Comment in the download part to download xcomet and print the location. Then it can be specified as parameter
        #print("Starting to download XComet")
        #login(token = access_token)
        if cache_dir != None:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
        #self.model_path = snapshot_download(repo_id="Unbabel/XCOMET-XXL", cache_dir=cache_dir)
        #print(self.model_path)
        #self.model = load_from_checkpoint(os.path.join(*[self.model_path, "checkpoints", "model.ckpt"]))
        self.model = load_from_checkpoint(os.path.join(*[f"{xcomet_path}", "checkpoints", "model.ckpt"]))
        self.batch_size = batch_size


    def __call__(self, gt, hyp):
        data = [{"src": g, "mt": h} for g, h in zip (gt, hyp)]
        model_output = self.model.predict(data, batch_size=self.batch_size, gpus=1)

        return model_output.scores



if __name__ == '__main__':
    b = XComet(sys.argv[1])

    print(sum(p.numel() for p in b.model.parameters()))
    print(b(["A test sentence", "Sentence B"],["So Cummings was told that these units must be preserved in their entirety.", "Satz B"]))
