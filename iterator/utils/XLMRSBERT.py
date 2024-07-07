import pytorch_lightning as pl
from more_itertools import chunked

from transformers import AutoTokenizer
from transformers import AutoModel

import torch


class XLMRSBERT():
    # Wrapper for XLMR SBERT. We use it for RAG
    ref_based = False
    name = 'XLMR'

    def __init__(self, bs=16):
        super().__init__()

        self.model = model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
        embeddings = model.encode(sentences)
        self.bs = bs

    def __call__(self, src, hyp):
        '''
        :param ref: A list of strings with source sentences
        :param hyp: A list of strings with hypothesis sentences
        :return: A list of Labse Scores
        '''
        self.eval()

        src_ids = self.tokenize(src)
        hyp_ids = self.tokenize(hyp)

        self.to('cuda:0')
        scores = []

        # Unfortunately I didn't find existing batching here (so I'd suppose it should be there)
        # Therefore I use some other lib for batching
        for x in chunked(range(src_ids['input_ids'].shape[0]), self.bs):
            s_in_batch = src_ids['input_ids'][x].cuda()
            h_in_batch = hyp_ids['input_ids'][x].cuda()
            s_att_batch = src_ids['attention_mask'][x].cuda()
            h_att_batch = hyp_ids['attention_mask'][x].cuda()

            scores += self.forward(s_in_batch, h_in_batch,
                               s_att_batch, h_att_batch).tolist()

        return scores

    def tokenize(self, sent):
        return self.tokenizer(sent, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        # pooling step from here: https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, src_ids, hyp_ids, src_mask, hyp_mask):
        src_model_out = self.hyp_model(input_ids=src_ids, attention_mask=src_mask)
        src_emb = self.mean_pooling(src_model_out, src_mask)

        hyp_model_out = self.src_model(input_ids=hyp_ids, attention_mask=hyp_mask)
        hyp_emb = self.mean_pooling(hyp_model_out, hyp_mask)
        cos_sim = self.cos(src_emb, hyp_emb)

        return cos_sim


if __name__ == '__main__':
    c = XlmrCosineSim()

    # Sample using ref and hyp lists
    # [0.9784649610519409]
    print(c(["Ein Test Satz"], ["A test sentence"]))

    # Sample using a fixed reference for a list of hypothesis
    # [0.7363132834434509, 0.7302700281143188, 0.9784649014472961]
    c_trimmed = c.get_abstraction("Ein Test Satz")
    print(c_trimmed(["A simple sentence for test", "Another simple sentence for test", 'A test sentence']))
