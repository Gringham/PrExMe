import argparse, os, csv
import pandas as pd

from BARTScore import BARTScore
from DSBA import DSBA
from LocalGembaMQM import LocalGembaMQM
from XComet import XComet

if __name__ == '__main__':
    #Apply different baselines to evaluate a dataset and save the scores.

    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    parser = argparse.ArgumentParser(description='Pass allowed models via command line.')
    parser.add_argument('--baseline', help='List of models to be allowed')
    parser.add_argument('--model', help='Model for the DSBA or local MQM baseline', required=False)
    parser.add_argument('--dataset', help='The dataset that should be evaluated. The output will be a single file containing all the scores')
    parser.add_argument('--save_generated', help='Save the generated texts in a file as well')
    parser.add_argument('--src_lang', help='Source language for the local MQM baseline')
    parser.add_argument('--tgt_lang', help='Target language for the local MQM baseline')
    parser.add_argument('--to', required=False, help='Number of rows to process')
    parser.add_argument('--parallel', required=False, help='Number of parallel processes')
    parser.add_argument('--hf_home', help='hf home', required=True, help='Path to the hf home directory')
    parser.add_argument('--tag', required=False, help='Tag for the output file')
    parser.add_argument('--out_dir', required=False, help='Output directory')
    parser.add_argument('--max_tokens', required=False, default=180, help='Maximum number of tokens')
    parser.add_argument('--mode', required=False, default="vllm-instruct", help='Mode for the baseline')
    args = parser.parse_args()

    # Set environment variables
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['HF_HOME'] = args.hf_home

    if ".tsv" in args.dataset:
        df = pd.read_csv(args.dataset, sep="\t", quoting=csv.QUOTE_NONE)
    elif ".json" in args.dataset:
        df = pd.read_json(args.dataset)
    df.SRC = df.SRC.fillna('')
    df.HYP = df.HYP.fillna('')

    if args.to != "None":
        df = df.head(int(args.to))

    if args.baseline == 'BARTScore':
        metric = BARTScore()
        name = args.dataset.replace("/", "_").split(".")[0] + "___" + args.baseline + "___" + args.tag
        pd.DataFrame(metric.evaluate_df(df), columns=[name]).to_json(args.out_dir + "/" + name + ".json")

    if args.baseline == 'DSBA':
        metric = DSBA(model=args.model, parallel=args.parallel, max_tokens=args.max_tokens)
        name = args.dataset.replace("/", "_").split(".")[0] + "___" + args.baseline + "___" + args.model.replace("/", "_")
        res = metric.evaluate_df(df)
        scores = res['scores']
        texts = res['texts']
        pd.DataFrame(scores, columns=[name]).to_json(args.out_dir + "/" + name + ".json")
        if args.save_generated:
            pd.DataFrame(texts, columns=[name]).to_json(args.out_dir + "/" + name + "___generated_texts"+ ".json")

    if args.baseline == 'LocalGembaMQM':
        metric = LocalGembaMQM(model=args.model, parallel=args.parallel, max_tokens=args.max_tokens)
        name = args.dataset.replace("/", "_").split(".")[0] + "___" + args.baseline + "___" + args.model.replace("/", "_")
        scores, texts = metric.evaluate_df(df, args.src_lang, args.tgt_lang)
        pd.DataFrame(scores, columns=[name]).to_json(args.out_dir + "/" + name + ".json")
        if args.save_generated:
            pd.DataFrame(texts, columns=[name]).to_json(args.out_dir + "/" + name + "___generated_texts"+ ".json")

    if args.baseline == 'XComet':
        metric = XComet("")
        name = args.dataset.replace("/", "_").split(".")[0] + "___" + args.baseline
        pd.DataFrame(metric.evaluate_df(df), columns=[name]).to_json(args.out_dir + "/" + name + ".json")
