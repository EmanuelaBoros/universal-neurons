from analysis.vocab_df import make_vocab_df
import argparse
import os
from transformer_lens import HookedTransformer
import torch
from utils import get_model_family

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-hf",
        help="Name of model from TransformerLens",
    )
    parser.add_argument(
        "--output_dir",
        default="dataframes/vocab_dfs/",
        help="Path to save dataset",
    )

    args = parser.parse_args()
    stat_df = make_vocab_df(args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(args.model, device="cpu")
    model_family = get_model_family(args.model)

    vocab_df = make_vocab_df(model)
    vocab_df.to_csv(os.path.join(args.output_dir, f"{model_family}.csv"))
