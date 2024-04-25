from analysis.neuron_df import make_neuron_stat_df
import argparse
import os
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
        default="dataframes/neuron_dfs/",
        help="Path to save dataset",
    )

    args = parser.parse_args()
    model_family = get_model_family(args.model)
    stat_df = make_neuron_stat_df(args.model)

    os.makedirs(args.output_dir, exist_ok=True)
    stat_df.to_csv(os.path.join(args.output_dir, f"{model_family}.csv"))
