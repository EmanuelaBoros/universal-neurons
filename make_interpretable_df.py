from explain import *

torch.set_grad_enabled(False)
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
    args = parser.parse_args()
    model_name = get_model_family(args.model)
    neuron_df = pd.read_csv(f"data/dataframes/neuron_dfs/{model_name}.csv")
    neuron_df["excess_corr"] = neuron_df["mean_corr"] - neuron_df["mean_baseline"]

    save_path = os.path.join("data", "dataframes", "interpretable_neurons", model_name)
    os.makedirs(save_path, exist_ok=True)
    neuron_df.query("excess_corr > 0.5").to_csv(
        os.path.join(save_path, "universal.csv"), index=False
    )
