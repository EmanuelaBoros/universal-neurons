import os
import torch
import argparse
import datasets
from transformer_lens import HookedTransformer
from analysis.vocab_df import create_normalized_vocab, get_unigram_df
from analysis.activations import make_dataset_df
from analysis.heuristic_explanation import *
from utils import get_model_family
from tqdm import tqdm


def run_and_save_token_explanations(
    activation_df, feature_df, neuron_cols, save_path, feature_type
):
    var_red_df = compute_feature_variance_reduction_df(
        activation_df, feature_df, neuron_cols, feature_type=feature_type
    )
    # mean_dif_df = compute_mean_dif_df(
    #     activation_df, feature_df, neuron_cols)

    prev_token_var_red_df = compute_feature_variance_reduction_df(
        activation_df,
        feature_df,
        neuron_cols,
        feature_type=feature_type,
        prev_token=True,
    )
    # prev_token_mean_dif_df = compute_mean_dif_df(
    #     activation_df, feature_df, neuron_cols, prev_token=True)

    var_red_df.to_csv(os.path.join(save_path, "variance_reduction.csv"))
    prev_token_var_red_df.to_csv(
        os.path.join(save_path, "prev_token_variance_reduction.csv")
    )


def make_activation_df(
    dataset_df, activation_path, model_name, dataset_name, layer, neurons, use_post=True
):
    activation_df = dataset_df.copy()
    neuron_cols = []
    new_df = pd.DataFrame()
    for ix, (l, n) in tqdm(enumerate(neurons), total=len(neurons)):

        if l != layer and layer != -1:  # if l==-1, we want all layers
            # 0.all.none.all.pt
            continue
        elif layer == -1:
            activations = torch.load(
                os.path.join(
                    activation_path, model_name, dataset_name, f"{l}.all.none.all.pt"
                )
            )
        else:
            activations = torch.load(
                os.path.join(activation_path, model_name, dataset_name, f"{l}.{n}.pt")
            )
        if use_post:
            try:
                activations = torch.nn.GELU()(activations.float()).numpy()
            except:
                activations = torch.nn.GELU()(activations.dequantize().float()).numpy()

        # activations = activations.dequantize()
        mean_activations = activations.reshape(activations.shape[0], -1).mean(axis=1)
        col = f"{l}.{n}"

        # activation_df[col] = mean_activations  # .numpy().flatten()

        activation_df.insert(activation_df.shape[1], col, mean_activations, True)

        neuron_cols.append(col)

    return activation_df, neuron_cols


def make_full_token_df(activation_df, decoded_vocab, model_family):
    vocab_df = pd.read_csv(f"dataframes/vocab_dfs/{model_family}.csv")
    vocab_df.loc[vocab_df.token_string.isna(), "token_string"] = "n/a"

    decoded_norm_vocab, token_ix_2_normed_ix = create_normalized_vocab(
        vocab_df, decoded_vocab
    )

    unigram_df = get_unigram_df(activation_df, decoded_norm_vocab, token_ix_2_normed_ix)

    if os.path.exists(f"dataframes/vocab_dfs/{model_family}_topics.csv"):
        topic_df = pd.read_csv(f"dataframes/vocab_dfs/{model_family}_topics.csv")
        feature_df = pd.concat([unigram_df, vocab_df, topic_df], axis=1)
    else:
        feature_df = pd.concat([unigram_df, vocab_df], axis=1)

    return feature_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="stanford-gpt2-medium-a")
    parser.add_argument("--dataset", type=str, default="pile.test.all-10m.512")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument(
        "--neuron_df_path",
        type=str,
        default="dataframes/interpretable_neurons/stanford-gpt2-medium-a/universal.csv",
    )
    parser.add_argument("--activation_path", type=str, default="cached_activations")
    parser.add_argument("--feature_type", type=str, default="token")

    args = parser.parse_args()
    model_name = args.model
    model_family = get_model_family(model_name)

    neuron_df = pd.read_csv(args.neuron_df_path)
    neurons = neuron_df[["layer", "neuron"]].values

    # ds = datasets.load_from_disk(
    #     os.path.join(
    #         os.getenv("DATASET_DIR", "token_datasets"), model_family, args.dataset
    #     )
    # )
    ds = datasets.load_from_disk(
        os.path.join("data", "token_datasets", model_family, args.dataset)
    )

    model = HookedTransformer.from_pretrained(model_name)

    decoded_vocab = {
        tix: model.tokenizer.decode(tix) for tix in model.tokenizer.get_vocab().values()
    }
    dataset_df = make_dataset_df(ds, decoded_vocab)
    activation_df, neuron_cols = make_activation_df(
        dataset_df,
        args.activation_path,
        model_family,
        args.dataset,
        args.layer,
        neurons,
    )

    if args.feature_type == "token":
        feature_df = make_full_token_df(activation_df, decoded_vocab, model_family)
    elif args.feature_type == "sequence":
        save_path = os.path.join(
            "dataframes",
            "dataset_dfs",
            # model_name.replace("small", "medium"),
            model_family,
            args.dataset,
        )
        feature_df = pd.read_pickle(os.path.join(save_path, "dataset.p"))

    save_path = os.path.join(
        # os.getenv("RESULTS_DIR", "results"),
        "data",
        "results",
        "explanations",
        model_family,
        args.dataset,
        args.feature_type + "_feature",
        f"layer_{args.layer}",
    )
    os.makedirs(save_path, exist_ok=True)

    run_and_save_token_explanations(
        activation_df, feature_df, neuron_cols, save_path, args.feature_type
    )
