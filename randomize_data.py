import argparse
import pickle
import random

parser = argparse.ArgumentParser(description="Randomize 500k data labels")

parser.add_argument(
    "--percent", type=int, default=90, help="The percent of labels to randomize."
)

parser.add_argument(
    "--aux_data_filename",
    default="ti_500K_pseudo_labeled.pickle",
    type=str,
    help="Path to pickle file containing unlabeled data and "
    "pseudo-labels used for RST",
)

args = parser.parse_args()


def unpickle(file_name):
    with open(file_name, "rb") as file:
        dict = pickle.load(file, encoding="bytes")
    return dict


def pickle_data(data, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(data, file)


def randomized_targets(percent, aux_targets, num_classes=10):
    return [
        random.randint(0, num_classes - 1)
        for _ in range(int(percent * len(aux_targets)))
    ]


def randomized_indices(percent, aux_targets):
    idxs = [i for i in range(int(percent * len(aux_targets)))]
    random.shuffle(idxs)
    return idxs


def generate_randomized_labels(percent, file_name="ti_500K_pseudo_labeled.pickle"):
    percent /= 100
    data = unpickle(file_name)
    aux_targets = data["extrapolated_targets"]
    rand_targets = randomized_targets(percent, aux_targets)
    rand_idxs = randomized_indices(percent, aux_targets)

    for rand_target, rand_idx in zip(rand_targets, rand_idxs):
        aux_targets[rand_idx] = rand_target

    split_name = file_name.split(".")
    new_file_name = f"{split_name[0]}_{int(percent*100)}.{split_name[1]}"

    data["extrapolated_targets"] = aux_targets
    pickle_data(data, new_file_name)


if __name__ == "__main__":
    generate_randomized_labels(args.percent, args.aux_data_filename)
