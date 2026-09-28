import torch
from utils.ASAP_utils import load_training_data, build_model, train_model

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device found : {device}")


# Params for Config
params_config = {
    "nb_inputs": 56,
    "in_seq_len": 100,
}
# Params for Training
params_train = {
    "nb_epoch": 1000,
}
# Params for Model
params_model = {
    "cell_multiheadatt": 16 * 4,
    "num_head_multiheadatt": 4,
    "pruning_stat": True,
    "cell_lstm": 20,
    "cell_dense": 20,
}
# Params for Generators
params_generator = {"batch_size": 32, "shuffle": True}


if __name__ == "__main__":
    """
    Training of ASAP model
    """
    data_path = "../data"

    [xij_tr, yij_tr, xij_val, yij_val] = load_training_data(data_path)

    model_tr = build_model(params_config, params_model)
    train_model(
        model_tr, xij_tr, yij_tr, xij_val, yij_val, params_train, params_generator
    )
