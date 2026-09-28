import numpy as np
import pandas as pd
import os
import json
import random
import torch
import torch.nn as nn
from utils.eval_utils import EvalUtils

# Input features (Visual and audio features = total of 28 features with 12 visual & 16 audio features)
IND_xij_U1V_rot = [i for i in range(0, 3)]
IND_xij_U1V_au = [i for i in range(3, 3 + 7)]
IND_xij_U1V_gaze = [i for i in range(3 + 7, 3 + 7 + 2)]
IND_xij_U1V = np.concatenate(
    (IND_xij_U1V_rot, IND_xij_U1V_au, IND_xij_U1V_gaze), axis=0
)  # Head Rotation(x,y,z) + Upper AUs (1,2,4,5,6,7) + Smile(AU12) + Gaze(x,y) of U1
IND_xij_U1A = [i for i in range(12, 28)]
IND_xij_U1 = np.concatenate((IND_xij_U1V, IND_xij_U1A), axis=0)

IND_xij_U2V_rot = [i for i in range(28, 28 + 3)]
IND_xij_U2V_au = [i for i in range(28 + 3, 28 + 3 + 7)]
IND_xij_U2V_gaze = [i for i in range(28 + 3 + 7, 28 + 3 + 7 + 2)]
IND_xij_U2V = np.concatenate(
    (IND_xij_U2V_rot, IND_xij_U2V_au, IND_xij_U2V_gaze), axis=0
)  # Head Rotation(x,y,z) + Upper AUs (1,2,4,5,6,7) + Smile(AU12) + Gaze(x,y) of U2
IND_xij_U2A = [i for i in range(28 + 12, 28 + 28)]
IND_xij_U2 = np.concatenate((IND_xij_U2V, IND_xij_U2A), axis=0)

IND_xij = np.concatenate((IND_xij_U1, IND_xij_U2), axis=0)
IND_xij_inv = np.concatenate((IND_xij_U2, IND_xij_U1), axis=0)

# Output features
IND_yij_U1V_rot = IND_xij_U1V_rot
IND_yij_U1V_au = IND_xij_U1V_au
IND_yij_U1V_gaze = IND_xij_U1V_gaze
IND_yij_U2V_rot = [i for i in range(12, 12 + 3)]
IND_yij_U2V_au = [i for i in range(12 + 3, 12 + 3 + 7)]
IND_yij_U2V_gaze = [i for i in range(12 + 3 + 7, 12 + 3 + 7 + 2)]

IND_yij_U1V = np.concatenate(
    (IND_yij_U1V_rot, IND_yij_U1V_au, IND_yij_U1V_gaze), axis=0
)
IND_yij_U2V = np.concatenate(
    (IND_yij_U2V_rot, IND_yij_U2V_au, IND_yij_U2V_gaze), axis=0
)


class DataGenerator:
    """Data Augmentation"""

    def __init__(self, Xij, yij, batch_size=256, shuffle=True):
        self.Xij = np.copy(Xij)
        self.yij = np.copy(yij)
        self.list_IDs = np.arange(len(self.Xij))
        self.indexes = np.arange(len(self.Xij))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, do_perm):
        "Generates data containing batch_size samples with the option of randomly permutating the sample input feature order (U1&U2 or U2&U1)"
        batch_X = self.Xij[list_IDs_temp, :, :]
        # Generate permutated data (U2&U1)
        if do_perm:
            batch_X[:, :, IND_xij] = batch_X[:, :, IND_xij_inv]
            batch_y = self.yij[list_IDs_temp, :, : int(self.yij.shape[2] / 2)]
        # Generate non-permutated data (U1&U2)
        else:
            batch_y = self.yij[list_IDs_temp, :, int(self.yij.shape[2] / 2) :]
        return torch.from_numpy(batch_X).float(), torch.from_numpy(batch_y).float()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Generate randomly if the batch will be permuted
        # Permutation of user feature indexes (U1 & U2 or U2 & U1; when batch_X:U1&U2 => batch_Y:U2, else batch_X:U2&U1 => batch_Y:U1)
        do_perm = np.random.choice(a=[False, True], p=[0.5, 0.5])
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        batch_X, batch_y = self.__data_generation(list_IDs_temp, do_perm)
        return batch_X, batch_y


def keras_hard_sigmoid(x):
    "Keras (TF2) hard_sigmoid: clip(0.2*x + 0.5, 0, 1)"
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


def init_keras_style(module):
    "Initialize weights like the Keras defaults (glorot_uniform kernels, orthogonal recurrent kernels, zero biases, unit forget bias)"
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for gate in param.data.chunk(4, dim=0):
                        nn.init.xavier_uniform_(gate)
                elif "weight_hh" in name:
                    for gate in param.data.chunk(4, dim=0):
                        nn.init.orthogonal_(gate)
                elif "bias_ih" in name:
                    nn.init.zeros_(param)
                    param.data[m.hidden_size : 2 * m.hidden_size] = 1.0
                elif "bias_hh" in name:
                    nn.init.zeros_(param)


class MultiHeadAttention(nn.Module):
    """Self-Attention Pruning"""

    def __init__(self, input_dim, d_model, num_heads, causal=False, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.causal = causal
        self.w_query = nn.Linear(input_dim, d_model)
        self.w_value = nn.Linear(input_dim, d_model)
        self.w_key = nn.Linear(input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pruning = nn.Linear(num_heads, num_heads)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        "(batch, seq_len, d_model) -> (batch, num_heads, seq_len, depth)"
        x = x.reshape(x.size(0), -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs, mask=None, pruning=False):
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v

        query = self.split_heads(self.w_query(q))
        value = self.split_heads(self.w_value(v))
        key = self.split_heads(self.w_key(k))

        scores = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(1) == 0, float("-inf")
            )
        if self.causal:
            seq_len = scores.size(-1)
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        attention = torch.matmul(self.dropout(torch.softmax(scores, dim=-1)), value)

        if pruning:
            pruning_mask = keras_hard_sigmoid(
                self.pruning(attention.permute(0, 3, 2, 1))
            )
            pruning_mask = torch.round(pruning_mask)
            pruning_mask = pruning_mask.permute(0, 3, 2, 1)
            attention = attention * pruning_mask

        attention = attention.permute(0, 2, 1, 3).reshape(
            attention.size(0), -1, self.d_model
        )

        x = self.dense(attention)

        return x


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ASAPModel(nn.Module):
    """ASAP model"""

    def __init__(self, params_config, params_model):
        super(ASAPModel, self).__init__()
        cell_lstm = params_model["cell_lstm"]
        cell_dense = params_model["cell_dense"]
        cell_att = params_model["cell_multiheadatt"]
        num_head_att = params_model["num_head_multiheadatt"]
        self.pruning_stat = params_model["pruning_stat"]
        nb_inputs = params_config["nb_inputs"]

        self.mha = MultiHeadAttention(
            input_dim=nb_inputs, d_model=cell_att, num_heads=num_head_att
        )
        self.lstm = nn.LSTM(cell_att, cell_lstm, batch_first=True)
        self.dense = nn.Linear(cell_lstm, cell_dense)

        self.output_RotXYZ = nn.Linear(cell_dense, 3)
        self.output_au1_intensity = nn.Linear(cell_dense, 1)
        self.output_au2_intensity = nn.Linear(cell_dense, 1)
        self.output_au4_intensity = nn.Linear(cell_dense, 1)
        self.output_au5_intensity = nn.Linear(cell_dense, 1)
        self.output_au6_intensity = nn.Linear(cell_dense, 1)
        self.output_au7_intensity = nn.Linear(cell_dense, 1)
        self.output_au12_intensity = nn.Linear(cell_dense, 1)
        self.output_GazeXY = nn.Linear(cell_dense, 2)

        init_keras_style(self)
        self.lstm.bias_hh_l0.requires_grad_(False)

    def forward(self, inputs):
        x = self.mha([inputs, inputs, inputs], pruning=self.pruning_stat)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = torch.relu(self.dense(x))

        output_RotXYZ = self.output_RotXYZ(x)
        output_GazeXY = self.output_GazeXY(x)

        return [
            output_RotXYZ[:, 0:1],
            output_RotXYZ[:, 1:2],
            output_RotXYZ[:, 2:3],
            torch.relu(self.output_au1_intensity(x)),
            torch.relu(self.output_au2_intensity(x)),
            torch.relu(self.output_au4_intensity(x)),
            torch.relu(self.output_au5_intensity(x)),
            torch.relu(self.output_au6_intensity(x)),
            torch.relu(self.output_au7_intensity(x)),
            torch.relu(self.output_au12_intensity(x)),
            output_GazeXY[:, 0:1],
            output_GazeXY[:, 1:2],
        ]


def compute_loss(outputs, batch_y, criterion):
    "Sum of the MSE of each output (as Keras compile with one mse loss per output)"
    loss = torch.zeros((), device=batch_y.device)
    for feat, output in enumerate(outputs):
        loss = loss + criterion(output, batch_y[:, -1, feat : feat + 1])
    return loss


class KerasAdam(torch.optim.Optimizer):
    "Adam with the Keras update rule (epsilon added to sqrt(v) before bias correction, unlike torch.optim.Adam)"

    def __init__(self, params, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super(KerasAdam, self).__init__(
            params, dict(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                state["m"].mul_(group["beta_1"]).add_(p.grad, alpha=1 - group["beta_1"])
                state["v"].mul_(group["beta_2"]).addcmul_(
                    p.grad, p.grad, value=1 - group["beta_2"]
                )
                alpha = (
                    group["lr"]
                    * np.sqrt(1 - group["beta_2"] ** state["step"])
                    / (1 - group["beta_1"] ** state["step"])
                )
                p.sub_(alpha * state["m"] / (state["v"].sqrt() + group["epsilon"]))


class ReduceLROnPlateau:
    "Keras ReduceLROnPlateau rules (reduces after `patience` epochs without improvement, unlike torch which waits one more)"

    def __init__(
        self, optimizer, factor=0.2, patience=3, min_delta=1e-4, cooldown=1, min_lr=1e-7
    ):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.best = float("inf")
        self.wait = 0
        self.cooldown_counter = 0

    def on_epoch_end(self, val_loss):
        "Returns the new learning rate if it was reduced, else None"
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.wait = 0
        elif self.cooldown_counter == 0:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = None
                old_lr = self.optimizer.param_groups[0]["lr"]
                if old_lr > np.float32(self.min_lr):
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    for group in self.optimizer.param_groups:
                        group["lr"] = new_lr
                self.cooldown_counter = self.cooldown
                self.wait = 0
                return new_lr
        return None


def build_model(params_config, params_model):
    "Build ASAP model (on the GPU if available, like TensorFlow)"
    model = ASAPModel(params_config, params_model)
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model_tr,
    xij_tr,
    yij_tr,
    xij_val,
    yij_val,
    params_train,
    params_generator,
    dir_path_batchtr="../trainedASAP",
):
    "Train ASAP model"
    device = next(model_tr.parameters()).device
    nb_epoch = params_train["nb_epoch"]
    optimizer = KerasAdam(model_tr.parameters())
    reduce_lr = ReduceLROnPlateau(
        optimizer, factor=0.2, patience=3, cooldown=1, min_lr=1e-7
    )
    early_stop_patience = 50
    criterion = nn.MSELoss()
    os.makedirs(dir_path_batchtr, exist_ok=True)
    # Checkpoint
    dir_weight_path_batchtr = dir_path_batchtr + "/weights"
    os.makedirs(dir_weight_path_batchtr, exist_ok=True)
    checkpoint_name = dir_weight_path_batchtr + "/best_weights-{epoch}.pth"
    # Logger for history
    dir_hist_path_batchtr = dir_path_batchtr + "/histories"
    os.makedirs(dir_hist_path_batchtr, exist_ok=True)
    logger_name = dir_hist_path_batchtr + "/history_log.csv"
    if not os.path.exists(logger_name):
        with open(logger_name, "w") as f:
            f.write("epoch,loss,lr,val_loss\n")
    # Generators
    training_generator = DataGenerator(xij_tr, yij_tr, **params_generator)
    validation_generator = DataGenerator(xij_val, yij_val, **params_generator)
    # Fit
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(nb_epoch):
        print(f"Epoch {epoch+1}/{nb_epoch}")
        lr = optimizer.param_groups[0]["lr"]

        model_tr.train()
        loss_sum = 0.0
        batch_order = list(range(len(training_generator)))
        random.shuffle(batch_order)
        for i in batch_order:
            batch_X, batch_y = training_generator[i]
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model_tr(batch_X), batch_y, criterion)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        train_loss = loss_sum / len(training_generator)

        model_tr.eval()
        loss_sum = 0.0
        with torch.no_grad():
            for i in range(len(validation_generator)):
                batch_X, batch_y = validation_generator[i]
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                loss_sum += compute_loss(model_tr(batch_X), batch_y, criterion).item()
        val_loss = loss_sum / len(validation_generator)

        training_generator.on_epoch_end()
        validation_generator.on_epoch_end()
        print(f" - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - lr: {lr:.4g}")

        # Logs
        with open(logger_name, "a") as f:
            f.write(f"{epoch},{train_loss},{lr},{val_loss}\n")

        # Checkpoint (save best only)
        if val_loss < best_val_loss:
            print(
                f"Epoch {epoch+1}: val_loss improved from {best_val_loss:.5f} to {val_loss:.5f}, saving model to {checkpoint_name.format(epoch=epoch+1)}"
            )
            best_val_loss = val_loss
            torch.save(model_tr.state_dict(), checkpoint_name.format(epoch=epoch + 1))
            epochs_without_improvement = 0
        else:
            print(f"Epoch {epoch+1}: val_loss did not improve from {best_val_loss:.5f}")
            epochs_without_improvement += 1

        # Reduce learning rate on plateau
        new_lr = reduce_lr.on_epoch_end(val_loss)
        if new_lr is not None:
            print(
                f"Epoch {epoch+1}: ReduceLROnPlateau reducing learning rate to {new_lr:.4g}."
            )

        # Early stopping
        if epochs_without_improvement >= early_stop_patience:
            print(f"Epoch {epoch+1}: early stopping")
            break


def save_prediction(pred_dict, dir_predBase_path="../predictions"):
    "Save prediction made by the ASAP model"
    os.makedirs(dir_predBase_path, exist_ok=True)
    pred_name = dir_predBase_path + "/prediction"
    dumped = json.dumps(pred_dict, cls=NumpyEncoder)
    with open(pred_name + ".json", "w") as f:
        f.write(dumped)
    print("Saved Prediction")


def predict_model(model, xij_test, params_config, save=False):
    "Prediction via the ASAP model"
    device = next(model.parameters()).device
    model.eval()
    in_seq_len = params_config["in_seq_len"]
    sample_range_end = len(xij_test)
    pred_dict = {}
    for s in range(0, sample_range_end):
        pred_t_U1_list_s = []
        timestep_range_end = xij_test[s].shape[0] - in_seq_len
        xij_test_t = None
        pred_t_U1 = None
        for t in range(timestep_range_end):
            if t == 0:
                # Flip U1&U2 to predict U1
                xij_test_t_U1 = np.copy(xij_test[s][t : t + in_seq_len, IND_xij_U1])
                xij_test_t_U2 = np.copy(xij_test[s][t : t + in_seq_len, IND_xij_U2])
                xij_test_t = np.concatenate((xij_test_t_U2, xij_test_t_U1), axis=1)
            # Autoregression
            else:
                xij_test_t = np.delete(xij_test_t, [0], axis=0)
                # Use previous prediction as input (rot and au of U1) & Rest as GT (Audio of U1 and All Audio & Vis of U1)
                xij_test_new_t_U1V = np.copy(pred_t_U1)
                xij_test_new_t_U1A = np.copy(
                    xij_test[s][t + in_seq_len - 1, IND_xij_U1A]
                )
                xij_test_new_t_U1 = np.concatenate(
                    (xij_test_new_t_U1V, xij_test_new_t_U1A)
                )
                xij_test_new_t_U2 = np.copy(xij_test[s][t + in_seq_len - 1, IND_xij_U2])
                # Flip U1&U2 to predict U1
                xij_test_new_t = np.concatenate((xij_test_new_t_U2, xij_test_new_t_U1))
                xij_test_new_t = np.reshape(xij_test_new_t, (1, len(xij_test_new_t)))
                xij_test_t = np.vstack((xij_test_t, xij_test_new_t))
            xij_test_t_in = torch.from_numpy(
                np.reshape(
                    xij_test_t, (1, xij_test_t.shape[0], xij_test_t.shape[1])
                ).astype(np.float32)
            ).to(device)
            with torch.no_grad():
                pred_t = model(xij_test_t_in)
            pred_t_U1 = np.array(
                [pred[0, 0].item() for pred in pred_t], dtype=np.float32
            )
            pred_t_U1_list_s.append(pred_t_U1)
        # Save pred of each sample into dictionary
        key_s = "Sample" + str(s)
        pred_dict[key_s] = pred_t_U1_list_s
        print("Sample" + str(s) + " complete")

    if save:
        save_prediction(pred_dict)

    return pred_dict


def evaluate_model(model, xij_test, params_config, data_path, saved_prediction=False):
    "Evaluate the model"
    if not saved_prediction:
        pred_dict = predict_model(model, xij_test, params_config, save=True)
    else:
        pred_dict = load_prediction()

    in_seq_len = params_config["in_seq_len"]
    sample_range_end = len(xij_test)

    # RMSE
    rmse_mean = 0
    for s in range(0, sample_range_end):
        yij_test_U1_s = np.copy(xij_test[s][in_seq_len:, IND_xij_U1V])
        yij_pred_U1_s = np.array(pred_dict["Sample" + str(s)])
        [rmse_mean_s, rmse_all_feat_s] = EvalUtils.rmse_fct(
            yij_test_U1_s, yij_pred_U1_s
        )
        rmse_mean += rmse_mean_s
    rmse_mean /= sample_range_end

    # KS Test
    yij_tr_U1 = load_train_data_labels(data_path)
    yij_tr_U1 = np.reshape(
        yij_tr_U1, (yij_tr_U1.shape[0] * yij_tr_U1.shape[1], yij_tr_U1.shape[2])
    )
    pred_U1 = np.array(pred_dict["Sample0"])
    for s in range(1, len(xij_test)):
        pred_U1_s = np.array(pred_dict["Sample" + str(s)])
        pred_U1 = np.vstack((pred_U1, pred_U1_s))

    ks_mean = 0
    for f in range(pred_U1.shape[1]):
        ks_f = EvalUtils.ks_test(yij_tr_U1, pred_U1, f)
        ks_mean += ks_f[0]
    ks_mean /= pred_U1.shape[1]

    # DTW between ¬PA&PB and Ground Truth(PA&PB) for smile(AU12)
    au12_idx = 9

    feat_list = [
        "RotX",
        "RotY",
        "RotZ",
        "AU1",
        "AU2",
        "AU4",
        "AU5",
        "AU6",
        "AU7",
        "AU12",
        "GazeX",
        "GazeY",
    ]
    dtw_df_real = pd.DataFrame(columns=[feat_list[au12_idx]], dtype=object)
    dtw_df_pred = pd.DataFrame(columns=[feat_list[au12_idx]], dtype=object)
    for s in range(0, sample_range_end):
        au12_real_U1 = np.copy(xij_test[s][in_seq_len:, au12_idx])
        au12_real_U2 = np.copy(xij_test[s][in_seq_len:, len(IND_yij_U1V) + au12_idx])
        au12_pred_U2 = np.array(pred_dict["Sample" + str(s)])[:, au12_idx]

        dtw_au12_real_s = EvalUtils.dtw_fct(au12_real_U1, au12_real_U2)
        dtw_au12_pred_s = EvalUtils.dtw_fct(au12_pred_U2, au12_real_U2)

        dtw_series_real = pd.Series(dtw_au12_real_s, index=dtw_df_real.columns)
        dtw_series_pred = pd.Series(dtw_au12_pred_s, index=dtw_df_pred.columns)
        dtw_df_real = pd.concat(
            [dtw_df_real, dtw_series_real.to_frame().T], ignore_index=True
        )
        dtw_df_pred = pd.concat(
            [dtw_df_pred, dtw_series_pred.to_frame().T], ignore_index=True
        )

    # # Mean of samples
    # DTW of ASAP predictions(¬PA&PB)
    dtw_mean_pred = pd.Series(
        dtw_df_pred[~dtw_df_pred.isin([np.nan, np.inf, -np.inf]).any(axis=1)].mean(
            axis=0
        ),
        index=dtw_df_pred.columns,
    ).values[0]
    # DTW of Ground Truth(PA&PB)
    dtw_mean_gt = pd.Series(
        dtw_df_real[~dtw_df_real.isin([np.nan, np.inf, -np.inf]).any(axis=1)].mean(
            axis=0
        ),
        index=dtw_df_real.columns,
    ).values[0]

    return [rmse_mean, ks_mean, dtw_mean_pred, dtw_mean_gt]


def load_training_data(data_path):
    "Load preprocessed train and validation datasets"
    xij_tr = np.load(
        data_path + "/XijTrain_inseq100_outseq1_stride1.npy",
        allow_pickle=True,
        mmap_mode="r",
    )[0].astype(np.float32)
    yij_tr = np.load(
        data_path + "/YijTrain_inseq100_outseq1_stride1.npy",
        allow_pickle=True,
        mmap_mode="r",
    )[0].astype(np.float32)
    xij_val = np.load(
        data_path + "/XijVal_inseq100_outseq1_stride1.npy",
        allow_pickle=True,
        mmap_mode="r",
    )[0].astype(np.float32)
    yij_val = np.load(
        data_path + "/YijVal_inseq100_outseq1_stride1.npy",
        allow_pickle=True,
        mmap_mode="r",
    )[0].astype(np.float32)

    # Randomize training dataset (shuffle samples)
    np.random.seed(42)
    np.random.shuffle(xij_tr)
    np.random.seed(42)
    np.random.shuffle(yij_tr)

    return [xij_tr, yij_tr, xij_val, yij_val]


def load_test_data(data_path):
    "Load preprocessed test dataset"
    xij_test = np.load(data_path + "/dataTest.npy", allow_pickle=True)

    return xij_test


def load_pretrained_model(
    params_config, params_model, best_weight_path="../trainedASAP/weights"
):
    "Load pretrained ASAP model"
    model_tr = build_model(params_config, params_model)
    checkpoint_filepath = best_weight_path + "/best_weights.pth"
    model_tr.load_state_dict(
        torch.load(
            checkpoint_filepath,
            map_location=next(model_tr.parameters()).device,
            weights_only=True,
        )
    )

    return model_tr


def load_train_data_labels(data_path):
    "Load preprocessed train dataset labels"
    yij_tr = np.load(
        data_path + "/YijTrain_inseq100_outseq1_stride1.npy",
        allow_pickle=True,
        mmap_mode="r",
    )[0].astype(np.float32)
    yij_tr = yij_tr[:, :, IND_yij_U1V]
    return yij_tr


def load_prediction():
    "Load prediction made by the ASAP model"
    dir_predBase_path = "../predictions"
    pred_name = dir_predBase_path + "/prediction"
    with open(pred_name + ".json", "r") as f:
        dumped_pred_dict = f.read()
    pred_dict = json.loads(dumped_pred_dict)
    print("Loaded Prediction")
    return pred_dict
