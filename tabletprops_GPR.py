import random, torch, joblib
import numpy as np
import pandas as pd
from keras.src.layers import BatchNormalization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os

SEED = 7
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

# Load stage 1 models
stage1_reg_model = tf.keras.models.load_model('final_stage1_regression_model.keras')
stage1_split_model = tf.keras.models.load_model('final_stage1_split_model.keras')

gpr = joblib.load('gprR.pkl')
scaler_X = joblib.load('gpr_scalerR.pkl')
gpc = joblib.load('gpcR.pkl')

df = pd.read_csv('ribbontotablet_splittingdata.csv')
X = np.array(df[['Roll Gap', 'Roll Pressure', 'Roll Speed',	'Screw Feed Speed',	'API', 'First Run']])

# Scale Stage-2 inputs
X_scaled = scaler_X.transform(X)

# GPR ribbon predictions
ribbon_preds = stage1_reg_model.predict(X_scaled)
#ribbon_preds, ribbon_std = gpr.predict(X_scaled, return_std=True)

pred_density = ribbon_preds[:,0]
pred_thickness = ribbon_preds[:,1]

# GPC split probabilities
split_probs = stage1_split_model.predict(X_scaled)
#split_probs = gpc.predict_proba(X_scaled)[:,1]

# LOAD FINAL STAGE-2 ENSEMBLE
stage2_ensemble = []
N_ENSEMBLE = 10

for i in range(N_ENSEMBLE):

    model_i = tf.keras.models.load_model(f"final_stage2_model_{i}.keras")
    stage2_ensemble.append(model_i)

# LOAD STAGE-2 SCALERS
scaler_stage2 = joblib.load('stage2_scaler.pkl')
scaler_Y_stage2 = joblib.load('stage2_output_scaler.pkl')

# Stage-2 feature matrix
X_stage2 = np.column_stack([pred_density, pred_thickness, split_probs, X[:,4]])
X_stage2_scaled = scaler_stage2.fit_transform(X_stage2)

# STAGE-2 ENSEMBLE PSD PREDICTIONS
stage2_preds = []

for model_i in stage2_ensemble:

    pred_scaled = model_i.predict(X_stage2_scaled, verbose=0)
    pred_real = scaler_Y_stage2.inverse_transform(pred_scaled)
    stage2_preds.append(pred_real)

stage2_preds = np.array(stage2_preds)

# MEAN + UNCERTAINTY
psd_mean = np.mean(stage2_preds, axis=0)
psd_std = np.std(stage2_preds, axis=0)

pred_d10 = psd_mean[:,0]
pred_d50 = psd_mean[:,1]
pred_d90 = psd_mean[:,2]

sigma_d10 = psd_std[:,0]
sigma_d50 = psd_std[:,1]
sigma_d90 = psd_std[:,2]

# FINAL STAGE-3 INPUTS
X_TP = np.array(df['Main Comp. Thick.'])
X_stage3 = np.column_stack([pred_density, pred_thickness, split_probs,
                            pred_d10, pred_d50, pred_d90,
                            #sigma_d10, sigma_d50, sigma_d90,
                            X_TP])

Y_tablet = np.array(df[['Weight', 'Elastic Recovery', 'Relative Density', 'Tensile Strength']])

# Scale Stage-3 inputs and outputs
scaler_stage3 = StandardScaler()
X_scaled_s3 = scaler_stage3.fit_transform(X_stage3)

scaler_Y_pre_s3 = StandardScaler()
Y_scaled_s3 = scaler_Y_pre_s3.fit_transform(Y_tablet)


# GP kernels
kernel3 = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gpr_stage3 = GaussianProcessRegressor(kernel=kernel3, alpha=1e-4, normalize_y=True)

gpr_stage3.fit(X_scaled_s3, Y_scaled_s3)

# Predictions on the same input data
Y_pred, sigma = gpr_stage3.predict(X_scaled_s3, return_std=True)

# Evaluate model performance
mae = mean_absolute_error(Y_scaled_s3, Y_pred)
mse = mean_squared_error(Y_scaled_s3, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_scaled_s3, Y_pred)

# Print metrics
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

r2_W = r2_score(Y_scaled_s3[:, 0], Y_pred[:, 0])
r2_ER = r2_score(Y_scaled_s3[:, 1], Y_pred[:, 1])
r2_RD = r2_score(Y_scaled_s3[:, 2], Y_pred[:, 2])
r2_TS = r2_score(Y_scaled_s3[:, 3], Y_pred[:, 3])

print(f"R² Score - W: {r2_W:.4f}")
print(f"R² Score - ER: {r2_ER:.4f}")
print(f"R² Score - RD: {r2_RD:.4f}")
print(f"R² Score - TS: {r2_TS:.4f}")


# Stage-3 Data Augmentation
n_augments = 10

X_aug = []
Y_aug = []

print("Generating synthetic Stage-3 samples...")

for i in range(len(X_scaled_s3)):
    x = X_scaled_s3[i]
    y_mean, y_std = gpr_stage3.predict([x], return_std=True)
    for _ in range(n_augments):
        y_sample = np.random.normal(loc=y_mean.ravel(), scale=y_std)
        X_aug.append(x)
        Y_aug.append(y_sample)

X_aug = np.array(X_aug)
Y_aug = np.array(Y_aug).reshape(len(Y_aug), len(Y_aug[0][0]))

# Combine real + synthetic
X_total = np.vstack([X_scaled_s3, X_aug])
Y_total = np.vstack([Y_scaled_s3, Y_aug])
print(X_total.shape)
print(Y_total.shape)


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# STAGE-3 ENSEMBLE LEARNING + TRANSFER LEARNING
# SYNTHETIC DATASET ONLY

X_syn = X_aug
Y_syn = Y_aug

print("\nSynthetic dataset shape:")
print(X_syn.shape)
print(Y_syn.shape)


# Train / val split for pretraining
X_syn_train, X_syn_val, Y_syn_train, Y_syn_val = train_test_split(X_syn, Y_syn, test_size=0.2, random_state=SEED, shuffle=True)

# MODEL ARCHITECTURE
def build_stage31_model(input_dim):
    model = Sequential([Dense(64, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
                        Dropout(0.1),
                        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
                        Dropout(0.1),
                        Dense(4)])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    return model

def build_stage3_model(input_dim):

    model = Sequential([Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                        BatchNormalization(),
                        Dropout(0.1),
                        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                        BatchNormalization(),
                        Dropout(0.1),
                        Dense(4)])

    optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])

    return model



#==========Baseline comparison================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Validation metrics
r2_scores, mae_scores, mse_scores = [], [], []
r2_W_scores, r2_ER_scores, r2_RD_scores, r2_TS_scores = [], [], [], []

# Training metrics
r2_scores_tr, mae_scores_tr, mse_scores_tr = [], [], []
r2_W_scores_tr, r2_ER_scores_tr, r2_RD_scores_tr, r2_TS_scores_tr = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled_s3)):

    print(f"Fold {fold+1}")

    X_train = X_scaled_s3[train_idx]
    X_val   = X_scaled_s3[val_idx]

    Y_train = Y_scaled_s3[train_idx]
    Y_val   = Y_scaled_s3[val_idx]

    # Same architecture as proposed model
    model = build_stage31_model(X_train.shape[1])

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        Y_train,
        validation_split=0.2,
        epochs=500,
        batch_size=8,
        callbacks=[early_stop],
        verbose=0,
        shuffle=True
    )

    # Predictions
    pred = model.predict(X_val, verbose=0)
    pred_train = model.predict(X_train, verbose=0)

    # Training metrics
    r2_scores_tr.append(r2_score(Y_train.flatten(), pred_train.flatten()))
    mae_scores_tr.append(mean_absolute_error(Y_train.flatten(), pred_train.flatten()))
    mse_scores_tr.append(mean_squared_error(Y_train.flatten(), pred_train.flatten()))

    r2_W_scores_tr.append(r2_score(Y_train[:,0], pred_train[:,0]))
    r2_ER_scores_tr.append(r2_score(Y_train[:,1], pred_train[:,1]))
    r2_RD_scores_tr.append(r2_score(Y_train[:,2], pred_train[:,2]))
    r2_TS_scores_tr.append(r2_score(Y_train[:, 3], pred_train[:, 3]))

    # Validation metrics
    r2_scores.append(r2_score(Y_val.flatten(), pred.flatten()))
    mae_scores.append(mean_absolute_error(Y_val.flatten(), pred.flatten()))
    mse_scores.append(mean_squared_error(Y_val.flatten(), pred.flatten()))

    r2_W_scores.append(r2_score(Y_val[:,0], pred[:,0]))
    r2_ER_scores.append(r2_score(Y_val[:,1], pred[:,1]))
    r2_RD_scores.append(r2_score(Y_val[:,2], pred[:,2]))
    r2_TS_scores.append(r2_score(Y_val[:, 3], pred[:, 3]))

print("------ Neural Network (Scratch) Results ------")

print("\nTraining Metrics")
print(f"Overall R² : {np.mean(r2_scores_tr):.4f}")
print(f"Overall MAE: {np.mean(mae_scores_tr):.4f}")
print(f"Overall MSE: {np.mean(mse_scores_tr):.4f}")

print(f"W R²: {np.mean(r2_W_scores_tr):.4f}")
print(f"ER R²: {np.mean(r2_ER_scores_tr):.4f}")
print(f"RD R²: {np.mean(r2_RD_scores_tr):.4f}")
print(f"TS R²: {np.mean(r2_TS_scores_tr):.4f}")

print("\nValidation Metrics")
print(f"Overall R² : {np.mean(r2_scores):.4f}")
print(f"Overall MAE: {np.mean(mae_scores):.4f}")
print(f"Overall MSE: {np.mean(mse_scores):.4f}")

print(f"W R²: {np.mean(r2_W_scores):.4f}")
print(f"ER R²: {np.mean(r2_ER_scores):.4f}")
print(f"RD R²: {np.mean(r2_RD_scores):.4f}")
print(f"TS R²: {np.mean(r2_TS_scores):.4f}")


# ENSEMBLE SETTINGS
N_ENSEMBLE = 10
ensemble_models = []

train_losses_all, val_losses_all = [], []
# ENSEMBLE PRETRAINING ON SYNTHETIC DATA
print("ENSEMBLE PRETRAINING ON SYNTHETIC DATA")

syn_r2, syn_mae, syn_mse = [], [], []
syn_r2_tr, syn_mae_tr, syn_mse_tr = [], [], []
r2_W_syn_tr, r2_ER_syn_tr, r2_RD_syn_tr, r2_TS_syn_tr = [], [], [], []
r2_W_syn_val, r2_ER_syn_val, r2_RD_syn_val, r2_TS_syn_val = [], [], [], []

for j in range(N_ENSEMBLE):

    print(f"Training ensemble member {j+1}/{N_ENSEMBLE}")
    tf.keras.backend.clear_session()

    #np.random.seed(i)
    #tf.random.set_seed(i)

    model_j = build_stage3_model(X_syn.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model_j.fit(
        X_syn_train,
        Y_syn_train,
        validation_data=(X_syn_val, Y_syn_val),
        epochs=500,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0,
        shuffle=True
    )

    # Store training and validation loss
    train_losses_all.append(history.history['loss'])
    val_losses_all.append(history.history['val_loss'])

    # VALIDATION ON SYNTHETIC DATA
    preds_syn = model_j.predict(X_syn_val)
    preds_syn_train = model_j.predict(X_syn_train)

    # train
    r2_syn_tr = r2_score(Y_syn_train, preds_syn_train)
    mae_syn_tr = mean_absolute_error(Y_syn_train, preds_syn_train)
    mse_syn_tr = mean_squared_error(Y_syn_train, preds_syn_train)
    syn_r2_tr.append(r2_syn_tr)
    syn_mae_tr.append(mae_syn_tr)
    syn_mse_tr.append(mse_syn_tr)
    r2_W_syn_tr.append(r2_score(Y_syn_train[:, 0], preds_syn_train[:, 0]))
    r2_ER_syn_tr.append(r2_score(Y_syn_train[:, 1], preds_syn_train[:, 1]))
    r2_RD_syn_tr.append(r2_score(Y_syn_train[:, 2], preds_syn_train[:, 2]))
    r2_TS_syn_tr.append(r2_score(Y_syn_train[:, 3], preds_syn_train[:, 3]))

    # validation
    r2_syn = r2_score(Y_syn_val, preds_syn)
    mae_syn = mean_absolute_error(Y_syn_val, preds_syn)
    mse_syn = mean_squared_error(Y_syn_val, preds_syn)
    syn_r2.append(r2_syn)
    syn_mae.append(mae_syn)
    syn_mse.append(mse_syn)
    r2_W_syn_val.append(r2_score(Y_syn_val[:, 0], preds_syn[:, 0]))
    r2_ER_syn_val.append(r2_score(Y_syn_val[:, 1], preds_syn[:, 1]))
    r2_RD_syn_val.append(r2_score(Y_syn_val[:, 2], preds_syn[:, 2]))
    r2_TS_syn_val.append(r2_score(Y_syn_val[:, 3], preds_syn[:, 3]))


    print(f"NN Validation R²: {r2_syn:.4f}")
    print(f"NN Validation MAE: {mae_syn:.4f}")
    print(f"NN Validation MSE: {mse_syn:.4f}")

    ensemble_models.append(model_j)

print("\n================================================")
print("NN PRETRAINING RESULTS")
print("training")
print(f"Mean R²: {np.mean(syn_r2_tr):.4f}")
print(f"Mean MAE: {np.mean(syn_mae_tr):.4f}")
print(f"Mean MSE: {np.mean(syn_mse_tr):.4f}")
print(f"W R²: {np.mean(r2_W_syn_tr):.4f}")
print(f"ER R²: {np.mean(r2_ER_syn_tr):.4f}")
print(f"RD R²: {np.mean(r2_RD_syn_tr):.4f}")
print(f"TS R²: {np.mean(r2_TS_syn_tr):.4f}")
print("validation")
print(f"Mean R²: {np.mean(syn_r2):.4f}")
print(f"Mean MAE: {np.mean(syn_mae):.4f}")
print(f"Mean MSE: {np.mean(syn_mse):.4f}")
print(f"W R²: {np.mean(r2_W_syn_val):.4f}")
print(f"ER R²: {np.mean(r2_ER_syn_val):.4f}")
print(f"RD R²: {np.mean(r2_RD_syn_val):.4f}")
print(f"TS R²: {np.mean(r2_TS_syn_val):.4f}")


# train_losses_all and val_losses_all are lists of lists
# Each element corresponds to one ensemble member
# Maximum number of epochs among all ensemble members
max_epochs = max(max(len(x) for x in train_losses_all),
                 max(len(x) for x in val_losses_all))

# Create arrays filled with NaN
train_array = np.full((len(train_losses_all), max_epochs), np.nan)
val_array   = np.full((len(val_losses_all), max_epochs), np.nan)

# Copy each loss curve
for i, losses in enumerate(train_losses_all):
    train_array[i, :len(losses)] = losses

for i, losses in enumerate(val_losses_all):
    val_array[i, :len(losses)] = losses

# Ensemble statistics
train_mean = np.nanmean(train_array, axis=0)
train_std  = np.nanstd(train_array, axis=0)

val_mean = np.nanmean(val_array, axis=0)
val_std  = np.nanstd(val_array, axis=0)

epochs = np.arange(1, max_epochs + 1)

# Plot
sns.set_style("whitegrid")
sns.set_context("talk")

plt.figure(figsize=(8,6))

# Mean training loss
plt.plot(epochs, train_mean, color="tab:blue", linewidth=2.5, label="Training")
plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, color="tab:blue", alpha=0.25)

# Mean validation loss
plt.plot(epochs, val_mean, color="tab:orange", linewidth=2.5, label="Validation")
plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, color="tab:orange", alpha=0.25)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Ensemble Pretraining Learning Curves")
plt.legend(frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig("ensemble_pretraining_learning_curve_tablets.pdf", dpi=600, bbox_inches="tight")
plt.show()


# TRANSFER LEARNING USING REAL DATA
print("\n================================================")
print("TRANSFER LEARNING ON REAL DATA")


# TRANSFER LEARNING FUNCTION
def tl_model(pretrained_model):
    # freeze early latent representation layers
    for layer in pretrained_model.layers[:-2]:
        layer.trainable = False

    pretrained_model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

    return pretrained_model


#loo = LeaveOneOut()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# METRIC STORAGE
r2_scores, mae_scores, mse_scores = [], [], []
r2_W_scores, r2_ER_scores, r2_RD_scores, r2_TS_scores = [], [], [], []
all_actuals, all_predictions, all_residuals = [], [], []
actuals_W, actuals_ER, actuals_RD, actuals_TS = [], [], [], []
predictuals_W, predictuals_ER, predictuals_RD, predictuals_TS = [], [], [], []
residuals_W, residuals_ER, residuals_RD, residuals_TS = [], [], [], []

r2_scores_tr, mae_scores_tr, mse_scores_tr = [], [], []
r2_W_tr, r2_ER_tr, r2_RD_tr, r2_TS_tr = [], [], [], []

# KFOLD TRANSFER LEARNING
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled_s3)):

    print(f"Fold {fold+1}")

    X_train = X_scaled_s3[train_idx]
    X_val = X_scaled_s3[val_idx]

    Y_train = Y_scaled_s3[train_idx]
    Y_val = Y_scaled_s3[val_idx]

    fold_preds = []
    fold_preds_train = []

    # FINE-TUNE EACH ENSEMBLE MEMBER
    for ens_idx, pretrained_model in enumerate(ensemble_models):

        tf.keras.backend.clear_session()

        model_tl = clone_model(pretrained_model)
        model_tl.set_weights(pretrained_model.get_weights())
        model_tl = tl_model(model_tl)
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model_tl.fit(X_train, Y_train, epochs=500, batch_size=8, callbacks=[early_stop], verbose=0, shuffle=True)
        pred = model_tl.predict(X_val)
        train_pred = model_tl.predict(X_train)

        fold_preds.append(pred)
        fold_preds_train.append(train_pred)

    # ENSEMBLE AVERAGING
    fold_preds = np.array(fold_preds)
    fold_preds_train = np.array(fold_preds_train)
    pred_mean = np.mean(fold_preds, axis=0)
    pred_mean_train = np.mean(fold_preds_train, axis=0)
    pred_std = np.std(fold_preds, axis=0)

    # STORE RESULTS
    #all_actuals.append(Y_val.flatten())
    #all_predictions.append(pred_mean.flatten())
    #all_residuals.append(Y_val.flatten() - pred_mean.flatten())
    actuals_W.extend(Y_val[:, 0])
    actuals_ER.extend(Y_val[:, 1])
    actuals_RD.extend(Y_val[:, 2])
    actuals_TS.extend(Y_val[:, 3])
    predictuals_W.extend(pred_mean[:, 0])
    predictuals_ER.extend(pred_mean[:, 1])
    predictuals_RD.extend(pred_mean[:, 2])
    predictuals_TS.extend(pred_mean[:, 3])
    residuals_W.extend(Y_val[:, 0] - pred_mean[:, 0])
    residuals_ER.extend(Y_val[:, 1] - pred_mean[:, 1])
    residuals_RD.extend(Y_val[:, 2] - pred_mean[:, 2])
    residuals_TS.extend(Y_val[:, 3] - pred_mean[:, 3])

    # train metrics
    r2_scores_tr.append(r2_score(Y_train.flatten(), pred_mean_train.flatten()))
    mae_scores_tr.append(mean_absolute_error(Y_train.flatten(), pred_mean_train.flatten()))
    mse_scores_tr.append(mean_squared_error(Y_train.flatten(), pred_mean_train.flatten()))

    r2_W_tr.append(r2_score(Y_train[:, 0], pred_mean_train[:, 0]))
    r2_ER_tr.append(r2_score(Y_train[:, 1], pred_mean_train[:, 1]))
    r2_RD_tr.append(r2_score(Y_train[:, 2], pred_mean_train[:, 2]))
    r2_TS_tr.append(r2_score(Y_train[:, 3], pred_mean_train[:, 3]))

    # Val METRICS
    r2_scores.append(r2_score(Y_val.flatten(), pred_mean.flatten()))
    mae_scores.append(mean_absolute_error(Y_val.flatten(), pred_mean.flatten()))
    mse_scores.append(mean_squared_error(Y_val.flatten(), pred_mean.flatten()))

    r2_W_scores.append(r2_score(Y_val[:,0],pred_mean[:,0]))
    r2_ER_scores.append(r2_score(Y_val[:,1],pred_mean[:,1]))
    r2_RD_scores.append(r2_score(Y_val[:,2],pred_mean[:,2]))
    r2_TS_scores.append(r2_score(Y_val[:,3],pred_mean[:,3]))

print("FINAL TRANSFER LEARNING RESULTS")
print("training")
print(f"Ensemble TL R²: {np.mean(r2_scores_tr):.4f}")
print(f"Ensemble TL MAE: {np.mean(mae_scores_tr):.4f}")
print(f"Ensemble TL MSE: {np.mean(mse_scores_tr):.4f}")

print("\n--------------- INDIVIDUAL OUTPUTS ---------------\n")
print(f"W R²: {np.mean(r2_W_tr):.4f}")
print(f"ER R²: {np.mean(r2_ER_tr):.4f}")
print(f"RD R²: {np.mean(r2_RD_tr):.4f}")
print(f"TS R²: {np.mean(r2_TS_tr):.4f}")

print("validation")
print(f"Ensemble TL R²: {np.mean(r2_scores):.4f}")
print(f"Ensemble TL MAE: {np.mean(mae_scores):.4f}")
print(f"Ensemble TL MSE: {np.mean(mse_scores):.4f}")

print("\n--------------- INDIVIDUAL OUTPUTS ---------------\n")
print(f"W R²: {np.mean(r2_W_scores):.4f}")
print(f"ER R²: {np.mean(r2_ER_scores):.4f}")
print(f"RD R²: {np.mean(r2_RD_scores):.4f}")
print(f"TS R²: {np.mean(r2_TS_scores):.4f}")


#---------------------Scatter plot-----------------------
# Use seaborn's whitegrid style
sns.set(style='whitegrid', context='talk', palette='colorblind')

# Plot settings
fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
axs = axs.flatten()
#titles = ['Weight Prediction', 'Elastic Recovery Prediction', 'Relative Density Prediction', 'Tensile Strength Prediction']
y_labels = ['Predicted Weight', 'Predicted ER', 'Predicted RD', 'Predicted TS']
x_labels = ['True Weight', 'True ER', 'True RD', 'True TS']
r2_scores_j = [np.mean(r2_W_scores), np.mean(r2_ER_scores),
               np.mean(r2_RD_scores), np.mean(r2_TS_scores)]
actuals = np.column_stack((actuals_W, actuals_ER, actuals_RD, actuals_TS))
predictuals = np.column_stack((predictuals_W, predictuals_ER,
                               predictuals_RD, predictuals_TS))
colors = sns.color_palette("colorblind")

for i in range(4):
    #Scatter plot
    axs[i].scatter(actuals[:,i], predictuals[:,i], alpha=0.7, edgecolor='k', s=60, color=colors[i])

    # Ideal fit line
    min_val = min(actuals[:,i].min(), predictuals[:,i].min())
    max_val = max(actuals[:,i].max(), predictuals[:,i].max())
    axs[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')

    # Labels, titles, and R² annotation
    #axs[i].set_title(titles[i], fontsize=16)
    axs[i].set_xlabel(x_labels[i], fontsize=14)
    axs[i].set_ylabel(y_labels[i], fontsize=14)
    axs[i].legend(fontsize=10)
    axs[i].grid(True, linestyle='--', alpha=0.6)

    # Annotate R² score
    axs[i].text(0.05, 0.92, f"$R^2 = {r2_scores_j[i]:.2f}$", transform=axs[i].transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8))

plt.suptitle('Tablet Property Predictions')
plt.tight_layout()
plt.savefig("GPR_NN_TL_tabletscatterR.pdf", format='pdf', bbox_inches='tight')
plt.show()


#-------------------------QQ plots--------------------------
# Create QQ plots in 2x2 layout
fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
axs = axs.flatten()
# QQ plot for output 1
stats.probplot(residuals_W, dist="norm", plot=axs[0])
axs[0].set_title("Weight")

# QQ plot for output 2
stats.probplot(residuals_ER, dist="norm", plot=axs[1])
axs[1].set_title("Elastic Recovery")

# QQ plot for output 3
stats.probplot(residuals_RD, dist="norm", plot=axs[2])
axs[2].set_title("Relative Density")

# QQ plot for output 4
stats.probplot(residuals_TS, dist="norm", plot=axs[3])
axs[3].set_title("Tensile Strength")

plt.suptitle('QQ Plot of Residuals - Tablets')
plt.tight_layout()
plt.savefig("GPR_NN_TL_tabletQQplotR.pdf", format='pdf', bbox_inches='tight')
plt.show()


# FINAL STAGE-3 ENSEMBLE TRAINING
print("FINAL FULL-DATA STAGE-3 TRAINING")

final_stage3_ensemble = []

for ens_idx, pretrained_model in enumerate(ensemble_models):

    print(f"Final ensemble member {ens_idx+1}")
    tf.keras.backend.clear_session()
    model_final = clone_model(pretrained_model)
    model_final.set_weights(pretrained_model.get_weights())
    model_final = tl_model(model_final)
    early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

    model_final.fit(X_scaled_s3, Y_scaled_s3, epochs=500, batch_size=8, callbacks=[early_stop], verbose=0, shuffle=True)

    final_stage3_ensemble.append(model_final)

print("\nFinal Stage-3 ensemble ready.\n")

# SAVE FINAL STAGE-3 ENSEMBLE
for i, model_j in enumerate(final_stage3_ensemble):
    model_j.save(f"final_stage3_model_{i}.keras")

joblib.dump(scaler_stage3,'stage3_scaler.pkl')
joblib.dump(scaler_Y_pre_s3,'stage3_output_scaler.pkl')

print("\nFinal Stage-3 models saved.\n")
