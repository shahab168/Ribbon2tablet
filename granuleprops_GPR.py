import random, torch, joblib
import numpy as np
import pandas as pd
from keras.src.layers import LayerNormalization
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


# LOAD FINAL STAGE-1 MODELS
stage1_reg_model = tf.keras.models.load_model('final_stage1_regression_model.keras')
stage1_split_model = tf.keras.models.load_model('final_stage1_split_model.keras')


gpr = joblib.load('gprR.pkl')
scaler_X = joblib.load('gpr_scalerR.pkl')
gpc = joblib.load('gpcR.pkl')

# -------------------------
# Load models
model = tf.keras.models.load_model('nn_model_fullR.keras')
split_model = tf.keras.models.load_model('split_model_fullR.keras')

df = pd.read_csv('ribbontotablet_splittingdata.csv')
X = np.array(df[['Roll Gap', 'Roll Pressure', 'Roll Speed',	'Screw Feed Speed',	'API', 'First Run']])
Y = np.array(df[['d10', 'd50', 'd90']])

# Scale Stage-2 inputs
X_scaled = scaler_X.transform(X)

scaler_Y_pre = StandardScaler()
Y_scaled = scaler_Y_pre.fit_transform(Y)

# GPR ribbon predictions
ribbon_preds = stage1_reg_model.predict(X_scaled)
#ribbon_preds, ribbon_std = gpr.predict(X_scaled, return_std=True)

pred_density = ribbon_preds[:,0]
pred_thickness = ribbon_preds[:,1]

# GPC split probabilities
split_probs = stage1_split_model.predict(X_scaled)
#split_probs = gpc.predict_proba(X_scaled)[:,1]

# Stage-2 feature matrix
X_stage2 = np.column_stack([pred_density, pred_thickness, split_probs, X[:,4]])
#X_stage2 = np.column_stack([X, pred_density, pred_thickness, split_probs])

scaler_stage2 = StandardScaler()
X_stage2_scaled = scaler_stage2.fit_transform(X_stage2)

# Granule quality descriptors
#d10, d50, d90 = Y[:,0], Y[:,1], Y[:,2]
#span = (d90 - d10) / d50
#fines_index = d10 / d50

#Y_stage2 = Y # np.column_stack([d50, span, fines_index])

# Stage-2 GPR
kernel2 = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gpr_stage2 = GaussianProcessRegressor(kernel=kernel2, alpha=1e-4, normalize_y=True)

gpr_stage2.fit(X_stage2_scaled, Y_scaled)

# Predictions on the same input data
Y_pred, sigma = gpr_stage2.predict(X_stage2_scaled, return_std=True)

# Evaluate model performance
mae = mean_absolute_error(Y_scaled, Y_pred)
mse = mean_squared_error(Y_scaled, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_scaled, Y_pred)

# Print metrics
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

r2_d10 = r2_score(Y_scaled[:, 0], Y_pred[:, 0])
r2_d50 = r2_score(Y_scaled[:, 1], Y_pred[:, 1])
r2_d90 = r2_score(Y_scaled[:, 2], Y_pred[:, 2])

print(f"R² Score - D10: {r2_d10:.4f}")
print(f"R² Score - D50: {r2_d50:.4f}")
print(f"R² Score - D90: {r2_d90:.4f}")


# ============Data Augmentation==================
n_augments = 10

X_aug = []
Y_aug = []

print("Generating synthetic Stage-2 samples...")

for i in range(len(X_stage2_scaled)):
    x = X_stage2_scaled[i]
    y_mean, y_std = gpr_stage2.predict([x], return_std=True)
    for _ in range(n_augments):
        y_sample = np.random.normal(loc=y_mean.ravel(), scale=y_std)
        X_aug.append(x)
        Y_aug.append(y_sample)

X_aug = np.array(X_aug)
Y_aug = np.array(Y_aug).reshape(len(Y_aug), len(Y_aug[0][0]))

# Combine real + synthetic
X_total = np.vstack([X_stage2_scaled, X_aug])
Y_total = np.vstack([Y_scaled, Y_aug])
print(X_total.shape)
print(Y_total.shape)


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# STAGE-2 ENSEMBLE LEARNING + TRANSFER LEARNING

# SYNTHETIC DATASET ONLY

X_syn = X_aug
Y_syn = Y_aug

print("\nSynthetic dataset shape:")
print(X_syn.shape)
print(Y_syn.shape)

# Train / val split for pretraining
X_syn_train, X_syn_val, Y_syn_train, Y_syn_val = train_test_split(X_syn, Y_syn, test_size=0.2, random_state=SEED, shuffle=True)

# MODEL ARCHITECTURE
def build_stage21_model(input_dim):
    model = Sequential([Dense(32, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
                        Dropout(0.1),
                        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
                        Dropout(0.1),
                        Dense(3)])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    return model

def build_stage2_model(input_dim):
    model = Sequential([Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                        LayerNormalization(),
                        Dropout(0.3),
                        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                        LayerNormalization(),
                        Dropout(0.3),
                        Dense(3)])

    optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])
    return model


#==========Baseline comparison================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Validation metrics
r2_scores, mae_scores, mse_scores = [], [], []
r2_d10_scores, r2_d50_scores, r2_d90_scores = [], [], []

# Training metrics
r2_scores_tr, mae_scores_tr, mse_scores_tr = [], [], []
r2_d10_scores_tr, r2_d50_scores_tr, r2_d90_scores_tr = [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_stage2_scaled)):

    print(f"Fold {fold+1}")

    X_train = X_stage2_scaled[train_idx]
    X_val   = X_stage2_scaled[val_idx]

    Y_train = Y_scaled[train_idx]
    Y_val   = Y_scaled[val_idx]

    # Same architecture as proposed model
    model = build_stage21_model(X_train.shape[1])

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

    # ==========================
    # Training metrics
    # ==========================

    r2_scores_tr.append(r2_score(Y_train.flatten(), pred_train.flatten()))
    mae_scores_tr.append(mean_absolute_error(Y_train.flatten(), pred_train.flatten()))
    mse_scores_tr.append(mean_squared_error(Y_train.flatten(), pred_train.flatten()))

    r2_d10_scores_tr.append(r2_score(Y_train[:,0], pred_train[:,0]))
    r2_d50_scores_tr.append(r2_score(Y_train[:,1], pred_train[:,1]))
    r2_d90_scores_tr.append(r2_score(Y_train[:,2], pred_train[:,2]))

    # ==========================
    # Validation metrics
    # ==========================

    r2_scores.append(r2_score(Y_val.flatten(), pred.flatten()))
    mae_scores.append(mean_absolute_error(Y_val.flatten(), pred.flatten()))
    mse_scores.append(mean_squared_error(Y_val.flatten(), pred.flatten()))

    r2_d10_scores.append(r2_score(Y_val[:,0], pred[:,0]))
    r2_d50_scores.append(r2_score(Y_val[:,1], pred[:,1]))
    r2_d90_scores.append(r2_score(Y_val[:,2], pred[:,2]))

print("------ Neural Network (Scratch) Results ------")

print("\nTraining Metrics")
print(f"Overall R² : {np.mean(r2_scores_tr):.4f}")
print(f"Overall MAE: {np.mean(mae_scores_tr):.4f}")
print(f"Overall MSE: {np.mean(mse_scores_tr):.4f}")

print(f"D10 R²: {np.mean(r2_d10_scores_tr):.4f}")
print(f"D50 R²: {np.mean(r2_d50_scores_tr):.4f}")
print(f"D90 R²: {np.mean(r2_d90_scores_tr):.4f}")

print("\nValidation Metrics")
print(f"Overall R² : {np.mean(r2_scores):.4f}")
print(f"Overall MAE: {np.mean(mae_scores):.4f}")
print(f"Overall MSE: {np.mean(mse_scores):.4f}")

print(f"D10 R²: {np.mean(r2_d10_scores):.4f}")
print(f"D50 R²: {np.mean(r2_d50_scores):.4f}")
print(f"D90 R²: {np.mean(r2_d90_scores):.4f}")


# ENSEMBLE SETTINGS
N_ENSEMBLE = 10
ensemble_models = []

train_losses_all, val_losses_all = [], []
# ENSEMBLE PRETRAINING ON SYNTHETIC DATA
print("ENSEMBLE PRETRAINING ON SYNTHETIC DATA")

syn_r2, syn_mae, syn_mse = [], [], []
syn_r2_tr, syn_mae_tr, syn_mse_tr = [], [], []
r2_d10_syn_tr, r2_d50_syn_tr, r2_d90_syn_tr = [], [], []
r2_d10_syn_val, r2_d50_syn_val, r2_d90_syn_val = [], [], []

for i in range(N_ENSEMBLE):

    print(f"Training ensemble member {i+1}/{N_ENSEMBLE}")
    tf.keras.backend.clear_session()

    #np.random.seed(i)
    #tf.random.set_seed(i)

    model_i = build_stage2_model(X_syn.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model_i.fit(
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
    preds_syn = model_i.predict(X_syn_val)
    preds_syn_train = model_i.predict(X_syn_train)

    # train
    r2_syn_tr = r2_score(Y_syn_train, preds_syn_train)
    mae_syn_tr = mean_absolute_error(Y_syn_train, preds_syn_train)
    mse_syn_tr = mean_squared_error(Y_syn_train, preds_syn_train)
    syn_r2_tr.append(r2_syn_tr)
    syn_mae_tr.append(mae_syn_tr)
    syn_mse_tr.append(mse_syn_tr)
    r2_d10_syn_tr.append(r2_score(Y_syn_train[:, 0], preds_syn_train[:, 0]))
    r2_d50_syn_tr.append(r2_score(Y_syn_train[:, 1], preds_syn_train[:, 1]))
    r2_d90_syn_tr.append(r2_score(Y_syn_train[:, 2], preds_syn_train[:, 2]))

    # validation
    r2_syn = r2_score(Y_syn_val, preds_syn)
    mae_syn = mean_absolute_error(Y_syn_val, preds_syn)
    mse_syn = mean_squared_error(Y_syn_val, preds_syn)
    syn_r2.append(r2_syn)
    syn_mae.append(mae_syn)
    syn_mse.append(mse_syn)
    r2_d10_syn_val.append(r2_score(Y_syn_val[:, 0], preds_syn[:, 0]))
    r2_d50_syn_val.append(r2_score(Y_syn_val[:, 1], preds_syn[:, 1]))
    r2_d90_syn_val.append(r2_score(Y_syn_val[:, 2], preds_syn[:, 2]))

    #print(f"NN Validation R²: {r2_syn:.4f}")
    #print(f"NN Validation MAE: {mae_syn:.4f}")
    #print(f"NN Validation MSE: {mse_syn:.4f}")

    ensemble_models.append(model_i)

print("\n================================================")
print("NN PRETRAINING RESULTS")
print("training")
print(f"Mean R²: {np.mean(syn_r2_tr):.4f}")
print(f"Mean MAE: {np.mean(syn_mae_tr):.4f}")
print(f"Mean MSE: {np.mean(syn_mse_tr):.4f}")
print(f"D10 R²: {np.mean(r2_d10_syn_tr):.4f}")
print(f"D50 R²: {np.mean(r2_d50_syn_tr):.4f}")
print(f"D90 R²: {np.mean(r2_d90_syn_tr):.4f}")
print("validation")
print(f"Mean R²: {np.mean(syn_r2):.4f}")
print(f"Mean MAE: {np.mean(syn_mae):.4f}")
print(f"Mean MSE: {np.mean(syn_mse):.4f}")
print(f"D10 R²: {np.mean(r2_d10_syn_val):.4f}")
print(f"D50 R²: {np.mean(r2_d50_syn_val):.4f}")
print(f"D90 R²: {np.mean(r2_d90_syn_val):.4f}")


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
plt.savefig("ensemble_pretraining_learning_curve_granules.pdf", dpi=600, bbox_inches="tight")
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
r2_d10_scores, r2_d50_scores, r2_d90_scores = [], [], []
actuals_d10, actuals_d50, actuals_d90 = [], [], []
predictuals_d10, predictuals_d50, predictuals_d90 = [], [], []
residuals_d10, residuals_d50, residuals_d90 = [], [], []

r2_scores_tr, mae_scores_tr, mse_scores_tr = [], [], []
r2_d10_scores_tr, r2_d50_scores_tr, r2_d90_scores_tr = [], [], []

# KFOLD TRANSFER LEARNING
for fold, (train_idx, val_idx) in enumerate(kf.split(X_stage2_scaled)):

    print(f"Fold {fold+1}")

    X_train = X_stage2_scaled[train_idx]
    X_val = X_stage2_scaled[val_idx]

    Y_train = Y_scaled[train_idx]
    Y_val = Y_scaled[val_idx]

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
    #all_actuals.append(Y_val)
    #all_predictuals.append(pred_mean)
    #all_residuals.append(Y_val - pred_mean)
    actuals_d10.extend(Y_val[:, 0])
    actuals_d50.extend(Y_val[:, 1])
    actuals_d90.extend(Y_val[:, 2])
    predictuals_d10.extend(pred_mean[:, 0])
    predictuals_d50.extend(pred_mean[:, 1])
    predictuals_d90.extend(pred_mean[:, 2])
    residuals_d10.extend(Y_val[:, 0] - pred_mean[:, 0])
    residuals_d50.extend(Y_val[:, 1] - pred_mean[:, 1])
    residuals_d90.extend(Y_val[:, 2] - pred_mean[:, 2])

    # train metrics
    r2_scores_tr.append(r2_score(Y_train.flatten(), pred_mean_train.flatten()))
    mae_scores_tr.append(mean_absolute_error(Y_train.flatten(), pred_mean_train.flatten()))
    mse_scores_tr.append(mean_squared_error(Y_train.flatten(), pred_mean_train.flatten()))

    r2_d10_scores_tr.append(r2_score(Y_train[:, 0], pred_mean_train[:, 0]))
    r2_d50_scores_tr.append(r2_score(Y_train[:, 1], pred_mean_train[:, 1]))
    r2_d90_scores_tr.append(r2_score(Y_train[:, 2], pred_mean_train[:, 2]))

    # val METRICS
    r2_scores.append(r2_score(Y_val.flatten(), pred_mean.flatten()))
    mae_scores.append(mean_absolute_error(Y_val.flatten(), pred_mean.flatten()))
    mse_scores.append(mean_squared_error(Y_val.flatten(), pred_mean.flatten()))

    r2_d10_scores.append(r2_score(Y_val[:,0],pred_mean[:,0]))
    r2_d50_scores.append(r2_score(Y_val[:,1],pred_mean[:,1]))
    r2_d90_scores.append(r2_score(Y_val[:,2],pred_mean[:,2]))


# FINAL RESULTS
#all_actuals = np.array(all_actuals)
#all_predictions = np.array(all_predictions)
print("FINAL TRANSFER LEARNING RESULTS")
print("Training")
print(f"Ensemble TL R²: {np.mean(r2_scores_tr):.4f}")
print(f"Ensemble TL MAE: {np.mean(mae_scores_tr):.4f}")
print(f"Ensemble TL MSE: {np.mean(mse_scores_tr):.4f}")

print("\n--------------- INDIVIDUAL OUTPUTS ---------------\n")

print(f"D10 R²: {np.mean(r2_d10_scores_tr):.4f}")
print(f"D50 R²: {np.mean(r2_d50_scores_tr):.4f}")
print(f"D90 R²: {np.mean(r2_d90_scores_tr):.4f}")

print("Validation")
print(f"Ensemble TL R²: {np.mean(r2_scores):.4f}")
print(f"Ensemble TL MAE: {np.mean(mae_scores):.4f}")
print(f"Ensemble TL MSE: {np.mean(mse_scores):.4f}")

print("\n--------------- INDIVIDUAL OUTPUTS ---------------\n")

print(f"D10 R²: {np.mean(r2_d10_scores):.4f}")
print(f"D50 R²: {np.mean(r2_d50_scores):.4f}")
print(f"D90 R²: {np.mean(r2_d90_scores):.4f}")


#---------------------Scatter plot-----------------------
# Use seaborn's whitegrid style
sns.set(style='whitegrid', context='talk', palette='colorblind')

# Plot settings
fig, axs = plt.subplots(1, 3, figsize=(12, 5), dpi=300)
#titles = ['D10 Prediction', 'D50 Prediction', 'D90 Prediction']
y_labels = ['Predicted D10', 'Predicted D50', 'Predicted D90']
x_labels = ['True D10', 'True D50', 'True D90']
r2_scores_j = [np.mean(r2_d10_scores), np.mean(r2_d50_scores), np.mean(r2_d90_scores)]
actuals = np.column_stack((actuals_d10, actuals_d50, actuals_d90))
predictuals = np.column_stack((predictuals_d10, predictuals_d50, predictuals_d90))
colors = sns.color_palette("colorblind")

for i in range(3):
    #Scatter plot
    axs[i].scatter(actuals[:,i], predictuals[:,i],
                   alpha=0.7, edgecolor='k', s=60, color=colors[i])

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
    axs[i].text(0.05, 0.92,
                f"$R^2 = {r2_scores_j[i]:.2f}$",
                transform=axs[i].transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.8))

plt.suptitle('Granule Property Predictions')
plt.tight_layout()
plt.savefig("GPR_NN_TL_granulescatterR.pdf", format='pdf', bbox_inches='tight')
plt.show()

#-------------------------QQ plots--------------------------
# Create QQ plots in 1x2 layout
fig, axs = plt.subplots(1, 3, figsize=(12, 5))

# QQ plot for output 1
stats.probplot(residuals_d10, dist="norm", plot=axs[0])
axs[0].set_title("D10")

# QQ plot for output 2
stats.probplot(residuals_d50, dist="norm", plot=axs[1])
axs[1].set_title("D50")

# QQ plot for output 3
stats.probplot(residuals_d90, dist="norm", plot=axs[2])
axs[2].set_title("D90")

plt.suptitle('QQ Plot of Residuals - Granules')
plt.tight_layout()
plt.savefig("GPR_NN_TL_granuleQQplotR.pdf", format='pdf', bbox_inches='tight')
plt.show()


# FINAL STAGE-2 ENSEMBLE TRAINING
print("FINAL FULL-DATA STAGE-2 TRAINING")

final_stage2_ensemble = []

for ens_idx, pretrained_model in enumerate(ensemble_models):

    print(f"Final ensemble member {ens_idx+1}")
    tf.keras.backend.clear_session()
    model_final = clone_model(pretrained_model)
    model_final.set_weights(pretrained_model.get_weights())
    model_final = tl_model(model_final)
    early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

    model_final.fit(X_stage2_scaled, Y_scaled, epochs=500, batch_size=8, callbacks=[early_stop], verbose=0, shuffle=True)

    final_stage2_ensemble.append(model_final)

print("\nFinal Stage-2 ensemble ready.\n")

# SAVE FINAL STAGE-2 ENSEMBLE
for i, model_i in enumerate(final_stage2_ensemble):
    model_i.save(f"final_stage2_model_{i}.keras")

joblib.dump(scaler_stage2,'stage2_scaler.pkl')
joblib.dump(scaler_Y_pre,'stage2_output_scaler.pkl')

print("\nFinal Stage-2 models saved.\n")
