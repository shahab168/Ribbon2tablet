import random, torch, joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
                             roc_curve, auc, f1_score, accuracy_score, confusion_matrix)
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tabpfn_client import TabPFNClassifier, TabPFNRegressor, set_access_token
set_access_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMmM1NGYxMDctN2RkZi00M2JiLWFhMzMtMzNlYmU1ZWQ5YWNmIiwiZXhwIjoxODExMzE0OTc5fQ.0_0qZaDmAPRMUltitH_lTdHgni5VE2cKgFQ4iXYEIFs")
import os

SEED = 9
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

# Load the dataset
df = pd.read_csv('splitting_data.csv')
X = np.array(df[['Roll Gap', 'Hydraulic Pressure', 'Roll Speed', 'Screw feed speed', 'API', 'FR']])
Y_all = np.array(df[['density', 'thickness', 'Split']])
Y = np.array(df[['density', 'thickness']])
Y_cls = np.array(df['Split'])

# Scale the input features
scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)

# Split only the original experimental data
X_train_real, X_test_real, Y_train_real, Y_test_real = train_test_split(
    X_scaled, Y, test_size=0.1, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

#------------------------GPR MODEL---------------------
# Kernel for GPR
# Use a product of a constant kernel and an RBF kernel (squared exponential) for flexibility
kernel_reg = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
# change matern's nu to 1.5 to achive previously incorporated results in the manuscript
kernel_cls = (C(1.0, (1e-4, 1e1)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5))

print("Training GPC...")
gpc = GaussianProcessClassifier(kernel=kernel_cls, n_restarts_optimizer=20) # remove n_restarts to achive previous included results in paper
gpc.fit(X_scaled, Y_cls)
split_probs = gpc.predict_proba(X_scaled)[:,1]

split_pred = gpc.predict(X_scaled)

roc_auc = roc_auc_score(Y_cls, split_probs)
f1 = f1_score(Y_cls, split_pred)

print(f"ROC-AUC Split: {roc_auc:.4f}")
print(f"F1 Split: {f1:.4f}")


# Initialize the Multi-output Gaussian Process Regressor
gpr = GaussianProcessRegressor(kernel=kernel_reg, alpha=1e-4)

# Fit the GPR model
print("Training the GPR model...")
gpr.fit(X_scaled, Y)

# Predictions on the same input data
Y_pred, sigma = gpr.predict(X_scaled, return_std=True)

# Evaluate model performance
mae = mean_absolute_error(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y, Y_pred)

# Print metrics
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

r2_density = r2_score(Y[:, 0], Y_pred[:, 0])
r2_thickness = r2_score(Y[:, 1], Y_pred[:, 1])
print(f"R² Score - Density: {r2_density:.4f}")
print(f"R² Score - Thickness: {r2_thickness:.4f}")


#-----------------------DATA AUGMENTATION------------------------
n_augments = 5 # How many synthetic samples per real point?

X_aug = []
Y_aug = []

print("Data augmentation in progress...")
for i in range(len(X_scaled)):
    x = X_scaled[i]
    y_mean, y_std = gpr.predict([x], return_std=True)
    # -------- Classification augmentation --------
    p_split = gpc.predict_proba([x])[0, 1]
    for _ in range(n_augments):
        #random.seed(52)
        y_reg_sample = np.random.normal(loc=y_mean.ravel(), scale=y_std)
        y_split_sample = np.random.binomial(1, p_split)
        final_sample = np.concatenate([y_reg_sample[0], [y_split_sample]])
        X_aug.append(x)
        Y_aug.append(final_sample)

X_aug = np.array(X_aug)
Y_aug = np.array(Y_aug).reshape(len(Y_aug), len(Y_aug[0]))

# Combine with real data
X_total = np.vstack([X_scaled, X_aug])
Y_total = np.vstack([Y_all, Y_aug])
#print(X_total.shape)

# ---------------- REGRESSION DATA ----------------
Y_total_reg = Y_total[:, :2]
# ---------------- CLASSIFICATION DATA ----------------
Y_total_cls = Y_total[:, 2]

# Train / val split for pretraining
X_syn = X_aug
Y_syn_reg = Y_aug[:,:2]
Y_syn_cls = Y_aug[:,2]


#-------------------------NEURAL NETWORK TRAINING AND EVALUATION-----------------
def build_model(input_dim):
    model = Sequential([Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
                        Dropout(0.1),
                        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                        Dropout(0.1),
                        Dense(2)])  # density and thickness

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_split_model(input_dim):
    model = Sequential([Dense(64, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
                        Dropout(0.2),
                        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
                        Dropout(0.2),
                        Dense(1, activation='sigmoid')])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


mseNN_tr, maeNN_tr, r2NN_tr, r2d_tr, r2t_tr = [], [], [], [], []
mse_scores, mae_scores, r2_scores = [], [], []
r2s_densityscores, r2s_thicknessscores = [], []

train_losses_all = []
val_losses_all = []
print("Neural network training on synthetic data...")

for train_idx, val_idx in kf.split(X_syn):
    X_train, X_val = X_syn[train_idx], X_syn[val_idx]
    Y_train, Y_val = Y_syn_reg[train_idx], Y_syn_reg[val_idx]

    model = build_model(X_total.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    hist = model.fit(X_train, Y_train,
              validation_data=(X_val, Y_val),
              epochs=200,
              batch_size=16,
              callbacks=[early_stop],
              verbose=0)

    # Store training and validation loss
    train_losses_all.append(hist.history['loss'])
    val_losses_all.append(hist.history['val_loss'])

    preds = model.predict(X_val)
    tr_preds = model.predict(X_train)

    # train set
    mseNN_tr.append(mean_squared_error(Y_train, tr_preds))
    maeNN_tr.append(mean_absolute_error(Y_train, tr_preds))
    r2NN_tr.append(r2_score(Y_train, tr_preds))
    r2d_tr.append(r2_score(Y_train[:, 0], tr_preds[:, 0]))
    r2t_tr.append(r2_score(Y_train[:, 1], tr_preds[:, 1]))

    # test/validation set
    mse_scores.append(mean_squared_error(Y_val, preds))
    mae_scores.append(mean_absolute_error(Y_val, preds))
    r2_scores.append(r2_score(Y_val, preds))
    r2s_densityscores.append(r2_score(Y_val[:, 0], preds[:, 0]))
    r2s_thicknessscores.append(r2_score(Y_val[:, 1], preds[:, 1]))

print("Avg. Metrics on augmented data (train)")
print(f"Avg. MSE on augmented data: {np.mean(mseNN_tr):.4f}")
print(f"Avg. MAE on augmented data: {np.mean(maeNN_tr):.4f}")
print(f"Avg. R2 score on augmented data: {np.mean(r2NN_tr):.4f}")

print(f"R² Score - Density: {np.mean(r2d_tr):.4f}")
print(f"R² Score - Thickness: {np.mean(r2t_tr):.4f}")


print("Avg. Metrics on augmented data (test/validation)")
print(f"Avg. MSE on augmented data: {np.mean(mse_scores):.4f}")
print(f"Avg. MAE on augmented data: {np.mean(mae_scores):.4f}")
print(f"Avg. R2 score on augmented data: {np.mean(r2_scores):.4f}")

print(f"R² Score - Density: {np.mean(r2s_densityscores):.4f}")
print(f"R² Score - Thickness: {np.mean(r2s_thicknessscores):.4f}")


# ---------------- CLASSIFICATION NN ----------------
print("Classification NN training on synthetic data...")

acc_scores = []
f1_scores_nn = []
auc_scores_nn = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#for train_idx, val_idx in kf.split(X_syn):
for train_idx, val_idx in skf.split(X_syn, (Y_syn_cls > 0.5).astype(int)):

    X_train, X_val = X_syn[train_idx], X_syn[val_idx]
    Y_train, Y_val = Y_syn_cls[train_idx], Y_syn_cls[val_idx]

    split_model = build_split_model(X_total.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    split_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=200, batch_size=16, callbacks=[early_stop], verbose=0)

    probs = split_model.predict(X_val)
    preds = (probs > 0.5).astype(int)
    acc_scores.append(np.mean(preds.flatten() == Y_val))
    f1_scores_nn.append(f1_score(Y_val, preds))
    auc_scores_nn.append(roc_auc_score(Y_val, preds))

print(f"NN Split Accuracy: {np.mean(acc_scores):.4f}")
print(f"NN Split F1: {np.mean(f1_scores_nn):.4f}")
print(f"NN Split ROC-AUC: {np.mean(auc_scores_nn):.4f}")


# Apply seaborn style for better aesthetics
sns.set(style='whitegrid', context='talk', palette='colorblind')

# Create a figure
plt.figure(figsize=(10, 6))

# Define a color palette
palette = sns.color_palette("colorblind", n_colors=len(train_losses_all))

# Plot training losses (dashed lines)
for i, loss in enumerate(train_losses_all):
    plt.plot(loss, label=f'Train Fold {i+1}', linestyle='--', color=palette[i])

# Plot validation losses (solid lines)
for i, val_loss in enumerate(val_losses_all):
    plt.plot(val_loss, label=f'Val Fold {i+1}', linestyle='-', color=palette[i])

# Add labels and title
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss (MSE)', fontsize=14)
plt.title('Training & Validation Loss Across Folds', fontsize=16)

# Add legend
plt.legend(title='Folds', fontsize=10, title_fontsize=12)

# Final layout adjustments
plt.tight_layout()
plt.savefig('loss_curves_across_foldsR.pdf', format='pdf', bbox_inches='tight')
plt.show()


#-----------------------TRANSFER LEARNING-----------------------------
print("Transfer learning in progress...")

# freeze early layers of the pre-trained model
def tl_model(pretrained_model):
    # Freeze first layers if you want to keep base features
    for layer in pretrained_model.layers[:-2]:  # Optional: freeze some base layers
        # for layer in model.layers:  # All layers are trainable (default fine-tuning)
        layer.trainable = False  # Fine-tune all layers # Or False to freeze
    # model.layers[-1].trainable = True

    # Recompile with a lower learning rate
    pretrained_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
    return pretrained_model


def tl_split_model(pretrained_model):
    for layer in pretrained_model.layers[:-2]:
        layer.trainable = False
    pretrained_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return pretrained_model


mse_tr, mae_tr, r2_tr, r2s_dscores_tr, r2s_tscores_tr = [], [], [], [], []
mse_val, mae_val, r2_val, r2s_dscores_val, r2s_tscores_val = [], [], [], [], []
actuals_d, actuals_t, predictuals_d, predictuals_t, residuals_d, residuals_t = [], [], [], [], [], []

for train_i, val_i in kf.split(X_scaled):
    X_tr, X_val = X_scaled[train_i], X_scaled[val_i]
    Y_tr, Y_val = Y[train_i], Y[val_i]

    # use the original pre-trained model everytime
    #final_model = tl_model(model)

    base_model = build_model(X_scaled.shape[1])
    base_model.set_weights(model.get_weights())
    final_model = tl_model(base_model)

    # Fine-tune using only real data
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = final_model.fit(X_tr, Y_tr,  # X_scaled, Y,
                        validation_split=0.2, # validation_data=(X_val, Y_val)
                        epochs=200,
                        batch_size=16,
                        callbacks=[early_stop],
                        verbose=0)

    preds = final_model.predict(X_val)
    train_preds = final_model.predict(X_tr)

    # saving values for scatter and QQ plots
    actuals_d.extend(Y_val[:,0])
    actuals_t.extend(Y_val[:,1])
    predictuals_d.extend(preds[:,0])
    predictuals_t.extend(preds[:,1])
    residuals_d.extend(Y_val[:,0] - preds[:,0])
    residuals_t.extend(Y_val[:,1] - preds[:,1])

    # train set
    mse_tr.append(mean_squared_error(Y_tr, train_preds))
    mae_tr.append(mean_absolute_error(Y_tr, train_preds))
    r2_tr.append(r2_score(Y_tr, train_preds))
    r2s_dscores_tr.append(r2_score(Y_tr[:, 0], train_preds[:, 0]))
    r2s_tscores_tr.append(r2_score(Y_tr[:, 1], train_preds[:, 1]))

    # test/validation set
    mse_val.append(mean_squared_error(Y_val, preds))
    mae_val.append(mean_absolute_error(Y_val, preds))
    r2_val.append(r2_score(Y_val, preds))
    r2s_dscores_val.append(r2_score(Y_val[:, 0], preds[:, 0]))
    r2s_tscores_val.append(r2_score(Y_val[:, 1], preds[:, 1]))


# ---------------- SPLIT CLASSIFIER TRANSFER LEARNING ----------------
print("Split-classifier transfer learning in progress...")

all_probs = []
all_preds = []
all_true = []
acc_tl = []
f1_tl = []
roc_tl = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#for train_i, val_i in kf.split(X_scaled):
for train_idx, val_idx in skf.split(X_scaled, (Y_cls > 0.5).astype(int)):

    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    Y_tr, Y_val = Y_cls[train_idx], Y_cls[val_idx]

    # build fresh pretrained classifier
    base_split_model = build_split_model(X_scaled.shape[1])
    # initialize with pretrained weights
    base_split_model.set_weights(split_model.get_weights())
    # apply TL freezing
    final_split_model = tl_split_model(base_split_model)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    final_split_model.fit(X_tr, Y_tr, validation_data=(X_val, Y_val), epochs=200, batch_size=16, callbacks=[early_stop], verbose=0)
    probs = final_split_model.predict(X_val)
    preds = (probs > 0.5).astype(int)

    # Store fold results
    all_true.extend(Y_val)
    all_probs.extend(probs)
    all_preds.extend(preds)

    acc_tl.append(np.mean(preds.flatten() == Y_val.flatten()))
    f1_tl.append(f1_score(Y_val, preds))
    roc_tl.append(roc_auc_score(Y_val, probs))


#------------------------FINAL MODEL EVALUATION & PLOTTING--------------------
# Evaluate the final model on test data (already fine-tuned)
print(f"Final Evaluation on training Set:")
print(f"Avg. MSE: {np.mean(mse_tr):.4f}")
print(f"Avg. MAE: {np.mean(mae_tr):.4f}")
print(f"Avg. R2: {np.mean(r2_tr):.4f}")

print(f"R² Score - Density: {np.mean(r2s_dscores_tr):.4f}")
print(f"R² Score - Thickness: {np.mean(r2s_tscores_tr):.4f}")


# Evaluate the final model on validation data
print(f"Final Evaluation on test/validation Set:")
print(f"Avg. MSE: {np.mean(mse_val):.4f} ({np.std(mse_val):.4f})")
print(f"Avg. MAE: {np.mean(mae_val):.4f} ({np.std(mae_val):.4f})")
print(f"Avg. R2: {np.mean(r2_val):.4f} ({np.std(r2_val):.4f})")

print(f"R² Score - Density: {np.mean(r2s_dscores_val):.4f} ({np.std(r2s_dscores_val):.4f})")
print(f"R² Score - Thickness: {np.mean(r2s_tscores_val):.4f} ({np.std(r2s_tscores_val):.4f})")

#--------Classification-------------
print("------Classification results----------")
print(f"TL Split Accuracy: {np.mean(acc_tl):.4f}")
print(f"TL Split F1: {np.mean(f1_tl):.4f}")
print(f"TL Split ROC-AUC: {np.mean(roc_tl):.4f}")


# ROC CURVE
fpr, tpr, _ = roc_curve(all_true, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6.5,6))
plt.plot(fpr, tpr, color='royalblue', lw=3, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1], [0,1], '--', color='gray', lw=2)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve - Ribbon Splitting Classifier', fontsize=15, weight='bold')
plt.legend(fontsize=13, frameon=False, loc='lower right')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("GPR_NN_TL_ROCcurve.pdf", format='pdf', bbox_inches='tight')
plt.show()

# CONFUSION MATRIX
cm = confusion_matrix(all_true, all_preds)

plt.figure(figsize=(6,5.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1.5, square=True, cbar=False, annot_kws={'fontsize':18, 'weight':'bold'})
plt.xticks([0.5,1.5], ['No Split','Split'], fontsize=13)
plt.yticks([0.5,1.5], ['No Split','Split'], fontsize=13, rotation=0)
plt.xlabel('Predicted', fontsize=14, weight='bold')
plt.ylabel('Actual', fontsize=14, weight='bold')
plt.title('Confusion Matrix - Ribbon Splitting Classifier', fontsize=15, weight='bold')
plt.tight_layout()
plt.savefig("GPR_NN_TL_confusionmat.pdf", format='pdf', bbox_inches='tight')
plt.show()


# ============================================================
# FINAL STAGE-1 FULL-DATA TRAINING
# FINAL REGRESSION MODEL
base_model_final = build_model(X_scaled.shape[1])
# initialize from pretrained synthetic+real model
base_model_final.set_weights(model.get_weights())
# apply transfer learning
final_regression_model = tl_model(base_model_final)
early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
# train on FULL ORIGINAL EXP DATASET
final_regression_model.fit(X_scaled, Y, epochs=300, batch_size=16, callbacks=[early_stop], verbose=1, shuffle=True)
print("Final Stage-1 regression model trained.")

# FINAL CLASSIFICATION MODEL
base_split_final = build_split_model(X_scaled.shape[1])
base_split_final.set_weights(split_model.get_weights())
final_split_model = tl_split_model(base_split_final)
early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
final_split_model.fit(X_scaled, Y_cls, epochs=300, batch_size=16, callbacks=[early_stop], verbose=1, shuffle=True)
print("Final Stage-1 classification model trained.")


#---------------------Scatter plot-----------------------
# Use seaborn's whitegrid style
sns.set(style='whitegrid', context='talk', palette='colorblind')

# Plot settings
fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
titles = ['Density Prediction', 'Thickness Prediction']
y_labels = ['Predicted Density', 'Predicted Thickness']
x_labels = ['True Density', 'True Thickness']
r2_scores_j = [np.mean(r2s_dscores_val), np.mean(r2s_tscores_val)]
actuals = np.column_stack((actuals_d, actuals_t))
predictuals = np.column_stack((predictuals_d, predictuals_t))
colors = sns.color_palette("colorblind")

for i in range(2):
    #Scatter plot
    axs[i].scatter(actuals[:,i], predictuals[:,i],
                   alpha=0.7, edgecolor='k', s=60, color=colors[i])

    # Ideal fit line
    min_val = min(actuals[:,i].min(), predictuals[:,i].min())
    max_val = max(actuals[:,i].max(), predictuals[:,i].max())
    axs[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')

    # Labels, titles, and R² annotation
    axs[i].set_title(titles[i], fontsize=16)
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

plt.tight_layout()
plt.savefig("GPR_NN_TL_scatterR.pdf", format='pdf', bbox_inches='tight')
plt.show()

#-------------------------QQ plots--------------------------
# Create QQ plots in 1x2 layout
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# QQ plot for output 1
stats.probplot(residuals_d, dist="norm", plot=axs[0])
axs[0].set_title("QQ Plot of Residuals (Ribbon Density)")

# QQ plot for output 2
stats.probplot(residuals_t, dist="norm", plot=axs[1])
axs[1].set_title("QQ Plot of Residuals (Ribbon Thickness)")

plt.tight_layout()
plt.savefig("GPR_NN_TL_QQplotR.pdf", format='pdf', bbox_inches='tight')
plt.show()


"""Save GPR and final TL model"""
# Assume you have a trained GPR model called `gpr with scaler`
joblib.dump(gpr, 'gprR.pkl')  # Save
joblib.dump(scaler_X, "gpr_scalerR.pkl")
joblib.dump(gpc, 'gpcR.pkl')

# SAVE FINAL STAGE-1 MODELS
# ---------------- Regression NN ----------------
final_regression_model.save('final_stage1_regression_model.keras')
# ---------------- Classification NN ----------------
final_split_model.save('final_stage1_split_model.keras')

print("\nFinal Stage-1 models and scalers saved.\n")
