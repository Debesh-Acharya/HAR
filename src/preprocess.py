import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# ===============================
# Paths and Constants
# ===============================
DATA_PATHS = [
    "data/PAMAP2_Dataset/Protocol",
    "data/PAMAP2_Dataset/Optional"
]

WINDOW_SIZE = 128
STEP_SIZE = 64

col_names = ["timestamp", "activityID", "heart_rate"] + [
    f"imu{i}_{axis}" for i in range(1, 10)
    for axis in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
]

# Exclude imu9 gyroscope features due to NaNs
exclude_features = {"imu9_gyro_x", "imu9_gyro_y", "imu9_gyro_z"}

feature_columns = [
    c for c in col_names
    if (("acc" in c or "gyro" in c or "heart_rate" in c) and c not in exclude_features)
]

# ActivityID to zero-based class index mapping (from documentation)
activity_map = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
    9: 7, 10: 8, 11: 9, 12: 10, 13: 11, 16: 12,
    17: 13, 18: 14, 19: 15, 20: 16, 24: 17
}

# ===============================
# Functions
# ===============================
def load_subject(file_path):
    df = pd.read_csv(
        file_path, sep=' ', header=None, names=col_names, skipinitialspace=True
    )
    df.ffill(inplace=True)
    df.bfill(inplace=True)  # fill backward NaNs for reliable data

    # Print NaN counts per feature column to debug
    nan_counts = df[feature_columns].isna().sum()
    print(f"NaN counts per feature column in {file_path}:")
    print(nan_counts[nan_counts > 0])

    # Filter to only include relevant activities (exclude 0 and those not in map)
    df = df[df["activityID"].isin(activity_map.keys())]
    # Map to contiguous class indices
    y = df["activityID"].map(activity_map).values
    X = df[feature_columns].values
    return X, y

def create_windows(X, y, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    X_windows, y_windows = [], []
    for start in range(0, len(X) - window_size, step_size):
        end = start + window_size
        X_windows.append(X[start:end])
        y_window = np.bincount(y[start:end]).argmax()
        y_windows.append(y_window)
    return np.array(X_windows), np.array(y_windows)

# ===============================
# Load all subjects
# ===============================
subject_files = []
for path in DATA_PATHS:
    subject_files.extend(glob.glob(os.path.join(path, "*.dat")))

X_all = []
y_all = []

for file in subject_files:
    print(f"Processing {file}...")
    X_subj, y_subj = load_subject(file)
    if len(X_subj) == 0:
        print(f"Skipping {file} due to no valid activities.")
        continue
    X_w, y_w = create_windows(X_subj, y_subj)
    X_all.append(X_w)
    y_all.append(y_w)

if len(X_all) == 0:
    raise ValueError("No valid data found after filtering. Check dataset and filtering criteria.")

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print("All subjects combined:", X_all.shape, y_all.shape)

# ===============================
# Normalize features
# ===============================
num_features = X_all.shape[2]
X_all_reshaped = X_all.reshape(-1, num_features)

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all_reshaped)
X_all_scaled = X_all_scaled.reshape(-1, WINDOW_SIZE, num_features)

# Remove sample windows that contain any NaNs
mask = ~np.isnan(X_all_scaled).any(axis=(1, 2))
X_all_scaled = X_all_scaled[mask]
y_all = y_all[mask]

print("Any NaNs in cleaned X_all_scaled?", np.isnan(X_all_scaled).any())
print("Any infinite values in cleaned X_all_scaled?", np.isinf(X_all_scaled).any())
print("Final shape X_all_scaled:", X_all_scaled.shape)
print("Final shape y_all:", y_all.shape)
print("Final label classes:", np.unique(y_all))

# Ensure output directory exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Save scaler
with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ===============================
# Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_all_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Save processed data
with open(os.path.join(models_dir, "preprocessed_data.pkl"), "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

print("Preprocessing complete.")
