import os
import glob
import pandas as pd
import numpy as np

DATA_PATHS = [
    "HAR_Project/data/PAMAP2_Dataset/Protocol",
    "HAR_Project/data/PAMAP2_Dataset/Optional"
]

col_names = ["timestamp", "activityID", "heart_rate"] + \
            [f"imu{i}_{axis}" for i in range(1, 10) for axis in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]]

subject_files = []
for path in DATA_PATHS:
    subject_files.extend(glob.glob(os.path.join(path, "*.dat")))

print("Files found:", subject_files)

for file in subject_files:
    df = pd.read_csv(file, sep=' ', header=None, names=col_names, skipinitialspace=True)
    print(f"{file} unique activity IDs: {np.unique(df['activityID'].values)}")
