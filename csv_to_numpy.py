import os
import numpy as np
from extract_labels import get_label_info

# Script to convert the CIMA csv files into the appropriate npy and json format to use the gen_data from MS-G3D
# The label files looks like: (["file_names"], ["values"])

files = os.listdir("./coordinates")  # This file should be placed under /dataset/cima/raw/

# shape = (N, DIM, T, V, M))
N = len(files)
DIM = 2  # Dimensions of coordinates
T = 300  # How long the samples should be
V = 19  # Features
M = 1  # Number of people
X_index, Y_index = 0, 1

sample_arr = np.zeros((N, DIM, T, V, M))

for n, file_name in enumerate(files):
    with open("./coordinates/" + file_name) as csv_file:
        header = csv_file.readline().split(",")

        x_arr = np.zeros((T, V, M))
        y_arr = np.zeros((T, V, M))
        for line in csv_file:
            frame_data = line.split(",")
            try:
                t = int(frame_data[0]) - 1
            except ValueError:
                continue

            if t >= T:
                break

            mat_x = np.array([])
            mat_y = np.array([])
            for j in range(1, len(frame_data)):
                val = float(frame_data[j])
                if j % 2 == 0:  # Add to Y
                    mat_y = np.append(mat_y, float(val))
                else:
                    mat_x = np.append(mat_x, float(val))

            mat_x = mat_x.reshape([V, M])
            mat_y = mat_y.reshape([V, M])

            x_arr[t, :, :] = mat_x
            y_arr[t, :, :] = mat_y
        sample_arr[n, X_index, :, :, :] = x_arr
        sample_arr[n, Y_index, :, :, :] = y_arr

print(sample_arr)