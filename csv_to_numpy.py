import os
import numpy as np

# shape = (N, 3, T, V, M))
sample_arr = np.zeros((1, 2, 300, 19, 1))

files = os.listdir("./coordinates")
for i, file_name in enumerate(files):
    with open(file_name) as csv_file:
        header = csv_file.readline().split(",")

        x_arr = np.zeros((4, 19, 1))
        y_arr = np.zeros((4, 19, 1))

        for line in csv_file:
            frame_data = line.split(",")
            frame = int(frame_data[0]) - 1
            if frame >= 300:
                break

            mat_x = np.array([])
            mat_y = np.array([])
            for i in range(1, len(frame_data)):
                val = float(frame_data[i])
                if i % 2 == 0:  # Add to Y
                    mat_y = np.append(mat_y, float(val))
                else:
                    mat_x = np.append(mat_x, float(val))

            mat_x = mat_x.reshape([19, 1])
            mat_y = mat_y.reshape([19, 1])

            x_arr[frame, :, :] = mat_x
            y_arr[frame, :, :] = mat_y

        sample_arr[i, 0, :, :, :] = x_arr
        sample_arr[i, 1, :, :, :] = y_arr
