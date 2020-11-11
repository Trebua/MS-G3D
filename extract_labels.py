import os

PATH_TO_LABEL_FILES = "./labels/"


def __convert_to_int(string):
    return int(string) if string.isnumeric() else string


def get_label_info():
    file_labels = {}
    file_fps = {}

    files = os.list_dir(PATH_TO_LABEL_FILES)
    for file_name in files:
        with open(PATH_TO_LABEL_FILES + file_name) as f:
            _ = f.readline()  # Header
            for line in f:
                info = line.split(" ")
                id, label, fps = [__convert_to_int(item) for item in info if item != ""]
                file_labels[id] = label
                file_fps[id] = fps
    return file_labels, file_fps
