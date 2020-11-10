import os
from shutil import rmtree, copy
import numpy as np
import argparse

joints = {
    0:  "Nose",
    1:  "Neck",
    2:  "RShoulder",
    3:  "RElbow",
    4:  "RWrist",
    5:  "LShoulder",
    6:  "LElbow",
    7:  "LWrist",
    8:  "RHip",
    9:  "RKnee",
    10: "RAnkle",
    11: "LHip",
    12: "LKnee",
    13: "LAnkle",
    14: "REye",
    15: "LEye",
    16: "REar",
    17: "LEar"
}
data_path = './data'
kinetics_path = os.path.join(data_path, 'kinetics')
parts = ['train', 'val']
to_remove = [14,15] #LEye and REye
keep_indexes = [i for i in range(0,18) if i not in to_remove]


def create_data_folder():
    if os.path.exists(folder_path):
        print(f'{folder_path} already exists.')
        delete = input(f'Delete {folder_path}? y/n: ')
        if delete == 'y':
            rmtree(folder_path)
            print(f'{folder_path} deleted. Will make a new one.')
        else:
            print('"No" selected, program exits.')
            return False
    os.makedirs(folder_path)
    print(f'{folder_path} created.')
    return True

def copy_labels():
    print()
    suffix = '_label.pkl'
    for part in parts:
        filename = f'{part}{suffix}'
        origin_path = os.path.join(kinetics_path, filename)
        print(f'Copying {origin_path} to {folder_path}..')
        copy(origin_path, folder_path)
        print(f'{folder_path} copied')


# The shape is (100, 3, 300, 18, 2), we want (100, 2, 300, 18, 2) - where the second shape has the last value removed
def remove_confidence(narray):
    return narray[:, :2, :, :, :]

# We want to remove the joins specified in remove_joints
def remove_joints(narray, keep_indexes):
    return narray[:, :, :,keep_indexes,:]

def get_joints_removed():
    j = []
    for i in to_remove:
        j.append(joints[i])
    return ', '.join(j)

def create_joint_data(remove_joint=True, remove_conf=True):
    print('\n')
    print(f'Remove joints: {remove_joint}')
    print(f'Remove confidence: {remove_conf}')
    suffix = '_data_joint.npy'
    for part in parts:
        filename = f'{part}{suffix}'
        origin_path = os.path.join(kinetics_path, filename)
        print(f'Processing {origin_path}')
        data = np.load(origin_path, mmap_mode='r')
        if remove_joint:
            print(f'Removing joints {get_joints_removed()}..')
            data = remove_joints(data, keep_indexes)
            print('Joints removed')
        if remove_conf:
            print('Removing confidence')
            data = remove_confidence(data)
            print('Confidence removed')
        new_file_path = os.path.join(folder_path, filename)
        np.save(new_file_path, data)
        print(f'{new_file_path} saved')

def create_dataset(remove_joint=True, remove_conf=True):
    created_folder = create_data_folder()
    if not created_folder:
        return
    copy_labels()
    create_joint_data(remove_joint=remove_joint, remove_conf=remove_conf)

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Specifies the name of the new dataset folder")
parser.add_argument("-j", "--joints", help="1 removes joints, 0 keeps joints")
parser.add_argument("-c", "--confidence", help="1 removes eyes, 0 keeps eyes")
args = parser.parse_args()
name = args.name
remove_joint = True if int(args.joints) else False
remove_conf = True if int(args.confidence) else False
folder_path = os.path.join(data_path, name)
create_dataset(remove_joint=remove_joint, remove_conf=remove_conf)
