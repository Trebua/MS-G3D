#!/bin/sh
python3 main.py --config ./config/kinetics-skeleton/train_joint.yaml --work-dir work_dir/kinetics/msg3d-joint
python3 main.py --config ./config/kinetics-skeleton/train_bone.yaml --work-dir work_dir/kinetics/msg3d-bone