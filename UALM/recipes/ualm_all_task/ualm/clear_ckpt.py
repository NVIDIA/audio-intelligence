# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os 
import shutil
from glob import glob

def clear_checkpoint_folder(folder_path, interval, latest, dry_run=True):
    assert interval > 0, "interval must be greater than 0"
    assert latest > 0, "latest must be greater than 0"

    get_step_num = lambda path: int(path.split('/')[-1].split('_')[-1])

    subfolders = glob(os.path.join(folder_path, 'step_*'))
    subfolders.sort(key=get_step_num)

    # delete the incomplete checkpoints
    for subfolder in subfolders:
        if "latest" not in os.listdir(subfolder):
            print(f"{subfolder} is not complete; deleting...")
            if not dry_run:
                shutil.rmtree(subfolder)
            
    # if the step_num is multiple of interval or the largest a few, keep the checkpoint
    subfolders = glob(os.path.join(folder_path, 'step_*'))
    subfolders.sort(key=get_step_num)

    all_step_nums = [get_step_num(subfolder) for subfolder in subfolders]
    all_step_nums.sort()
    latest = min(latest, len(all_step_nums))
    valid_step_nums = [num for num in all_step_nums if num % interval == 0] + all_step_nums[-latest:]
    valid_step_nums = sorted(list(set(valid_step_nums)))

    for subfolder in subfolders:
        if get_step_num(subfolder) not in valid_step_nums:
            print(f"{subfolder} is not a valid checkpoint; deleting...")
            if not dry_run:
                shutil.rmtree(subfolder)
        else:
            print(f"{subfolder} is a valid checkpoint; keeping...")


def clear_checkpoint_for_exp(interval, latest, dry_run=True):
    folder_paths = glob(os.path.join(
        "exp",
        "*",
        "checkpoints",
    ))
    for folder_path in folder_paths:
        clear_checkpoint_folder(folder_path, interval, latest, dry_run)


def main(interval, latest, dry_run):
    clear_checkpoint_for_exp(interval, latest, dry_run)


if __name__ == "__main__":
    main(interval=10000, latest=3, dry_run=False)