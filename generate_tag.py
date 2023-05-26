import os


def read_annotations(file_path):
    annotations = {}
    with open(file_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split('\t')
            annotations[parts[0]] = parts[1]
    return annotations


def revise_labels(wnids_file, annotations_file, output_file):
    with open(wnids_file, 'r') as file:
        wnids = file.read().splitlines()

    annotations = read_annotations(annotations_file)

    revised_labels = {}
    for idx, wnid in enumerate(wnids):
        revised_labels[wnid] = idx

    with open(output_file, 'w') as file:
        for filename, wnid in annotations.items():
            if wnid in revised_labels:
                revised_label = revised_labels[wnid] + 1
                file.write(f"{filename}\t{revised_label}\n")


wnids_file = 'wnids.txt'
annotations_file = 'val_annotations.txt'
output_file = 'new_labels.txt'

revise_labels(wnids_file, annotations_file, output_file)
