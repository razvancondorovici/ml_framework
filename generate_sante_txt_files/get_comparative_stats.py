
import argparse
from csv import excel
import copy
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import shutil


def visualize_filtered_data(features_dict, title):

    all_files_pkl = list(features_dict.keys())
    all_classes_pkl = [file[-12:-4] for file in all_files_pkl]

    classes = list(set(all_classes_pkl))
    print(classes)
    labels_list = []
    cell_class_list = []
    for filename, features in features_dict.items():
        # Extract class from filename (last 8 characters before extension)
        class_label = filename[-12:-4]
        class_label = 'abnormal' if class_label[0] == 'z' else 'normal'
        cell_class = filename[-13:-4]
        cell_class_list.append(cell_class)
        labels_list.append(class_label)

    # y = np.array(labels_list)
    # plot_hist(y, title)
    # y = np.array(cell_class_list)
    # plot_hist(y, title)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    y = np.array(labels_list)
    plot_hist(y, "Labels Histogram", axes[0])
    y = np.array(cell_class_list)
    plot_hist(y, "Cell Class Histogram", axes[1])

    plt.tight_layout()
    plt.title(title)
    plt.savefig(f"{title}.png", dpi=300)


def plot_hist(y, title, ax=None):

    from collections import Counter
    # Count occurrences of each unique label
    counts = Counter(y)
    # Extract labels and their frequencies
    unique_labels = list(counts.keys())
    frequencies = list(counts.values())
    # plt.figure(figsize=(8, 4))

    bars = ax.bar(unique_labels, frequencies)

    # Add count labels above bars
    for bar, freq in zip(bars, frequencies):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            str(freq),
            ha='center', va='bottom'
        )

    ax.set_xlabel("Labels")
    ax.set_ylabel("Occurrences")
    ax.set_title(title)


def main():

    args = parser.parse_args()
    no_score_csv_path = args.no_score_csv_path
    sus_score_csv_path = args.sus_score_csv_path

    no_score_df = pd.read_csv(no_score_csv_path)
    no_score_df_matrix = no_score_df.loc[:, ["id", "combined_score"]]
    sus_score_df = pd.read_csv(sus_score_csv_path)
    sus_score_df_matrix = sus_score_df.loc[:, ["id", "combined_score"]]
    score_df = pd.concat([no_score_df_matrix, sus_score_df_matrix], ignore_index=True, sort=False)
    keys = score_df["id"].tolist()
    values = score_df["combined_score"].tolist()
    d = dict(zip(keys, values))
    visualize_filtered_data(d, f"Histogram of Label Occurrences - after filtering with Th = {args.th}")
    with open(os.path.join("generate_sante_txt_files", "features_normal_v2.pkl"), 'rb') as f:
        features_dict_normal = pickle.load(f)

    with open(os.path.join("generate_sante_txt_files", "features_suspecte_v2.pkl"), 'rb') as f:
        features_dict_suspecte = pickle.load(f)

    features_dict = {**features_dict_normal, **features_dict_suspecte}
    visualize_filtered_data(features_dict, f"Histogram of Label Occurrences - before filtering")

    # Se identifica caile datelor brute
    raw_dir = args.raw_dir
    filtered_relative_paths = [item.split("_")[0] + os.sep + "_".join(item.split("_")[1:]) for item in keys]
    data_output_dir = args.new_data_dir
    os.makedirs(data_output_dir, exist_ok=True)
    # data_output_dir = os.path.join(data_output_dir + "_" + args.th)
    # Se copiaza in alt dir de tipul "generating once again"

    original_df = pd.DataFrame(columns=[
        "id_patient", "id_slide_crop",
        "slide_score", "inflammation_score", "cell_name", "id_path"])
    inter_df = copy.deepcopy(original_df)
    inter_df.reset_index(drop=True, inplace=True)
    new_data = pd.DataFrame(columns=original_df.columns)
    dict_anno = {}
    dict_anno["name"] = []
    dict_anno["anno"] = {}
    for rel_path in filtered_relative_paths:
        src = os.path.join(raw_dir, rel_path)
        dst = os.path.join(data_output_dir, rel_path)
        # Create the folder structure in the output directory
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        count_cells = 0
        # Copy the file
        try:
            shutil.copy2(src, dst)
            cell_type_cropped_final = os.path.basename(dst).split(".")[0].split("_")[-1]
            dict_anno["name"] = cell_type_cropped_final
            dict_anno["count"] = -1
            count_cells += 1
            new_data.loc[count_cells, 'cell_name'] = cell_type_cropped_final
            new_data.loc[count_cells, 'id_patient'] = os.path.dirname(dst).split(os.sep)[-1]
            new_data.loc[count_cells, 'id_slide_crop'] = os.path.basename(dst).split(".")[0].split("_")[0]
            new_data.loc[count_cells, 'id_path'] = dst
            inter_df = pd.concat([inter_df, new_data])
            inter_df.reset_index(drop=True, inplace=True)

        except Exception as exp:
            print(exp)

    csv = os.path.join(data_output_dir, "partition_info.csv")
    inter_df.to_csv(csv, index=False)
    # Se creaza txt files


if __name__ == "__main__":
    print("=" * 15 + "Start mapping old anno to the corrected ones" + "=" * 15)
    parser = argparse.ArgumentParser(prog='Mapping cells annotations')
    parser.add_argument('--no_score_csv_path', type=str, help='Path to the csv_file')
    parser.add_argument('--sus_score_csv_path', type=str, help='Path to the csv_file')
    parser.add_argument('--output_dir', type=str, help='Path to the Pap smear slide to dump data')
    parser.add_argument('--raw_dir', type=str, help='')
    parser.add_argument('--new_data_dir', type=str, help='')
    parser.add_argument('--th', type=str, help='')

    main()