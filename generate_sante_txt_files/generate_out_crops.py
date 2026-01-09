"""
GENRATING RAW AND FLITERED DATA FROM .JPG AND .XMNL FILES PROVIDED BY THE MEDICAL STAFF
For each annotated object:
   • Normalize the cell label text (strip spaces, unify separators, lowercase).
   • Handle long/irregular labels by mapping them to known categories (e.g., artefact/placard).
   • Crop the bounding box region from the original image, clamp coordinates to image bounds,
     resize to 224x224, and save the crop under:
       <output_dir>/<output_dirname>/<id_patient>/<id_slide_crop>_<count>_<cell_name>.jpg
   • Append metadata (cell_name, id_patient, id_slide_crop, id_path) to an intermediate dataframe.
   • Skip saving/recording when the computed output path already exists in `inter_df` (duplicate check).
- Requires: numpy, pandas, cv2 (OpenCV), PIL, xml.etree.ElementTree, matplotlib, seaborn, copy, os.
"""

import argparse
import copy
import os
import cv2
import re
import Levenshtein
import numpy as np
import pandas as pd
import seaborn as sns

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from utils import get_cell_type

csv_anno_dict = {}

classes = ["id_patient", "id_slide_crop", "inflammation_score", "cell_name"]

original_df = pd.DataFrame(columns=[
    "id_patient", "id_slide_crop",
    "slide_score", "inflammation_score", "cell_name", "id_path"])
inter_df = copy.deepcopy(original_df)
inter_df.reset_index(drop=True, inplace=True)
slide_df = pd.DataFrame(columns=["id_patient", "id_slide_crop", "slide_score",])
slide_df.reset_index(drop=True, inplace=True)
patient_ids = {}


def clean_file_generator(directory, anno_ext):

    def write_to_csv(dir_name, filename):
        # became obsolete to find the inflamation score
        patient_id = int(dir_name.split("-")[0])
        inflamation_score = -1  #  inflamation score non-specified
        return patient_id, inflamation_score

    def get_corresponding_img(img_path_pattern, directory):
        # aceeasi distanta Lev. calculata, dar pentru a gasi perechi de adnotare + imagine
        lev_dist_min = 1000
        img_file_final = None
        jpg_files = list(filter(lambda x: x.endswith(".jpg"), os.listdir(directory)))

        for img_file in jpg_files:
            lev_dist = Levenshtein.ratio(img_path_pattern, img_file)
            if 1 - lev_dist < lev_dist_min:
                lev_dist_min = 1 - lev_dist
                img_file_final = img_file
            if lev_dist_min == 0:
                break

        return img_file_final

    for dirpath, dirnames, filenames in os.walk(directory):
        for i, filename in enumerate(filenames):
            filename_old = filename
            filename = filename.replace(" ", "")
            filename = filename.lower()

            if filename.endswith(anno_ext):
                if "out_crops" in filename: continue
                anno_path = os.path.join(dirpath, filename_old)
                img_path = None
                name = filename.split(".")[0]
                file_idx = re.findall("\d+", name)[0]
                patient_id, inflammation_score = write_to_csv(os.path.basename(dirpath), name)

                img_path_pattern = str(file_idx) + ".jpg"
                img_file_final = get_corresponding_img(img_path_pattern, dirpath)
                if img_file_final:
                    if re.findall("\d+", filename)[0] == re.findall("\d+", img_file_final)[0]:
                        img_path = os.path.join(dirpath, img_file_final)

                csv_anno_dict["id_patient"] = patient_id
                csv_anno_dict["id_slide_crop"] = file_idx
                csv_anno_dict["inflammation_score"] = inflammation_score

                if img_path and anno_path:
                    yield img_path, anno_path

def get_long_cell_type(cell_type, no_field):

    if no_field == 0:
        classes = {"no": ["normal", "normale", "sanatos", "sanatoasa"],
                   "mz": ["metaplaziat", "metaplaziate", "transformat"],
                   "dz": ["displazic", "displazie", "displaziat", "displaziata"]}
    elif no_field == 1:
        classes = {"sq": ["scuamos", "scuamoase", "sq", "scuame"],
                   "gl": ["gl", "glandular", "transformat"]}
    else:
        classes = {"sup": ["sup", "superficial", "superficiala"],
                   "int": ["int", "intermed", "intermediar"],
                   "baz": ["bazala", "bazal"],
                   "enc": ["endoc", "enc", "endom", "end", "placard", "camp", "lambou"]}
    best_similarity = -1

    # loop over all keys and their associated synonyms
    for key, synonyms in classes.items():
        for candidate in synonyms:
            similarity = Levenshtein.ratio(cell_type, candidate)
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key
                if similarity == 1.0:
                    break

    return best_key

def set_slide_score(inter_df):

    lp = inter_df["id_patient"].iloc[-1]
    ls = inter_df.loc[inter_df["id_patient"] == lp, "id_slide_crop"].iloc[-1]

    data = inter_df.loc[(inter_df["id_patient"] == lp) &
                        (inter_df["id_slide_crop"] == ls)]

    slide_df.loc[data.index[0], "id_patient"] = lp
    slide_df.loc[data.index[0], "id_slide_crop"] = ls

    return inter_df


def main(inter_df):

    args = parser.parse_args()
    output_dir = args.output_dir
    dataset_path = args.dirpath
    output_dirname = args.output_dirname
    dataset = clean_file_generator(directory=dataset_path, anno_ext=".xml")
    count_cells = -1

    cols = original_df.columns
    for item in cols:
        if item != 'id_patient' or item != 'id_slide_crop' or item != 'inflammation_score':
            csv_anno_dict.update({item: 0})

    for item in classes:
        csv_anno_dict.update({item: []})
    csv_anno_dict["id_path"] = ""

    dict_anno = {}
    dict_anno["name"] = []
    dict_anno["anno"] = {}
    dict_anno["anno"]["xmin"], dict_anno["anno"]["ymin"], dict_anno["anno"]["xmax"], dict_anno["anno"]["ymax"] = \
        list(), list(), list(), list()

    for idx, (img_path, anno_path) in enumerate(dataset):
        print("File ->", os.path.basename(anno_path))
        tree = ET.parse(anno_path)
        root = tree.getroot()
        from PIL import Image
        img_pil = Image.open(img_path)
        image = np.array(img_pil)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        new_data = pd.DataFrame(columns=original_df.columns)
        count = 0
        for child in root:
            if child.tag == 'object':
                for item in child:
                    if item.tag == 'name':
                        count += 1
                        cell_type = item.text
                        cell_type = cell_type.replace(" ", "")
                        cell_type = cell_type.replace("_", "-")
                        cell_type = cell_type.lower()

                        if len(cell_type.split("*")[0]) > 9:
                            print("scenariu de scris cuvinte")
                            cell_type_old = cell_type.split("*")[0].split("-")
                            if len(cell_type_old) >= 3:
                                if "lambou" in cell_type_old[0] or "placard" in cell_type_old[0] or "camp" in cell_type_old or "placad" in cell_type_old[0]:
                                    cell_type = "placard"
                                elif "art" in cell_type_old[0] or "art" in cell_type_old[1]:
                                    cell_type = "artefact"
                                else:
                                    first_word = get_long_cell_type(cell_type_old[0], 0)
                                    second_word = get_long_cell_type(cell_type_old[1], 1)
                                    third_word = get_long_cell_type(cell_type_old[2], 2)
                                    cell_type = "-".join([first_word, second_word, third_word])
                                print(f"din {cell_type_old} -> {cell_type}")
                            else:
                                if "art" in cell_type_cropped or "camp" in cell_type_cropped or "flora" in cell_type_cropped:
                                    cell_type = "artefact"

                        count_cells += 1
                        cell_type_cropped = cell_type[0:9]
                        if "art" in cell_type_cropped or "camp" in cell_type_cropped or "flora" in cell_type_cropped:
                            cell_type_cropped = "artefact"
                        cell_type_cropped_final = get_cell_type(cell_type_cropped)
                        dict_anno["name"] = cell_type_cropped_final
                        dict_anno["count"] = count
                        print(f"FINAL AVEM {cell_type_cropped_final} DIN {cell_type_cropped}")
                        new_path = os.path.join(output_dir, output_dirname, str(csv_anno_dict['id_patient']),
                         f"{csv_anno_dict['id_slide_crop']}_{dict_anno["count"]}_{dict_anno["name"]}.jpg")

                        if inter_df.loc[(inter_df["id_path"] == new_path)].size != 0:
                            print("!!!!!!!!!!!!duplicate!!!!!!!!!!!!!!!")
                            print("File ->", os.path.basename(anno_path))
                            print("se va continua ...")
                            continue

                        new_data.loc[count_cells, 'cell_name'] = cell_type_cropped_final
                        new_data.loc[count_cells, 'id_patient'] = csv_anno_dict['id_patient']
                        new_data.loc[count_cells, 'id_slide_crop'] = csv_anno_dict['id_slide_crop']
                        new_data.loc[count_cells, 'id_path'] = new_path

                    if item.tag == 'bndbox':

                        dict_anno["anno"]["xmin"] = int(item.find(".//xmin").text.strip(" '\n\""))
                        dict_anno["anno"]["ymin"] = int(item.find(".//ymin").text.strip(" '\t\""))
                        dict_anno["anno"]["xmax"] = int(item.find(".//xmax").text.strip(" '\t\""))
                        dict_anno["anno"]["ymax"] = int(item.find(".//ymax").text.strip(" '\t\""))

                        x_min = int(dict_anno["anno"]["xmin"])
                        y_min = int(dict_anno["anno"]["ymin"])
                        x_max = int(dict_anno["anno"]["xmax"])
                        y_max = int(dict_anno["anno"]["ymax"])

                        # Ensure the coordinates are within the image boundaries
                        h, w = image.shape[:2]
                        x_min = max(0, min(x_min, w - 1))
                        y_min = max(0, min(y_min, h - 1))
                        x_max = max(0, min(x_max, w - 1))
                        y_max = max(0, min(y_max, h - 1))

                        cropped = image[y_min:y_max, x_min:x_max]
                        cropped_resize = cv2.resize(cropped, (224, 224))
                        os.makedirs(os.path.join(output_dir, output_dirname, str(csv_anno_dict['id_patient'])), exist_ok=True)
                        cv2.imwrite(
                            os.path.join(output_dir, str(output_dirname), str(csv_anno_dict['id_patient']),
                            f"{csv_anno_dict['id_slide_crop']}_{dict_anno["count"]}_{dict_anno["name"]}.jpg"),
                            cropped_resize)

        inter_df = pd.concat([inter_df, new_data])
        # scriem ultimul entry
        inter_df.reset_index(drop=True, inplace=True)
        inter_df = set_slide_score(inter_df)

        if inter_df["id_patient"].iloc[-1] not in patient_ids:
            patient_ids[slide_df["id_patient"].iloc[-1]] = []
        else:
            patient_ids[slide_df["id_patient"].iloc[-1]].append(slide_df["slide_score"].iloc[-1])


    print("nr ID-uri ", np.unique(inter_df["id_patient"]))
    cont = 0
    for i in np.unique(inter_df["id_patient"]):
        lp = inter_df.loc[inter_df["id_patient"] == i]
        ls = np.unique(lp["id_slide_crop"])
        cont += len(ls)
    print("numar tablouri", cont)

    csv = os.path.join(output_dir, output_dirname, "partition_info.csv")
    inter_df.to_csv(csv, index=False)

    new_patient_ids = copy.deepcopy(patient_ids)
    sum_of_slides = {}
    for ids, list_slide_scores in new_patient_ids.items():
        sum_of_slides[ids] = len(list_slide_scores)
        if sum(patient_ids[ids]) == 0:
            patient_ids.pop(ids)

    # print("nr ID-uri dupa filtrare", np.unique(patient_ids.keys))
    print("nr ID-uri dupa filtrare", len(patient_ids.keys()))
    print("nr total de tablouri", sum(sum_of_slides.values()))

    ###############################################################################
    plt.figure(figsize=(20, 16))
    from matplotlib.colors import ListedColormap, BoundaryNorm
    df_matrix = pd.DataFrame.from_dict(patient_ids, orient='index')
    cmap = ListedColormap(['green', 'red'])  # 0=green, 1=red

    mask = df_matrix.isna()
    # Main heatmap: 0/1 values
    sns.heatmap(df_matrix, cmap=cmap, norm=None, mask=mask,
                cbar=False, linewidths=0.5, linecolor='black', yticklabels=True)
    # Overlay missing values in black
    ax = sns.heatmap(mask, mask=~mask, cmap=ListedColormap(['black']),
                     cbar=False, linewidths=0.5, linecolor='black', yticklabels=True)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=10)
    # plt.yticks([])
    plt.xlabel('Sample Index')
    plt.ylabel('ID')
    plt.title('Pap Smear Slides Distribution by Risk Status')
    plt.legend(title='Status')
    plt.savefig(os.path.join(output_dir, output_dirname, "risk.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Call pentru partition_proprietary_data
    import subprocess
    out_dir = os.path.join(output_dir, output_dirname)
    print("CALLING partition_proprietary_data.py")
    with subprocess.Popen(
            ["python3", os.path.join("generate_sante_txt_files", "partition_proprietary_data.py"),
                "--root_dir", out_dir,
                "--csv", csv],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"partition_proprietary_data.py failed with exit code {rc}")

    #Filtering data if specified
    if args.filtering:
        print("CALLING data_analysis.py")
        with subprocess.Popen(
                ["python3", "data_analysis.py",
                 "--sus_pkl_files", os.path.join("generate_sante_txt_files", "features_suspecte_v2.pkl"),
                 "--no_pkl_files", os.path.join("generate_sante_txt_files", "features_normal_v2.pkl"),
                 "--Th", str(args.Th)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
        ) as proc:
            for line in proc.stdout:
                print(line, end="")
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"data_analysis.py failed with exit code {rc}")

        print("CALLING get_comparative_stats.py")

        no_file, sus_file = None, None
        for fname in os.listdir("generate_sante_txt_files"):
            m = re.match(r"^review_candidates_(no|sus)_th_(\d+)\.csv$", fname)
            if not m:
                continue
            if m.group(1) == "no":
                no_file = fname
            else:
                sus_file = fname

        with subprocess.Popen(
                ["python3", os.path.join("generate_sante_txt_files",
                                            "get_comparative_stats.py"),
                 "--no_score_csv_path", os.path.join("generate_sante_txt_files", no_file),
                 "--sus_score_csv_path", os.path.join("generate_sante_txt_files", sus_file),
                 "--output_dir", "generate_sante_txt_files",
                 "--raw_dir", out_dir,
                 "--th", str(args.Th),
                 "--new_data_dir", os.path.join(output_dir, "filtered_crops")],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
        ) as proc:
            for line in proc.stdout:
                print(line, end="")
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"get_comparative_stats.py failed with exit code {rc}")

        filtered_dir = os.path.join(output_dir, "filtered_crops")
        csv = os.path.join(filtered_dir, "partition_info.csv")
        print("CALLING partition_proprietary_data.py for training data")
        with subprocess.Popen(
                ["python3", os.path.join("generate_sante_txt_files", "partition_proprietary_data.py"),
                 "--root_dir", filtered_dir,
                 "--csv", csv],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
        ) as proc:
            for line in proc.stdout:
                print(line, end="")
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"partition_proprietary_data.py failed with exit code {rc}")

    # Create txt files either for raw or filtered data
    print("CALLING create_LOOCV_files.py")
    txt_files_path = os.path.join(output_dir, output_dirname, "txt_files") if not args.filtering \
        else os.path.join(output_dir, "filtered_crops", "txt_files")
    with subprocess.Popen(
            ["python3", os.path.join("generate_sante_txt_files", "create_LOOCV_files.py"),
             "--txt_files_path", txt_files_path,
             "--partition_info_csv", csv,
                ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"create_LOOCV_files.py failed with exit code {rc}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Count Number of cells',
        description='Tool for counting the number of cells from a single slide')
    parser.add_argument('--dirpath', type=str, help='Path to the Pap smear slide to be analyzed')
    parser.add_argument('--output_dir', type=str, help='Path to the Pap smear slide to dump figures')
    parser.add_argument('--output_dirname', type=str, help='', default="raw_crops")
    parser.add_argument("--filtering", action="store_true", help="apply filtering")
    parser.add_argument('--Th', type=int, help='filtering threshold', default=1)
    main(inter_df)