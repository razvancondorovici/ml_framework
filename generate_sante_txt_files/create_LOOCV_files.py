"""
PREGATIREA DATELOR PENTRU ANTRENARE IN FISIERE TXT
URMAND STRATEGIA DE ANTRENARE LOOCV
ASIGURAND UN PROCENT MINIM DE DISPLAZII IN FIECARE FISIER
"""


import argparse
import numpy as np
import copy
import pandas as pd
import os
import random


def main():

    args = parser.parse_args()
    out_dir = args.txt_files_path
    os.makedirs(out_dir, exist_ok=True)
    df = args.partition_info_csv
    df = pd.read_csv(df)

    df_matrix = df.loc[:, ["id_patient", "id_slide_crop", "cell_name", "id_path"]]
    no_options = ["no-sq-sup", "no-sq-int", "no-sq-baz", "no-gl-enc", "placard"]
    mz_options = ["mz-sq-sup", "mz-sq-int", "mz-sq-baz", "mz-gl-enc"]
    dz_options = ["dz-sq-sup", "dz-sq-int", "dz-sq-baz", "dz-gl-enc"]

    no_cells = df_matrix.loc[df_matrix['cell_name'].isin(no_options)].reset_index(drop=True)
    dz_cells = df_matrix.loc[df_matrix['cell_name'].isin(dz_options)].reset_index(drop=True)

    no_patiens_ids = list(np.unique(no_cells["id_patient"]))
    dz_patiens_ids = list(np.unique(dz_cells["id_patient"]))

    dz_ids_for_testing, no_ids_for_testing = [], []
    no_cells_copy = copy.deepcopy(no_cells)

    # keep a dynamic list of remaining dysplastic IDs
    dz_patients_remaining = dz_patiens_ids.copy()
    random.shuffle(dz_patients_remaining)

    for i in range(10):
        dz_train_slide_df = pd.DataFrame(columns=df_matrix.columns).reset_index(drop=True)
        dz_test_slide_df = pd.DataFrame(columns=df_matrix.columns).reset_index(drop=True)
        no_train_slide_df = pd.DataFrame(columns=df_matrix.columns).reset_index(drop=True)
        no_test_slide_df = pd.DataFrame(columns=df_matrix.columns).reset_index(drop=True)
        train_txt = open(os.path.join(out_dir, f"training_{i}.txt"), "w")
        test_txt = open(os.path.join(out_dir, f"test_{i}.txt"), "w")

        #################################################################
        # target number of dysplastic cells (~10%)
        test_target = int(0.10 * len(dz_cells))
        current_test_count = 0
        dz_test_ids = []

        while current_test_count < test_target:
            if not dz_patients_remaining:
                dz_patients_remaining = dz_patiens_ids.copy()
                random.shuffle(dz_patients_remaining)
                print(f"Recycling dysplastic IDs for fold {i}")

            # pop next patient ID
            dz_idx = dz_patients_remaining.pop(0)
            id_current = dz_cells.loc[dz_cells["id_patient"] == dz_idx]
            dz_test_slide_df = pd.concat([dz_test_slide_df, id_current])
            dz_test_ids.append(int(dz_idx))
            current_test_count += len(id_current)

        print(f"Fold {i}: picked {len(dz_test_ids)} dysplastic IDs → "
              f"{dz_test_slide_df.shape[0]} cells (~{current_test_count/test_target:.1%} of target)")

        #################################################################
        # build corresponding normal test set (10% or more)
        no_id_current = no_cells_copy.loc[no_cells_copy['id_patient'].isin(dz_test_ids)].reset_index(drop=True)
        no_test_slide_df = pd.concat([no_test_slide_df, no_id_current])
        no_ids_for_testing = list(np.unique(no_test_slide_df['id_patient']))

        test_target_no = int(0.10 * len(no_cells))
        while (no_test_slide_df.shape[0] < test_target_no) and not no_cells_copy.empty:
            next_df = no_cells_copy[~no_cells_copy['id_patient'].isin(no_ids_for_testing)].reset_index(drop=True)
            if next_df.empty:
                break
            next_id = next_df.iloc[-1]['id_patient']
            id_current = no_cells_copy.loc[no_cells_copy["id_patient"] == next_id].reset_index(drop=True)
            no_ids_for_testing.append(next_id)
            no_test_slide_df = pd.concat([no_test_slide_df, id_current])
            no_cells_copy = no_cells_copy[~no_cells_copy['id_patient'].isin([next_id])]

        #################################################################
        test_fold = pd.concat([no_test_slide_df, dz_test_slide_df]).reset_index(drop=True)
        print(f"Fold {i}: test fold = {len(test_fold)} cells, "
              f"{len(np.unique(test_fold['id_patient']))} unique patients")
        print("Dysplastic test IDs:", dz_test_ids)

        for idx in test_fold.values:
            cell = idx[-1]
            att = "normal" if "no" in idx[-2] or "placard" in idx[-2] or "grup" in idx[-2] else "suspecte"
            new_name = os.path.join(att,
                                    os.path.basename(os.path.dirname(cell)) + "_" + os.path.basename(cell))
            test_txt.write(f"{new_name}\n")

        # TRAINING
        id_current_dz = dz_cells.loc[~dz_cells['id_patient'].isin(list(np.unique(test_fold['id_patient'])))].reset_index(
            drop=True)
        id_current_no = no_cells.loc[~no_cells['id_patient'].isin(list(np.unique(test_fold['id_patient'])))].reset_index(
            drop=True)
        no_train_slide_df = pd.concat([id_current_dz, id_current_no])  # img anormale set testare

        print(np.unique(no_train_slide_df["id_patient"]))
        for idx in no_train_slide_df.values:
            cell = idx[-1]
            att = "normal" if "no" in idx[-2] or "placard" in idx[-2] or "grup" in idx[-2] else "suspecte"
            new_name = os.path.join(att,
                                    os.path.basename(os.path.dirname(cell)) + "_" + os.path.basename(cell))
            train_txt.write(f"{new_name}\n")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Count Number of cells',
        description='Tool for counting the number of cells from a single slide')
    parser.add_argument('--txt_files_path', type=str, help='Path where txt files will be created')
    parser.add_argument('--partition_info_csv', type=str, help='Path to the csv containing partition info')

    main()
