
import Levenshtein
classes_org = ["no-sq-sup", "no-sq-baz", "no-sq-int", "no-gl-enc",
               "mz-sq-sup", "mz-sq-baz", "mz-sq-int", "mz-gl-enc",
               "dz-sq-sup", "dz-sq-baz", "dz-sq-int", "dz-gl-enc", "artefact", "placard"]


def get_cell_type(cell_type):
    lev_dist_min = 1000
    best_key = None
    for class_type in classes_org:
        lev_dist = Levenshtein.ratio(cell_type, class_type)
        if 1 - lev_dist < lev_dist_min:
            lev_dist_min = 1 - lev_dist
            best_key = class_type
            if lev_dist_min == 0:
                break

    return best_key