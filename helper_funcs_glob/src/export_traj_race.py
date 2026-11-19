import numpy as np
import uuid
import hashlib
import os


def get_driving_direction(traj_race: np.ndarray) -> str:
    """
    Determine driving direction ('clockwise' or 'counterclockwise') from trajectory coordinates.

    traj_race: race trajectory with x_m in column 1 and y_m in column 2.
    """
    x = traj_race[:, 1]
    y = traj_race[:, 2]
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if signed_area == 0:
        psi = np.unwrap(traj_race[:, 3])
        return "counterclockwise" if (psi[-1] - psi[0]) > 0 else "clockwise"
    return "counterclockwise" if signed_area > 0 else "clockwise"


def export_traj_race(file_paths: dict,
                     traj_race: np.ndarray) -> None:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function is used to export the generated trajectory into a file. The generated files get an unique UUID and a
    hash of the ggv diagram to be able to check it later.

    Inputs:
    file_paths:     paths for input and output files {ggv_file, traj_race_export, traj_ltpl_export, lts_export}
    traj_race:      race trajectory [s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2]
    """

    # create random UUID
    rand_uuid = str(uuid.uuid4())

    # hash ggv file with SHA1
    if "ggv_file" in file_paths:
        with open(file_paths["ggv_file"], 'br') as fh:
            ggv_content = fh.read()
    else:
        ggv_content = np.array([])
    ggv_hash = hashlib.sha1(ggv_content).hexdigest()

    # write UUID and GGV hash into file
    with open(file_paths["traj_race_export"], 'w') as fh:
        fh.write("# " + rand_uuid + "\n")
        fh.write("# " + ggv_hash + "\n")

    # export race trajectory
    header = "s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2"
    fmt = "%.7f; %.7f; %.7f; %.7f; %.7f; %.7f; %.7f"
    with open(file_paths["traj_race_export"], 'ab') as fh:
        np.savetxt(fh, traj_race, fmt=fmt, header=header)

    # determine driving direction
    direction = get_driving_direction(traj_race)

    # determine destination path for driving direction file
    if "traj_direction_export" in file_paths:
        direction_file_path = file_paths["traj_direction_export"]
    elif "traj_race_export" in file_paths:
        base_dir = os.path.dirname(file_paths["traj_race_export"])
        raceline_filename = os.path.basename(file_paths["traj_race_export"])
        if "_raceline.csv" in raceline_filename:
            dir_filename = raceline_filename.replace("_raceline.csv", "_driving_direction.txt")
        else:
            dir_filename = os.path.splitext(raceline_filename)[0] + "_driving_direction.txt"
        direction_file_path = os.path.join(base_dir, dir_filename)
    else:
        direction_file_path = None

    if direction_file_path is not None:
        with open(direction_file_path, 'w') as fh:
            fh.write(direction + "\n")


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
