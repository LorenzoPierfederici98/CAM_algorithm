#     Exam project for the CMEPDA course.
#     Copyright (C) 2024  Lorenzo Pierfederici
#     l.pierfederici@studenti.unipi.it
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

# pylint: disable=C0103
# pylint: disable=W1203


"""Module implementing the CAM algorithm functions."""


import logging
from multiprocessing.sharedctypes import RawArray
import numpy as np
import matplotlib.pyplot as plt

from ant_constructor import Ant
from environment_constructor import ImageData
from watershed import ground_truth, image_segmenter


logging.basicConfig(
    format="%(asctime)s:%(levelname)s: %(message)s",
    filename="../results/log_results.txt",
    filemode="w",
    level=logging.INFO,
    encoding="utf-8",
)


def set_colony(anthill_coordinates, image_matrix):
    """Initializes the ant colony from a voxel given by the user.
    All the 26 first-neighbouring voxels are then occupied by an ant.

    Args
    ----
    anthill_coordinates : list[int]
        The coordinates of the first ant, chosen by the user.

    image_matrix : ndarray
        The matrix of the image to be segmented.

    Returns
    -------
    colony : list[obj]
        The list containing all the 27 ants of the colony.

    anthill_voxel : list[int]
        The coordinates of the first ant, chosen by the user.

    Raises
    ------
    IndexError
        If the input coordinates are not valid.
    """

    if (
        anthill_coordinates[0] >= image_matrix.shape[0]
        or anthill_coordinates[1] >= image_matrix.shape[1]
        or anthill_coordinates[2] >= image_matrix.shape[2]
        or (np.array(anthill_coordinates) < 0).any()
    ):
        print("Invalid coordinates.\n")
        raise IndexError

    anthill_voxel = anthill_coordinates
    first_ant = Ant(image_matrix, anthill_voxel)
    first_ant_neighbours = first_ant.find_first_neighbours()
    colony = [Ant(image_matrix, list(elem)) for elem in first_ant_neighbours]
    colony = colony[: len(colony) // 2] + [first_ant] + colony[len(colony) // 2 :]
    return colony, anthill_voxel


def update_pheromone_map(ant_worker):
    """Updates the pheromone map and signals the ant voxel as occupied to the colony.

    Args
    ----
    ant_worker : obj
        The ant member of the ant colony.

    Returns
    -------
    pheromone : float
        The value of pheromone released in the ant voxel.
    """

    arr = np.frombuffer(np_x, dtype=np.float32).reshape(np_x_shape)
    pheromone = ant_worker.pheromone_release(ant_worker.voxel_coordinates)
    [x, y, z] = ant_worker.voxel_coordinates
    arr[x, y, z, 0] += pheromone + ant_worker.eta
    arr[x, y, z, 1] = 1
    return pheromone - ant_worker.eta


def find_next_voxel(ant_worker):
    """Finds the next destination of the ant and signals the ant voxel as free to the colony.

    Args
    ----
    ant_worker : obj
        The ant member of the ant colony.

    Returns
    -------
    next_vox : list[int]
        The coordinates of the ant destination.
    """

    arr = np.frombuffer(np_x, dtype=np.float32).reshape(np_x_shape)
    [x, y, z] = ant_worker.voxel_coordinates
    first_neigh = ant_worker.find_first_neighbours()
    next_vox = ant_worker.evaluate_destination(first_neigh, arr)
    arr[x, y, z, 1] = 0
    if len(next_vox) != 0:
        arr[next_vox[0], next_vox[1], next_vox[2], 1] = 1
    return next_vox


def plot_display(
    a_ratio,
    image_matrix,
    pheromone_matrix,
    anthill_coordinates,
):
    """Displays the axial view of the image matrix, the axial, coronal
    and sagittal views of the pheromone map; the z, x and y slices of
    the different views correspond to those ones of the anthill location.

    Args
    ----
    a_ratio : dict
        Dictionary of values that preserve the
        aspect ratio of the different slices.

    image_matrix : ndarray
        The matrix of the image to be segmented.

    pheromone_matrix : ndarray
        The matrix of the pheromone map built by the ants.

    anthill_coordinates : list[int]
        The coordinates of the voxel chosen as the anthill position,
        used to display different slices of the pheromone map.
    """

    _, ax = plt.subplots(2, 2, figsize=(10, 7))
    norm = "symlog"
    cmap = "gray"

    ax[0][0].set_aspect(a_ratio["axial"])
    plot_1 = ax[0][0].imshow(image_matrix[..., anthill_coordinates[2]], cmap="gray")
    ax[0][0].set_title("Original image, axial view")
    plot_2 = ax[0][1].imshow(
        pheromone_matrix[..., anthill_coordinates[2]], norm=norm, cmap=cmap
    )

    ax[0][1].set_title("Pheromone map, axial view")
    plot_3 = ax[1][0].imshow(
        pheromone_matrix[:, anthill_coordinates[1], :], norm=norm, cmap=cmap
    )
    ax[0][1].plot(anthill_coordinates[1], anthill_coordinates[0], "ro", label="Anthill")
    ax[0][1].legend()

    ax[1][0].set_title("Pheromone map, coronal view")
    plot_4 = ax[1][1].imshow(
        pheromone_matrix[anthill_coordinates[0], ...], norm=norm, cmap=cmap
    )
    ax[1][1].set_title("Pheromone map, sagittal view")

    for plot, ax in zip(
        [plot_1, plot_2, plot_3, plot_4], [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
    ):
        plt.colorbar(plot, ax=ax)

    plt.tight_layout()
    plt.savefig("../results/CAM_results.png")


def pool_initializer(pheromone_matrix_shared, pheromone_matrix_shape):
    """Function to be used as the Pool initializer to share the
    pheromone map between the processes instantiated by Pool.map.
    Sets the pheromone map and its shape as global variables.

    Args
    ----
    pheromone_matrix_shared : ndarray
        The RawArray from multiprocessing used to share the pheromone map.

    pheromone_matrix_shape : tuple[int]
        The shape of the pheromone map.
    """

    global np_x
    np_x = pheromone_matrix_shared
    global np_x_shape
    np_x_shape = pheromone_matrix_shape


def dictionaries(image_voxels, image_matrix, pheromone_matrix):
    """Computes the dictionaries of the voxels part of the image,
    the voxels visited by the ants and the intersection between them.

    Args
    ----
    image_voxels : ndarray
        The voxels which are part of the image.

    image_matrix : ndarray
        The image matrix.

    pheromone_matrix : ndarray
        The pheromone map.

    Returns
    -------
    visited_voxels_dict : dict
        The dictionary whose keys are the coordinates of the voxels
        visited by the ants, the values are the pheromone quantities
        in the voxels.

    common_dict : dict
        The dictionary of voxels which are both part of the image matrix
        and among those ones visited by the ants.
    """

    visited_voxs = np.unique(
        np.transpose(np.array(np.nonzero(pheromone_matrix))).reshape(-1, 3),
        axis=0,
    )
    image_voxels_dict = {
        f"{list(elem)}": int(value)
        for elem, value in zip(
            image_voxels,
            image_matrix[image_voxels[:, 0], image_voxels[:, 1], image_voxels[:, 2]],
        )
    }
    visited_voxels_dict = {
        f"{list(elem)}": int(value)
        for elem, value in zip(
            visited_voxs,
            pheromone_matrix[
                visited_voxs[:, 0], visited_voxs[:, 1], visited_voxs[:, 2]
            ],
        )
    }
    common_dict = {}
    common_keys = set(image_voxels_dict).intersection(visited_voxels_dict)
    for key in common_keys:
        common_dict[key] = visited_voxels_dict[key]
    logging.info(f"Image voxels: {image_voxels.shape[0]}\n")
    logging.info(f"Visited voxels: {visited_voxs.shape[0]}\n")
    return visited_voxels_dict, common_dict


def metrics(image_voxels, visited_voxels_dict, common_dict, pheromone_threshold):
    """Provides the algorithm evaluation matrics: sensitivity, exploration level and
    contamination level as functions of the pheromone threshold.
    
    Args
    ----
    image_voxels : ndarray
        The voxels which are part of the image.

    visited_voxels_dict : dict
        The dictionary of the voxels visited by the ants.

    common_dict : dict
        The dictionary of voxels which are both part of the
        image matrix and among those ones visited by the ants.
    
    pheromone_threshold : ndarray
        The array of pheromone values used as thresholds to
        evaluate the metrics.

    Returns
    -------
    sensitivity : ndarray
        Ratio of the number of segmented voxels (i.e with a pheromone value greater than
        the threshold) part of the image and the total number of image voxels, one
        for every pheromone threshold value.

    expl_level : ndarray
        Ratio of the number of segmented voxels and the total number of image voxels.

    cont_level : ndarray
        Difference of the sensitivity and expl_level values at a certain threshold
        i.e the percentage of segmented voxels which are not part of the image.
    """

    sensitivity = np.zeros(len(pheromone_threshold))
    expl_level = np.zeros(len(pheromone_threshold))
    cont_level = np.zeros(len(pheromone_threshold))

    for i, val in enumerate(pheromone_threshold):
        temp_common_dict = dict(
            (key, value) for key, value in common_dict.items() if value >= val
        )
        temp_visited_dict = dict(
            (key, value) for key, value in visited_voxels_dict.items() if value >= val
        )
        sensitivity[i] = len(temp_common_dict) / image_voxels.shape[0]
        expl_level[i] = len(temp_visited_dict) / image_voxels.shape[0]
        cont_level[i] = expl_level[i] - sensitivity[i]
    return sensitivity, expl_level, cont_level


def statistics(ants_number, args_parser, mean_list, image_matrix, pheromone_matrix):
    """Provides and displays the statistics about the run.

    Args
    ----
    ants_number : list[int]
        The list of the number of ants per cycle.

    args_parser : obj
        Namespace of ArgumentParser.

    image_matrix : ndarray
        The image matrix.

    mean_list : list[float]
        The pheromone mean per iteration.

    pheromone_matrix : ndarray
        The pheromone map.
    """

    if args_parser.cmd == "dicom":
        image_voxels, thresh_mean = ground_truth(image_matrix)
        pheromone_threshold = np.linspace(
            np.amin(pheromone_matrix),
            np.amax(pheromone_matrix),
            100,
        )

    else:
        image_voxels = np.transpose(np.array(np.nonzero(image_matrix))).reshape(-1, 3)
        pheromone_threshold = np.linspace(
            np.amin(pheromone_matrix),
            np.amax(pheromone_matrix),
            int(np.amax(pheromone_matrix) / 50.01),
        )


    visited_voxels_dict, common_dict = dictionaries(
        image_voxels, image_matrix, pheromone_matrix
    )

    sensitivity, expl_level, cont_level = metrics(
        image_voxels, visited_voxels_dict, common_dict, pheromone_threshold
    )

    if args_parser.cmd == "dicom":
        index = np.where(pheromone_threshold >= 10 * thresh_mean + 0.01)[0][0]
    else:
        index = np.where(pheromone_threshold >= 50.01)[0][0]

    print(pheromone_threshold.shape)
    print(sensitivity[:10])
    print(cont_level[:10])
    print(pheromone_threshold[:10])

    logging.info(f"Pheromone threshold: {pheromone_threshold[index]:.1f}\n")
    logging.info(f"Sensitivity: {100 * sensitivity[index]:.1f} %\n")
    logging.info(f"Expl. level: {100 * expl_level[index]:.1f} %\n")
    logging.info(f"Cont. level: {100 * cont_level[index]:.1f} %\n")

    _, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0][0].plot(ants_number)
    ax[0][0].set_title("Number of ants per cycle")
    ax[0][1].plot(mean_list)
    ax[0][1].set_title("Average pheromone release per cycle")
    ax[1][0].plot(pheromone_threshold / 1000, sensitivity, label="S")
    ax[1][0].plot(pheromone_threshold / 1000, expl_level, label="E")
    ax[1][0].set_xlabel("Pheromone threshold x$10^3$")
    ax[1][0].legend()
    ax[1][0].set_title("S and E vs pheromone threshold")
    ax[1][1].plot(cont_level, sensitivity, marker="o", linestyle="")
    ax[1][1].set_ylabel("S")
    ax[1][1].set_xlabel("C")
    ax[1][1].set_title("S vs C")
    plt.tight_layout()
    plt.savefig("../results/CAM_statistics.png")


def set_image_and_pheromone(args_parser):
    """Instantiates the image matrix and the pheromone map. A RawArray
    of multiprocessing.sharedctypes is first created and then used as a buffer
    to share the pheromone map as a numpy array between processes.

    Args
    ----
    args_parser : obj
        Namespace of ArgumentParser.

    Returns
    -------
    image_matrix : ndarray
        The image_matrix.

    a_ratio : dict
        Dictionary of values that preserve the
        aspect ratio of the different slices.

    pheromone_map_init : ndarray
        The initialized pheromone map.

    pheromone_shared_ : RawArray[float]
        The buffer used to share the pheromone map between processes.

    pheromone_matrix : ndarray
        The pheromone map which will be deployed in the algorithm.
    """

    a_ratio = {"axial": 1, "sagittal": 1, "coronal": 1}

    if args_parser.cmd == "dicom":
        #extrema = [168, 415, 284, 460, 257, 277]
        extrema = [138, 475, 234, 600, 257, 277]
        image_cropped, a_ratio = ImageData.image_from_file(
            args_parser.file_path, extrema
        )
        image_matrix = image_segmenter(image_cropped)
        imagedata = ImageData(image_matrix.shape)
    
    elif args_parser.cmd == "cube":
        imagedata = ImageData(args_parser.matrix_dimensions)
        image_matrix = imagedata.create_cube(
            args_parser.center_coordinates, args_parser.cube_length
        )

    elif args_parser.cmd == "sphere/ellipsoid":
        imagedata = ImageData(args_parser.matrix_dimensions)
        image_matrix = imagedata.create_sphere_ellipsoid(
            args_parser.center_coordinates, args_parser.radius, args_parser.semi_axes
        )

    elif args_parser.cmd == "donut":
        imagedata = ImageData(args_parser.matrix_dimensions)
        image_matrix = imagedata.create_donut(
            args_parser.center_coordinates, args_parser.radius
        )

    pheromone_map_init_ = imagedata.initialize_pheromone_map()
    global np_x_shape
    np_x_shape = pheromone_map_init_.shape
    pheromone_shared_ = RawArray("f", np.array(pheromone_map_init_.shape).prod().item())
    global np_x
    np_x = pheromone_shared_
    # RawArray used as buffer to share the pheromone_map as np array
    pheromone_matrix = np.frombuffer(pheromone_shared_, dtype=np.float32).reshape(
        pheromone_map_init_.shape
    )
    return (
        image_matrix,
        a_ratio,
        pheromone_map_init_,
        pheromone_shared_,
        pheromone_matrix,
    )
