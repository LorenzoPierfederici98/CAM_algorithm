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


"""Module implementing the CAM algorithm."""


import time
import argparse
import logging
import multiprocessing
from multiprocessing.sharedctypes import RawArray
import numpy as np
import matplotlib.pyplot as plt

from ant_constructor import Ant
from environment_constructor import ImageData
from region_growing import ground_truth


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
    arr[x, y, z, 0] += pheromone
    arr[x, y, z, 1] = 1
    return pheromone


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


def statistics(ants_number, args_parser, image_matrix, pheromone_matrix):
    """Provides statistics about the run such as: the number of image voxels,
    the number of voxels visited by the ants, the visited voxels which are also part 
    of the image voxels and the respective number of visits. It also provides the
    algorithm evaluation metrics such as sensitivity, exploration and
    contamination level (defined in the documentation) as functions
    of the pheromone threshold.

    Args
    ----
    ants_number : list[int]
        The list of the number of ants per cycle.

    args_parser : obj
        Namespace of ArgumentParser.

    image_matrix : ndarray
        The image matrix.

    pheromone_matrix : ndarray
        The pheromone map.
    """

    if args_parser.cmd == "dicom":
        _, image_voxels = ground_truth(image_matrix)
    else:
        image_voxels = np.transpose(np.array(np.nonzero(image_matrix))).reshape(-1, 3)

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
    logging.info(
        f"Of which belonging to the image: {len(common_dict)} ({(100 * len(common_dict) / image_voxels.shape[0]):.1f}%)\n"
    )

    pheromone_threshold = np.linspace(
        np.amin(pheromone_matrix), np.amax(pheromone_matrix), 500
    )
    sensitivity = np.zeros(len(pheromone_threshold))
    expl_level = np.zeros(len(pheromone_threshold))
    cont_level = np.zeros(len(pheromone_threshold))

    for index, val in enumerate(pheromone_threshold):
        temp_common_dict = dict(
            (key, value) for key, value in common_dict.items() if value >= val
        )
        temp_visited_dict = dict(
            (key, value) for key, value in visited_voxels_dict.items() if value >= val
        )
        sensitivity[index] = len(temp_common_dict) / image_voxels.shape[0]
        expl_level[index] = len(temp_visited_dict) / image_voxels.shape[0]
        cont_level[index] = expl_level[index] - sensitivity[index]

    _, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0][0].plot(ants_number)
    ax[0][0].set_title("Number of ants per cycle")
    ax[0][1].hist(common_dict.values(), bins=50)
    ax[0][1].set_title("Hist of pheromone values into image objects")
    ax[0][1].set_yscale("log")
    ax[0][1].set_xticks([])
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
    """Instantiates the image from a path given by the user and
    the pheromone map as a four-dimensional numpy array of zeros. A RawArray
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
        extrema = [168, 415, 284, 460, 257, 277]
        image_matrix, a_ratio = ImageData.image_from_file(args_parser.file_path, extrema)
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
        image_matrix = imagedata.create_donut(args_parser.center_coordinates, args_parser.radius)


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Module implementing the CAM algorithm."
    )
    parser.add_argument(
        "-a",
        "--anthill_coordinates",
        help="The anthill voxel position.",
        type=int,
        nargs=3,
        metavar="int",
    )
    parser.add_argument(
        "-n",
        "--n_iteration",
        help="Number of iterations before stopping.",
        type=int,
        metavar="int",
    )
    subparsers = parser.add_subparsers(help="sub-command help", dest="cmd")
    parser_path = subparsers.add_parser("dicom", help="Returns an image from a DICOM folder.")
    parser_path.add_argument(
        "-f", "--file_path", help="The DICOM folder path.", type=str, metavar="str",
    )
    parser_cube = subparsers.add_parser(
        "cube", help="Returns a cube as the image matrix."
    )
    parser_cube.add_argument(
        "-m",
        "--matrix_dimensions",
        help="Image matrix dimensions.",
        type=int,
        nargs=3,
        metavar="int",
    )
    parser_cube.add_argument(
        "-c",
        "--center_coordinates",
        help="The cube center.",
        type=int,
        nargs=3,
        metavar="int",
    )
    parser_cube.add_argument(
        "-l", "--cube_length", help="The cube length.", type=int, metavar="int"
    )
    parser_curve = subparsers.add_parser(
        "sphere/ellipsoid", help="Returns a sphere/ellipsoid as the image matrix."
    )
    parser_curve.add_argument(
        "-m",
        "--matrix_dimensions",
        help="Image matrix dimensions.",
        type=int,
        nargs=3,
        metavar="int",
    )
    parser_curve.add_argument(
        "-c",
        "--center_coordinates",
        help="The center of the figure.",
        type=int,
        nargs=3,
        metavar="int",
    )
    parser_curve.add_argument(
        "-s",
        "--semi_axes",
        help="The semi-axes lengths.",
        type=float,
        nargs=3,
        metavar="float",
    )
    parser_curve.add_argument(
        "-r",
        "--radius",
        help="The radius of the figure.",
        type=int,
        metavar="int",
    )
    parser_donut = subparsers.add_parser(
        "donut",
        help="Returns a donut as the image matrix i.e a sphere with a concentric hole with half external radius as the internal radius.",
    )
    parser_donut.add_argument(
        "-m",
        "--matrix_dimensions",
        help="Image matrix dimensions.",
        type=int,
        nargs=3,
        metavar="int",
    )
    parser_donut.add_argument(
        "-c",
        "--center_coordinates",
        help="The center of the donut.",
        type=int,
        nargs=3,
        metavar="int",
    )
    parser_donut.add_argument(
        "-r",
        "--radius",
        help="The external radius of the donut.",
        type=int,
        metavar="int",
    )
    args = parser.parse_args()

    image, aspect_ratio, pheromone_map_init, pheromone_shared, pheromone_map = (
        set_image_and_pheromone(args)
    )
    # copies pheromone_map_init into pheromone_map
    np.copyto(pheromone_map, pheromone_map_init)

    n_iteration = 0
    energy_death = 1.
    energy_reproduction = 1.3
    ant_number = []
    pheromone_mean_sum = 0
    colony_length = 0

    ant_colony, anthill_position = set_colony(args.anthill_coordinates, image)
    start_time_local = time.perf_counter()

    while len(ant_colony) != 0 and n_iteration <= args.n_iteration:
        print(f"Iter:{n_iteration}\t#ants:{len(ant_colony)}\n")

        chunksize = max(2, len(ant_colony) // 6)
        with multiprocessing.Pool(
            processes=6,
            initializer=pool_initializer,
            initargs=(pheromone_shared, pheromone_map.shape),
        ) as pool:
            released_pheromone = pool.map(
                update_pheromone_map, ant_colony, chunksize=chunksize
            )
            next_voxel = pool.map(find_next_voxel, ant_colony, chunksize=chunksize)

        colony_length += len(ant_colony)
        pheromone_mean_sum += sum(released_pheromone)
        pheromone_mean = pheromone_mean_sum / colony_length

        for i, ant in enumerate(ant_colony):
            if len(next_voxel[i]) == 0:
                del ant_colony[i]
                del released_pheromone[i]
                continue
            ant.voxel_coordinates = next_voxel[i]
            ant.update_energy(released_pheromone[i] / pheromone_mean)

        for i, ant in enumerate(ant_colony):
            if ant.energy < energy_death:
                del ant_colony[i]
                if len(ant_colony) == 0:
                    print("No ants left.\n")
                continue
            if ant.energy > energy_reproduction:
                ant.energy = 1.0 + ant.alpha
                first_neighbours = ant.find_first_neighbours()
                valid_index = np.where(
                    pheromone_map[
                        first_neighbours[:, 0],
                        first_neighbours[:, 1],
                        first_neighbours[:, 2],
                        1,
                    ]
                    == 0
                )[0]
                valid_neighbours = np.array(
                    [first_neighbours[index] for index in valid_index]
                )
                if valid_neighbours.shape[0] == 0:
                    continue

                second_neighbours = ant.find_second_neighbours()
                image_second_neigh = ant_colony[i].pheromone_release(second_neighbours)

                image_second_neigh_mean = image_second_neigh.mean()
                image_second_max = np.amax(image_second_neigh)
                image_second_min = np.amin(image_second_neigh)

                try:
                    n_offspring = min(
                        int(
                            26
                            * (image_second_neigh_mean - image_second_min)
                            / (image_second_max - image_second_min)
                        ),
                        int(valid_neighbours.shape[0]),
                    )
                except (ValueError, OverflowError):
                    n_offspring = int(valid_neighbours.shape[0])

                for neigh in valid_neighbours[:n_offspring]:
                    ant_colony.append(Ant(image, list(neigh)))
                pheromone_map[
                    valid_neighbours[:, 0],
                    valid_neighbours[:, 1],
                    valid_neighbours[:, 2],
                    1,
                ] = 1
                continue

        ant_number.append(len(ant_colony))
        n_iteration += 1

    logging.info(f"Image dimensions: {image.shape}\n")
    logging.info(f"Anthill coordinates: {args.anthill_coordinates}\n")
    logging.info(f"Energy death: {energy_death}\n")
    logging.info(f"Energy reproduction: {energy_reproduction}\n")
    logging.info(f"# iterations: {n_iteration}\n")
    logging.info(
        f"Elapsed time: {(time.perf_counter() - start_time_local) / 60:.3f} min\n"
    )

    statistics(ant_number, args, image, pheromone_map[..., 0])
    plot_display(aspect_ratio, image, pheromone_map[..., 0], anthill_position)
    plt.show()
