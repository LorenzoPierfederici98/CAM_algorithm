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

"""Module implementing the CAM algorithm."""

import time
import multiprocessing
from multiprocessing.sharedctypes import RawArray
import numpy as np
import matplotlib.pyplot as plt
from ant_constructor import Ant
from environment_constructor import ImageData


def update_pheromone_map(ant_worker):
    """Updates the pheromone map and signals to the colony that the ant voxel as occupied.

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
    pheromone = ant_worker.pheromone_release()
    [x, y, z] = ant_worker.voxel_coordinates
    arr[x, y, z, 0] += pheromone
    arr[x, y, z, 1] = 1
    return pheromone


def find_next_voxel(ant_worker):
    """Finds the next destination of the ant and signals to the colonty that the ant voxel is free.

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
    ants_number,
    image_matrix,
    visited_voxs_dict,
    pheromone_matrix,
    anthill_coordinates,
):
    """Displays the plots of the number of ants per cycle,
    the original image matrix, the bar plot of the visited voxels
    and the pheromone map built by the ants.

    Args
    ----
    ants_number : list[int]
        The list of the nu,ber of ants per cycle.

    image_matrix : ndarray
        The matrix of the image to be segmented.

    visited_voxs_dict : dict
        The dictionary containing all the visited voxels.

    pheromone_matrix : ndarray
        The matrix of the pheromone map built by the ants.

    anthill_coordinates : list[int]
        The coordinates of the voxel chosen as the anthill position,
        used to display different slices of the pheromone map.
    """

    _, ax = plt.subplots(2, 3)
    ax[0][0].plot(ants_number)
    ax[0][0].set_title("Number of ants per cycle")
    plot_1 = ax[0][1].imshow(image_matrix[:, :, anthill_coordinates[2]], cmap="gray")
    ax[0][2].bar(list(visited_voxs_dict.keys()), visited_voxs_dict.values())
    ax[0][2].set_title("Bar plot of visited voxels")
    ax[0][2].set_xticks([])
    ax[0][1].set_title("Original image, axial view")
    plot_2 = ax[1][0].imshow(
        pheromone_matrix[:, :, anthill_coordinates[2], 0], cmap="gray"
    )
    ax[1][0].set_title("Pheromone map, axial view")
    plot_3 = ax[1][1].imshow(
        pheromone_matrix[:, anthill_coordinates[1], :, 0], cmap="gray"
    )
    ax[1][1].set_title("Pheromone map, coronal view")
    plot_4 = ax[1][2].imshow(
        pheromone_matrix[anthill_coordinates[0], :, :, 0], cmap="gray"
    )
    ax[1][2].set_title("Pheromone map, sagittal view")
    for plot, ax in zip([plot_1, plot_2, plot_3, plot_4], [ax[0][1], ax[1][0], ax[1][1], ax[1][2]]):
        plt.colorbar(plot, ax=ax)
    plt.tight_layout()


def pool_initializer(pheromone_matrix, pheromone_matrix_shape):
    """Function to be used as the Pool initializer to share the
    pheromone map between the processes instantiated by Pool.map.
    Sets the pheromone map and its shape as global variables.

    Args
    ----
    Pheromone_Map : ndarray
        The RawArray from multiprocessing used to share the pheromone map.

    Pheromone_Map_shape : tuple[int]
        The shape of the pheromone map.
    """
    global np_x
    np_x = pheromone_matrix
    global np_x_shape
    np_x_shape = pheromone_matrix_shape


def statistics(image_matrix, pheromone_matrix):
    image_voxels = np.transpose(np.array(np.nonzero(image_matrix))).reshape(-1, 3)
    visited_voxs = np.unique(
        np.transpose(np.array(np.nonzero(pheromone_matrix[:, :, :, 0]))).reshape(-1, 3),
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
            (
                pheromone_map[
                    visited_voxs[:, 0],
                    visited_voxs[:, 1],
                    visited_voxs[:, 2],
                    0,
                ]
            )
            / (
                image_matrix[visited_voxs[:, 0], visited_voxs[:, 1], visited_voxs[:, 2]]
                + 0.01
            ),
        )
    }
    common_dict = {}
    common_keys = set(image_voxels_dict).intersection(visited_voxels_dict)
    for key in common_keys:
        common_dict[key] = visited_voxels_dict[key]
    print(f"Image voxels: {image_voxels.shape[0]}\n")
    print(
        f"Visited voxels: {visited_voxs.shape[0]}\nOf which belonging to the image: {len(common_dict)} ({(100 * len(common_dict) / image_voxels.shape[0]):.1f}%)\n"
    )
    return visited_voxels_dict


matrix_dim = [80, 80, 80]
imagedata = ImageData(matrix_dim)
cube_length = 40
center_coordinates = [40, 40, 40]
cube_image = imagedata.create_cube(cube_length, center_coordinates)

pheromone_map_init = imagedata.initialize_pheromone_map()
np_x_shape = pheromone_map_init.shape
pheromone_shared = RawArray("i", np.array(pheromone_map_init.shape).prod().item())
np_x = pheromone_shared
# RawArray used as buffer to share the pheromone_map as np array
pheromone_map = np.frombuffer(pheromone_shared, dtype=np.float32).reshape(
    pheromone_map_init.shape
)
np.copyto(pheromone_map, pheromone_map_init)

anthill_position = [30, 30, 30]
first_ant = Ant(cube_image, anthill_position)
first_ant_neighbours = first_ant.find_first_neighbours()
ant_colony = [Ant(cube_image, list(elem)) for elem in first_ant_neighbours]
ant_colony = (
    ant_colony[: len(ant_colony) // 2]
    + [first_ant]
    + ant_colony[len(ant_colony) // 2 :]
)

n_iteration = 0
energy_death = 1.0
energy_reproduction = 1.3
pheromone_values = np.array([])
ant_number = []
pheromone_mean_sum = 0
colony_length = 0
if __name__ == "__main__":

    start_time_local = time.perf_counter()
    while len(ant_colony) != 0 and n_iteration <= 10:
        print(f"Iter:{n_iteration}\t#ants:{len(ant_colony)}\n")
        chunksize = max(2, len(ant_colony) // 4)
        with multiprocessing.Pool(
            processes=4,
            initializer=pool_initializer,
            initargs=(pheromone_shared, pheromone_map_init.shape),
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
                for neigh in valid_neighbours:
                    ant_colony.append(Ant(cube_image, list(neigh)))
                pheromone_map[
                    valid_neighbours[:, 0],
                    valid_neighbours[:, 1],
                    valid_neighbours[:, 2],
                    1,
                ] = 1
                continue

        ant_number.append(len(ant_colony))
        n_iteration += 1

    print(f"Elapsed time: {(time.perf_counter() - start_time_local) / 60:.3f} min\n")
    visited_voxels = statistics(cube_image, pheromone_map)
    plot_display(
        ant_number, cube_image, visited_voxels, pheromone_map, anthill_position
    )
    plt.show()
