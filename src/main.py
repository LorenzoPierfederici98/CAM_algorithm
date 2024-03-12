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
    visited_voxels_dict,
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

    visited_voxels_dict : dict
        The dictionary containing all the visited voxels.

    pheromone_matrix : ndarray
        The matrix of the pheromone map built by the ants.

    anthill_coordinates : list[int]
        The coordinates of the voxel chosen as the anthill position.
    """

    _, ax = plt.subplots(2, 3)
    ax[0][0].plot(ants_number)
    ax[0][0].set_title("Number of ants per cycle")
    ax[0][1].imshow(image_matrix[:, :, 40], cmap="gray")
    ax[0][2].bar(list(visited_voxels_dict.keys()), visited_voxels_dict.values())
    ax[0][2].set_title("Bar plot of visited voxels")
    ax[0][2].set_xticks([])
    ax[0][1].set_title("Original image, axial view")
    ax[1][0].imshow(pheromone_matrix[:, :, anthill_coordinates[2], 0], cmap="gray")
    ax[1][0].set_title("Pheromone map, axial view")
    ax[1][1].imshow(pheromone_matrix[:, anthill_coordinates[1], :, 0], cmap="gray")
    ax[1][1].set_title("Pheromone map, coronal view")
    ax[1][2].imshow(pheromone_matrix[anthill_coordinates[0], :, :, 0], cmap="gray")
    ax[1][2].set_title("Pheromone map, sagittal view")
    plt.tight_layout()


def pool_initializer(Pheromone_Map, Pheromone_Map_shape):
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
    np_x = Pheromone_Map
    global np_x_shape
    np_x_shape = Pheromone_Map_shape


matrix_dim = [80, 80, 80]
imagedata = ImageData(matrix_dim)
cube_length = 40
center_coordinates = [40, 40, 40]
cube_image = imagedata.create_cube(cube_length, center_coordinates)
non_zero_image_voxels = np.transpose(np.array(np.nonzero(cube_image))).reshape(-1, 3)

pheromone_map = imagedata.initialize_pheromone_map()
np_x_shape = pheromone_map.shape
pheromone_shared = RawArray("i", np.array(pheromone_map.shape).prod().item())
np_x = pheromone_shared
# RawArray used as buffer to share the pheromone_shared_main as np array
pheromone_shared_main = np.frombuffer(pheromone_shared, dtype=np.float32).reshape(
    pheromone_map.shape
)
np.copyto(pheromone_shared_main, pheromone_map)

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

if __name__ == "__main__":
    start_time_local = time.perf_counter()
    while len(ant_colony) != 0 and n_iteration <= 100:
        # print(f'Iter: {n_iteration}\t#ants: {len(ant_colony)}\n')
        chunksize = max(2, len(ant_colony) // 6)
        with multiprocessing.Pool(
            processes=6,
            initializer=pool_initializer,
            initargs=(pheromone_shared, pheromone_map.shape),
        ) as pool:
            released_pheromone = pool.map(
                update_pheromone_map, ant_colony, chunksize=chunksize
            )
        if None in released_pheromone:
            print(n_iteration, released_pheromone)
        for i, ant in enumerate(ant_colony):
            pheromone_values = np.append(pheromone_values, released_pheromone[i])
        pheromone_mean = pheromone_values.mean()
        with multiprocessing.Pool(
            processes=6,
            initializer=pool_initializer,
            initargs=(pheromone_shared, pheromone_map.shape),
        ) as pool:
            next_voxel = pool.map(find_next_voxel, ant_colony, chunksize=chunksize)
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
                    pheromone_shared_main[
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
                pheromone_shared_main[
                    valid_neighbours[:, 0],
                    valid_neighbours[:, 1],
                    valid_neighbours[:, 2],
                    1,
                ] = 1
                continue
        ant_number.append(len(ant_colony))
        n_iteration += 1
    print(f"Elapsed time: {(time.perf_counter() - start_time_local):.3f}\n")

    non_zero_voxels = np.unique(
        np.transpose(np.array(np.nonzero(pheromone_shared_main[:, :, :, 0]))).reshape(
            -1, 3
        ),
        axis=0,
    )
    print(f"Image voxels: {non_zero_image_voxels.shape[0]}\n")
    print(f"Visited voxels: {non_zero_voxels.shape[0]}\n")
    print(
        f"Visited voxels percentage: {100 * non_zero_voxels.shape[0] / non_zero_image_voxels.shape[0] :.3f} %\n"
    )
    visited_voxels = {
        f"{list(elem)}": int(value)
        for elem, value in zip(
            non_zero_voxels,
            (
                pheromone_shared_main[
                    non_zero_voxels[:, 0],
                    non_zero_voxels[:, 1],
                    non_zero_voxels[:, 2],
                    0,
                ]
            )
            / (
                cube_image[
                    non_zero_voxels[:, 0], non_zero_voxels[:, 1], non_zero_voxels[:, 2]
                ]
                + 0.01
            ),
        )
    }
    plot_display(
        ant_number, cube_image, visited_voxels, pheromone_shared_main, anthill_position
    )
    plt.show()
