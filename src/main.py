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

import numpy as np
import matplotlib.pyplot as plt
import time
from ant_constructor import Ant
from environment_constructor import ImageData

def update_pheromone_map(ant_worker, pheromone_matrix, pheromone):
    [x, y, z] = ant_worker.voxel_coordinates
    pheromone_matrix[x, y, z, 0] += pheromone
    pheromone_matrix[x, y, z, 1] = True

def find_next_voxel(ant_worker, pheromone_matrix):
    [x, y, z] = ant_worker.voxel_coordinates
    first_neigh = ant_worker.find_first_neighbours()
    next_vox = ant_worker.evaluate_destination(first_neigh, pheromone_matrix)
    pheromone_matrix[x, y, z, 1] = False
    if len(next_vox) != 0:
        pheromone_matrix[next_vox[0], next_vox[1], next_vox[2], 1] = True
    return next_vox

def plot_display(
    ants_number,
    image_matrix,
    visited_voxels_dict,
    pheromone_matrix,
    anthill_coordinates,
):
    _, ax = plt.subplots(2, 3)
    ax[0][0].plot(ants_number)
    ax[0][0].set_title("Number of ants")
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

matrix_dim = [80, 80, 80]
imagedata = ImageData(matrix_dim)
cube_length = 40
center_coordinates = [40, 40, 40]
cube_image = imagedata.create_cube(cube_length, center_coordinates)
non_zero_image_voxels = np.transpose(np.array(np.nonzero(cube_image))).reshape(-1, 3)
print(f"Non zero image voxels: {non_zero_image_voxels.shape[0]}")
pheromone_map = imagedata.initialize_pheromone_map()

anthill_position = [20, 20, 20]
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
    while len(ant_colony) != 0 and n_iteration <= 100:
        print(f'Iteration: {n_iteration}\n')
        released_pheromone = np.zeros(len(ant_colony))
        for i, ant in enumerate(ant_colony):
            released_pheromone[i] = ant.pheromone_release()
            update_pheromone_map(ant, pheromone_map, released_pheromone[i])
            pheromone_values = np.append(pheromone_values, released_pheromone[i])
        pheromone_mean = pheromone_values.mean()
        for i, ant in enumerate(ant_colony):
            time_0 = time.time()
            next_voxel = find_next_voxel(ant, pheromone_map)
            ant.update_energy(released_pheromone[i] / pheromone_mean)
            if len(next_voxel) == 0:
                del ant_colony[i]
                continue
            ant.voxel_coordinates = next_voxel
        for i, ant in enumerate(ant_colony):
            if ant.energy < energy_death:
                del ant_colony[i]
                if len(ant_colony) == 0:
                    print('No ants left.\n')
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
                    == False
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
                ] = True
                continue
        ant_number.append(len(ant_colony))
        print(f'Number of ants: {len(ant_colony)}\n')
        n_iteration += 1

    non_zero_voxels = np.unique(
        np.transpose(np.array(np.nonzero(pheromone_map[:, :, :, 0]))).reshape(-1, 3),
        axis=0,
    )
    print(f"Visited voxels: {non_zero_voxels.shape[0]}\n")
    print(
        f"Visited voxels percentage: {100 * non_zero_voxels.shape[0] / non_zero_image_voxels.shape[0] :.3f} %\n"
    )
    visited_voxels = {
        f"{list(elem)}": int(value)
        for elem, value in zip(
            non_zero_voxels,
            (
                pheromone_map[
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
        ant_number, cube_image, visited_voxels, pheromone_map, anthill_position
    )
    plt.show()
