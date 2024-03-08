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
from ant_constructor import Ant
from environment_constructor import ImageData

def update_pheromone_map(ant, pheromone_map, pheromone_values):
    [x, y, z] = ant.voxel_coordinates
    pheromone_map[x, y, z, 0] += ant.pheromone_release()
    pheromone_map[x, y, z, 1] = True
    pheromone_values = np.append(pheromone_values, pheromone_map[x, y, z, 0])

def find_next_voxel(ant, pheromone_map):
    [x, y, z] = ant.voxel_coordinates
    first_neighbours = ant.find_first_neighbours()
    next_voxel = ant.evaluate_destination(first_neighbours, pheromone_map)
    pheromone_map[x, y, z, 1] = False
    return next_voxel


matrix_dim = [512, 512, 512]
imagedata = ImageData(matrix_dim)
cube_length = 100
center_coordinates = [200, 200, 200]
cube_image = imagedata.create_cube(cube_length, center_coordinates)
pheromone_map = imagedata.initialize_pheromone_map(cube_image)

anthill_position = [200, 200, 200]
first_ant = Ant(cube_image, anthill_position)
ant_colony = [Ant(cube_image, list(elem)) for elem in first_ant.find_first_neighbours()]
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

    while len(ant_colony) != 0 and n_iteration <= 10:
        for _, ant in enumerate(ant_colony):
            update_pheromone_map(ant, pheromone_map, pheromone_values)
        pheromone_mean = (pheromone_values / len(ant_colony)).mean()
        for i, ant in enumerate(ant_colony):
            next_voxel = find_next_voxel(ant, pheromone_map)
            if len(next_voxel) == 0:
                del ant_colony[i]
                continue
            ant.voxel_coordinates = next_voxel
        for i, ant in enumerate(ant_colony):
            ant.update_energy(ant.pheromone_release() / pheromone_mean)
            if ant.energy < energy_death:
                del ant_colony[i]
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
                    is False
                )[0]
                valid_neighbours = first_neighbours[valid_index]
                if valid_neighbours.shape[0] == 0:
                    continue
                ant_colony.append(
                    [Ant(cube_image, list(elem)) for elem in valid_neighbours]
                )
                pheromone_map[valid_neighbours[:, 0], valid_neighbours[:, 1], valid_neighbours[:, 2], 1] = True
                continue
        ant_number.append(len(ant_colony))
        n_iteration += 1

    fig, ax = plt.subplots(2, 3)
    ax[0][0].plot(ant_number)
    ax[0][1].imshow(cube_image[:, :, 200], cmap="gray")
    ax[0][1].set_title('Original image, axial view')
    ax[1][0].imshow(pheromone_map[:, :, 200, 0], cmap="gray")
    ax[1][1].imshow(pheromone_map[:, 200, :, 0], cmap="gray")
    ax[1][2].imshow(pheromone_map[200, :, :, 0], cmap="gray")
    fig.delaxes(ax[0][2])
    plt.tight_layout()
    plt.show()
