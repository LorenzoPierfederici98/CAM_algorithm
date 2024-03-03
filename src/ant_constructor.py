#     Exam project for the CMEPDA course.
#     Copyright (C) 2024  Lorenzo Pierfederici  l.pierfederici@studenti.unipi.it

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

"""Module building the ant class."""

import numpy as np


class Ant:
    """Class describing the ant worker who builds the pheromone map.
    
    Attributes
    ----------
    pheromone_map : ndarray
        The current state of the pheromone map built by all the ants.

    voxel_coordinates : list of int
        The coordinates of the ant current voxel.

    energy : float
        The ant energy

    Methods
    -------
    update_energy : updates the ant energy.

    find_first_neighbours : finds the first-order neighbours of the current voxel.

    find_second_neighbours : finds the first-order neighbours of the current voxel.

    compute_probability : computes the probability to choose a voxel destination.

    pheromone_release : releases pheromone in a voxel.
    """

    def __init__(self, matrix, voxel_coordinates):
        self.pheromone_map = matrix
        self.voxel_coordinates = voxel_coordinates
        if len(voxel_coordinates) != 0:
            self.x = voxel_coordinates[0]
            self.y = voxel_coordinates[1]
            self.z = voxel_coordinates[2]
        self._beta = 3.5
        self._delta = 0.2
        self._alpha = 0.2
        self.energy = 1.0 + self._alpha

    @property
    def beta(self):
        """Defines the beta attribute as read-only."""
        return self._beta

    @property
    def delta(self):
        """Defines the delta attribute as read-only."""
        return self._delta

    @property
    def alpha(self):
        """Defines the alpha attribute as read-only."""
        return self._alpha

    def update_energy(self, value):
        """Updates the ant energy."""
        self.energy = self.energy - self._alpha * (1.0 - value)

    def find_first_neighbours(self):
        """Finds the indexes of the first-order neighbours of the ant current voxel with no "Pac-Man" effect i.e considering the borders of the image matrix.

        Returns
        -------
        neighbours : list of int
            List of the indexes of the first-order neighbours of the ant current voxel.
        """

        # Create the matrix whose rows contain the indexes of the
        # neighbouring voxels: [0, 0, 0] is the current voxel, [-1, 0, 0] is
        # the voxel on the left of the current voxel in the x-dimension, etc.
        neighbours = np.transpose(np.indices((3, 3, 3)) - 1).reshape(-1, 3)
        # Delete the coordinates of the current voxel i.e [0, 0, 0] and compute
        # the coordinates of the first order neighbours of the current voxel.
        neighbours = np.delete(neighbours, 13, axis=0) + self.voxel_coordinates
        if self.x == 0 or self.y == 0 or self.z == 0:
            neighbours = np.delete(
                neighbours, np.unique(np.where(neighbours < 0)[0]), axis=0
            )
        if self.x == self.pheromone_map.shape[0] - 1:
            neighbours = np.delete(
                neighbours, np.unique(np.where(neighbours[:, 0] > self.x)[0]), axis=0
            )
        if self.y == self.pheromone_map.shape[1] - 1:
            neighbours = np.delete(
                neighbours, np.unique(np.where(neighbours[:, 1] > self.y)[0]), axis=0
            )
        if self.z == self.pheromone_map.shape[2] - 1:
            neighbours = np.delete(
                neighbours, np.unique(np.where(neighbours[:, 2] > self.z)[0]), axis=0
            )
        return neighbours

    def find_second_neighbours(self):
        """Finds the indexes of the second-order neighbours of the ant current voxel. The construction is similar to that one of the find_first_neighbours function.

        Returns
        -------
        neighbours : list of int
            List of the indexes of the second-order neighbours of the current voxel.
        """

        neighbours = np.transpose(np.indices((5, 5, 5)) - 2).reshape(-1, 3)
        neighbours = np.delete(neighbours, 62, axis=0) + self.voxel_coordinates
        if (self.x in (0, 1, 2)) or (self.y in (0, 1, 2)) or (self.z in (0, 1, 2)):
            neighbours = np.delete(
                neighbours, np.unique(np.where(neighbours < 0)[0]), axis=0
            )
        if self.x in (self.pheromone_map.shape[0] - 1, self.pheromone_map.shape[0] - 2):
            neighbours = np.delete(
                neighbours,
                np.unique(np.where(neighbours[:, 0] >= self.pheromone_map.shape[0])[0]),
                axis=0,
            )
        if self.y in (self.pheromone_map.shape[1] - 1, self.pheromone_map.shape[1] - 2):
            neighbours = np.delete(
                neighbours,
                np.unique(np.where(neighbours[:, 1] >= self.pheromone_map.shape[1])[0]),
                axis=0,
            )
        if self.z in (self.pheromone_map.shape[2] - 1, self.pheromone_map.shape[2] - 2):
            neighbours = np.delete(
                neighbours,
                np.unique(np.where(neighbours[:, 2] >= self.pheromone_map.shape[2])[0]),
                axis=0,
            )
        return neighbours

    def compute_probability(self):
        """Computes the probability for a voxel to be chosen as the next destination of the ant."""
        return


if __name__ == "__main__":
    A = np.zeros((5, 5, 5))
    ant = Ant(A, [2, 2, 2])
    print(ant.find_second_neighbours().shape)
    ant.alpha = 3
