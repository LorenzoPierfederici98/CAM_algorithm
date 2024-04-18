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

# pylint: disable=C0103

"""Module building the ant class."""

import numpy as np


class Ant:


    """Class describing the ant worker which builds the pheromone map.

    Attributes
    ----------
    image_matrix : ndarray
        The original image matrix.

    voxel_coordinates : list[int]
        The coordinates of the ant current voxel.

    energy : float
        The ant energy.

    Methods
    -------
    update_energy : updates the ant energy.

    find_first_neighbours : finds the first-order neighbours
    of the current voxel.

    find_second_neighbours : finds the second-order neighbours of the current voxel.

    pheromone_release : releases pheromone in a voxel.

    evaluate_destination : computes the probability and chooses a voxel destination.
    """


    def __init__(self, image_matrix, voxel_coordinates):
        self.image_matrix = image_matrix
        self.voxel_coordinates = voxel_coordinates
        self._beta = 3.5
        self._delta = 0.2
        self._alpha = 0.2
        self._eta = 0.01
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


    @property
    def eta(self):
        """Defines the eta attribute as read-only."""
        return self._eta


    def update_energy(self, value):
        """Updates the ant energy."""
        self.energy = self.energy - self._alpha * (1.0 - value)


    def find_first_neighbours(self):
        """Finds the indexes of the first-order neighbours of
        the ant current voxel with no "Pac-Man" effect i.e
        considering the borders of the image matrix. The current
        voxel is not considered as a first neighbour.

        Returns
        -------
        neighbours : ndarray
            Array of the indexes of the first-order neighbours of the ant current voxel.
        """

        # Create the matrix whose rows contain the indexes of the
        # neighbouring voxels: [0, 0, 0] is the current voxel, [-1, 0, 0] is
        # the voxel on the left of the current voxel in the x-dimension, etc.
        neighbours = np.transpose(np.indices((3, 3, 3)) - 1).reshape(-1, 3)
        # Delete the coordinates of the current voxel i.e [0, 0, 0] and compute
        # the coordinates of the first order neighbours of the current voxel.
        neighbours = np.delete(neighbours, 13, axis=0) + self.voxel_coordinates
        if (
            self.voxel_coordinates[0] == 0
            or self.voxel_coordinates[1] == 0
            or self.voxel_coordinates[2] == 0
        ):
            neighbours = np.delete(
                neighbours, np.unique(np.where(neighbours < 0)[0]), axis=0
            )
        if self.voxel_coordinates[0] == self.image_matrix.shape[0] - 1:
            neighbours = np.delete(
                neighbours,
                np.unique(np.where(neighbours[:, 0] > self.voxel_coordinates[0])[0]),
                axis=0,
            )
        if self.voxel_coordinates[1] == self.image_matrix.shape[1] - 1:
            neighbours = np.delete(
                neighbours,
                np.unique(np.where(neighbours[:, 1] > self.voxel_coordinates[1])[0]),
                axis=0,
            )
        if self.voxel_coordinates[2] == self.image_matrix.shape[2] - 1:
            neighbours = np.delete(
                neighbours,
                np.unique(np.where(neighbours[:, 2] > self.voxel_coordinates[2])[0]),
                axis=0,
            )
        return neighbours


    def find_second_neighbours(self):
        """Finds the indexes of the second-order neighbours of the ant current
        voxel. The construction is similar to that one of the find_first_neighbours function.

        Returns
        -------
        neighbours : ndarray
            Array of the indexes of the second-order neighbours of the current voxel.
        """

        neighbours = np.transpose(np.indices((5, 5, 5)) - 2).reshape(-1, 3)
        neighbours = np.delete(neighbours, 62, axis=0) + self.voxel_coordinates
        if (
            (self.voxel_coordinates[0] in (0, 1, 2))
            or (self.voxel_coordinates[1] in (0, 1, 2))
            or (self.voxel_coordinates[2] in (0, 1, 2))
        ):
            neighbours = np.delete(
                neighbours, np.unique(np.where(neighbours < 0)[0]), axis=0
            )
        if self.voxel_coordinates[0] in (
            self.image_matrix.shape[0] - 1,
            self.image_matrix.shape[0] - 2,
        ):
            neighbours = np.delete(
                neighbours,
                np.unique(np.where(neighbours[:, 0] >= self.image_matrix.shape[0])[0]),
                axis=0,
            )
        if self.voxel_coordinates[1] in (
            self.image_matrix.shape[1] - 1,
            self.image_matrix.shape[1] - 2,
        ):
            neighbours = np.delete(
                neighbours,
                np.unique(np.where(neighbours[:, 1] >= self.image_matrix.shape[1])[0]),
                axis=0,
            )
        if self.voxel_coordinates[2] in (
            self.image_matrix.shape[2] - 1,
            self.image_matrix.shape[2] - 2,
        ):
            neighbours = np.delete(
                neighbours,
                np.unique(np.where(neighbours[:, 2] >= self.image_matrix.shape[2])[0]),
                axis=0,
            )
        return neighbours


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x - 517.))


    def pheromone_release(self, voxel_coordinates):
        """Computes the quantity of pheromone to be released into the voxel to
        build the pheromone map. The quantity of pheromone released corresponds to
        the intensity of the voxel of the image matrix plus a small offset
        which certifies that the voxel was visited.

        Args
        ----
        voxel_coordinates : list or ndarray
            The voxel cooridnates in which the pheromone is released.

        Returns
        -------
        pheromone_value : float or list[float]
            The pheromone value to be stored into the pheromone map voxel.
        """

        propor_factor = 10

        if isinstance(voxel_coordinates, list):
            pheromone_value = (
                propor_factor
                * self.sigmoid(self.image_matrix[
                    voxel_coordinates[0],
                    voxel_coordinates[1],
                    voxel_coordinates[2],
                ])
                + self._eta
            )
        elif isinstance(voxel_coordinates, np.ndarray):
            pheromone_value = (
                propor_factor
                * self.sigmoid(self.image_matrix[
                    voxel_coordinates[:, 0],
                    voxel_coordinates[:, 1],
                    voxel_coordinates[:, 2],
                ])
                + self._eta
            )
        return pheromone_value


    def evaluate_destination(self, first_neighbours, pheromone_map):
        """Computes the probability for the first-neighbouring voxels to be
        chosen as the next destination of the ant. The next voxel
        is chosen with a roulette wheel algorithm among those neighbours.

        Args
        ------
        first_neighbours : ndarray
            Array whose rows correspond to the indexes of the first-neighbouring voxels.

        pheromone_map : ndarray
            Four-dimensional matrix. The first 3 dimensions contain the image matrix,
            the fourth dimension is used to store value 0 or 1 whether the
            corresponding voxel is free or occupied, respectively.

        Returns
        -------
        valid_first_neigbours : list[int]
            Indexes of the chosen voxel-destination. Returns an empty list
            if there are no possible voxel destinations.
        """

        # The maximum number of visits per voxel depends on the
        # quantity of pheromone in that voxel relative to that one in
        # the whole pheromone map.
        if np.amax(pheromone_map[:, :, :, 0]) != 0 and self.pheromone_release(
            self.voxel_coordinates
        ) < np.amax(pheromone_map[:, :, :, 0]):
            max_visit_number = int(
                40
                + 80
                * (
                    self.pheromone_release(self.voxel_coordinates)
                    - np.amax(pheromone_map[:, :, :, 0])
                )
                / (
                    np.amin(pheromone_map[:, :, :, 0])
                    - np.amax(pheromone_map[:, :, :, 0])
                )
            )
        else:
            max_visit_number = 40
        # Select only voxels with a number of visits less than max_visit_number
        mask = np.where(
            pheromone_map[
                first_neighbours[:, 0],
                first_neighbours[:, 1],
                first_neighbours[:, 2],
                0,
            ]
            < max_visit_number * self.pheromone_release(first_neighbours)
        )[0]
        valid_first_neighbours = first_neighbours[mask]
        if valid_first_neighbours.shape[0] == 0:
            return []
        # Indexes of the voxels already occupied by an ant
        occupied_indexes = pheromone_map[
            valid_first_neighbours[:, 0],
            valid_first_neighbours[:, 1],
            valid_first_neighbours[:, 2],
            1,
        ].nonzero()[0]
        if occupied_indexes.shape[0] != 0:
            # Delete the occupied voxels from the array of valid
            # first neighbours
            valid_first_neighbours = np.delete(
                valid_first_neighbours, occupied_indexes, axis=0
            )
            if valid_first_neighbours.shape[0] == 0:
                return []
        W_PROBABILITY = (
            1.0
            + pheromone_map[
                valid_first_neighbours[:, 0],
                valid_first_neighbours[:, 1],
                valid_first_neighbours[:, 2],
                0,
            ]
            / (
                1.0
                + self.delta
                * pheromone_map[
                    valid_first_neighbours[:, 0],
                    valid_first_neighbours[:, 1],
                    valid_first_neighbours[:, 2],
                    0,
                ]
            )
        ) ** (self.beta)
        probability = W_PROBABILITY / W_PROBABILITY.sum()
        rand_num = np.random.uniform(0, np.max(probability))
        next_voxel_index = np.where(probability >= rand_num)[0][0]
        return list(valid_first_neighbours[next_voxel_index])
