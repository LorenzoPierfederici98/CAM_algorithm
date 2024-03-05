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

"""Module for the unit-testing of the ant class."""

import sys
import os
import unittest
import numpy as np

# This adds src to the list of directories the interpreter will search
# for the required module.
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from src.ant_constructor import Ant


class AntTesting(unittest.TestCase):
    """Class inheriting from unittest.TestCase for the ant class functionalities testing."""

    def test_private_attributes(self):
        """Tests that the private attributes are read-only."""
        ant = Ant([], [])

        def test_alpha(a):
            a.alpha = 0.0

        def test_beta(a):
            a.beta = 0.0

        def test_delta(a):
            a.delta = 0.0

        self.assertRaises(AttributeError, test_alpha, ant)
        self.assertRaises(AttributeError, test_beta, ant)
        self.assertRaises(AttributeError, test_delta, ant)

    def test_energy_update(self):
        """Checks the update of the ant energy."""
        ant = Ant([], [])
        ant_energy = ant.energy
        update_value = 10.0
        ant.update_energy(update_value)
        self.assertAlmostEqual(
            ant_energy - ant.alpha * (1.0 - update_value), ant.energy
        )

    def test_find_first_neighbours(self):
        """Checks the number of the first-order neighbours."""
        dimensions = [10, 10, 3]
        matrix = np.zeros(dimensions)
        voxel_coordinates = np.array(
            [
                [0, 0, 0],
                np.array(dimensions) - 1,
                [dimensions[0] - 1, dimensions[1] - 1, dimensions[2] - 2],
                [dimensions[0] - 2, dimensions[1] - 2, dimensions[2] - 2],
            ]
        )
        for i in range(len(voxel_coordinates[:, 0])):
            ant = Ant(matrix, voxel_coordinates[i])
            neighbour_number = [7, 7, 11, 26]
            self.assertEqual(neighbour_number[i], ant.find_first_neighbours().shape[0])

    def test_find_second_neighbours(self):
        """Checks the number of the second-order neighbours."""
        dimensions = [10, 10, 5]
        matrix = np.zeros(dimensions)
        voxel_coordinates = np.array(
            [
                [0, 0, 0],
                np.array(dimensions) - 1,
                [dimensions[0] - 1, dimensions[1] - 1, dimensions[2] - 2],
                [dimensions[0] - 3, dimensions[1] - 3, dimensions[2] - 3],
            ]
        )
        for i in range(len(voxel_coordinates[:, 0])):
            ant = Ant(matrix, voxel_coordinates[i])
            neighbour_number = [26, 26, 35, 124]
            self.assertEqual(neighbour_number[i], ant.find_second_neighbours().shape[0])

    def test_evaluate_destination(self):
        """Checks the evaluate_destination function which chooses the
        next voxel destination of the ant."""
        dimensions = [5, 5, 5]
        image_matrix = np.zeros(dimensions)
        image_matrix[2:4, 2:4, 2:4] = 5.0
        current_voxel = [2, 2, 2]
        ant = Ant(image_matrix, current_voxel)
        first_neighbours = ant.find_first_neighbours()
        # All the neighbouring voxels are occupied i.e voxel_dict key
        # has value True for all first-neighbours
        voxel_dict = {f"{elem}": True for elem in first_neighbours}
        self.assertEqual(
            [], ant.evaluate_destination(first_neighbours, image_matrix, voxel_dict)
        )
        # Now only the second half of first-neighbours is occupied
        voxel_dict.update(
            {
                f"{elem}": False
                for elem in first_neighbours[: int(first_neighbours.shape[0] / 2)]
            }
        )
        # Choose a random first-neighbour from those occupied 
        # for N_TEST iteration(s) and check that it isn't chosen 
        # as the next voxel destination
        N_TEST = 15
        for _ in range(N_TEST):
            j = np.random.randint(
                int(first_neighbours.shape[0] / 2), first_neighbours.shape[0]
            )
            self.assertFalse(
                (first_neighbours[j] ==
                ant.evaluate_destination(first_neighbours, image_matrix, voxel_dict)).all()
            )

if __name__ == "__main__":
    unittest.main()
