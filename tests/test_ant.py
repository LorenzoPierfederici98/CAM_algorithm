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

import numpy as np
import sys
import os
import unittest

# This adds src to the list of directories the interpreter will search
# for the required module.
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from src.ant_constructor import Ant

class AntTesting(unittest.TestCase):
    """Class inheriting from unittest.TestCase for the ant class functionalities testing.
    """
    def test_energy_update(self):
        """Checks the update of the ant energy."""
        matrix = []
        voxel_coordinates = []
        ant = Ant(matrix, voxel_coordinates)
        ant_energy = ant.energy
        update_value = 10.
        ant.update_energy(update_value)
        self.assertAlmostEqual(ant_energy - ant.alpha * (1. - update_value), ant.energy)

    def test_find_first_neighbours(self):
        """Checks the number of the first-order neighbours."""
        dimensions = [10, 10, 3]
        matrix = np.zeros(dimensions)
        voxel_coordinates = np.array([[0, 0, 0], np.array(dimensions) - 1, [dimensions[0] - 1, dimensions[1] - 1, dimensions[2] - 2], [dimensions[0] - 2, dimensions[1] - 2, dimensions[2] - 2]])
        for i in range(len(voxel_coordinates[:, 0])):
            ant = Ant(matrix, voxel_coordinates[i])
            neighbour_number = [7, 7, 11, 26]
            self.assertEqual(neighbour_number[i], ant.find_first_neighbours().shape[0])

    def test_find_second_neighbours(self):
        """Checks the number of the second-order neighbours."""
        dimensions = [10, 10, 5]
        matrix = np.zeros(dimensions)
        voxel_coordinates = np.array([[0, 0, 0], np.array(dimensions) - 1, [dimensions[0] - 1, dimensions[1] - 1, dimensions[2] - 2], [dimensions[0] - 3, dimensions[1] - 3, dimensions[2] - 3]])
        for i in range(len(voxel_coordinates[:, 0])):
            ant = Ant(matrix, voxel_coordinates[i])
            neighbour_number = [26, 26, 35, 124]
            self.assertEqual(neighbour_number[i], ant.find_second_neighbours().shape[0])

if __name__ == '__main__':
    unittest.main()
