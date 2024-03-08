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

"""Module building the environment of the ant colony i.e 
creates the image matrix and initializes the pheromone map."""

import traceback
import numpy as np
import matplotlib.pyplot as plt

class ImageData:
    """Class handling the image matrix and the pheromone map.
    
    Attributes
    ----------
    matrix_dimensions : list of int
        The shape of the image matrix
        
    Methods
    -------
    create_cube : creates a cube as an image matrix.

    initialize_pheromone_map : initializes the pheromone map.
    """

    def __init__(self, matrix_dimensions):
        self.matrix_dimensions = matrix_dimensions

    def create_cube(self, length, center):
        """Creates a cube with a certain length in a random position.

        Inputs
        ------
        length : int
            The length of the cube, which must be less than matrix dimensions.

        center : list of int
            The coordinates of the center of the cube.

        Returns
        -------
        image_matrix : ndarray
            The cube image matrix.
        """

        image_matrix = np.zeros(self.matrix_dimensions)
        image_matrix[
            center[0] - length : center[0] + length,
            center[1] - length : center[1] + length,
            center[2] - length : center[2] + length,
        ] = 10.0
        return image_matrix

    def initialize_pheromone_map(self, image_matrix):
        """Initializes the pheromone map. The first 3 dimensions store
        the voxels values of the image matrix, the fourth dimension stores
        False for every voxel i.e it isn't occupied by an ant.

        Inputs
        ------
        image_matrix : ndarray
            The matrix of the image to be segmented.

        Returns
        -------
        pheromone_map : ndarray
            The initialized pheromone map.
        """

        pheromone_map = np.zeros(
            [
                self.matrix_dimensions[0],
                self.matrix_dimensions[1],
                self.matrix_dimensions[2],
                2,
            ]
        )
        pheromone_map[:, :, :, 1] = False
        return pheromone_map
    
    def image_display(self, ax, image_matrix, slice_value, image_title, view='axial', cmap='viridis'):
        """Displays a certain view of the image matrix"""
        slice_string = ''
        try:
            if view == 'axial':
                ax.imshow(image_matrix[:, :, slice_value], cmap=cmap)
                slice_string = 'z'
            elif view == 'coronal':
                ax.imshow(image_matrix[:, slice_value, :], cmap=cmap)
                slice_string = 'y'
            elif view == 'sagittal':
                ax.imshow(image_matrix[slice_value, :, :], cmap=cmap)
                slice_string = 'z'
            else:
                raise ValueError
        except IndexError as e:
            print('Invalid value of slice\n')
            traceback.print_exception(e.__class__, e, e.__traceback__)
        except ValueError as e:
            print('Invalid input, view must be either \'axial\', \'coronal\' or \'sagittal\'.\n')
            traceback.print_exception(e.__class__, e, e.__traceback__)

        ax.set_title(f'{image_title}: {view} view {slice_string} = {slice_value}')
        #plt.colorbar()

if __name__ == "__main__":
    matrix_dim = [512, 512, 512]
    imagedata = ImageData(matrix_dim)
