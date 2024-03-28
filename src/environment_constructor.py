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

import os
import glob
import pydicom
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

    image_from_file : classmethod, alternative constructor which provides
    an image from a dicom folder.
    """

    def __init__(self, matrix_dimensions):
        self.matrix_dimensions = matrix_dimensions


    def create_cube(self, length, center):
        """Creates a cube with a certain length from the given center-coordinates.

        Args
        ------
        length : int
            The length of the cube, which must be less than matrix dimensions.

        center : list[int]
            The coordinates of the center of the cube.

        Returns
        -------
        image_matrix : ndarray
            The cube image matrix.
        """
        image_matrix = np.zeros(self.matrix_dimensions)
        image_matrix[
            center[0] - length // 2 : center[0] + length // 2,
            center[1] - length // 2 : center[1] + length // 2,
            center[2] - length // 2 : center[2] + length // 2,
        ] = 5.0
        image_matrix[
            center[0] - length // 4 : center[0] + length // 4,
            center[1] - length // 4 : center[1] + length // 4,
            center[2] - length // 4 : center[2] + length // 4,
        ] = 10.0
        return image_matrix


    def initialize_pheromone_map(self):
        """Initializes the pheromone map. The first 3 dimensions store
        the voxels values of the image matrix, the fourth dimension stores
        False for every voxel i.e it isn't occupied by an ant.

        Args
        ------
        image_matrix : ndarray
            The matrix of the image to be segmented.

        Returns
        -------
        pheromone_map : ndarray
            The initialized pheromone map.
        """

        pheromone_map = np.zeros(
            (
                self.matrix_dimensions[0],
                self.matrix_dimensions[1],
                self.matrix_dimensions[2],
                2,
        )
        )
        return pheromone_map

    @classmethod
    def image_from_file(cls, file_path):
        """Loads an image from a dicom folder whose path is
        given by the user. The output image is the result of
        the stacked slices of the dicom folder, sorted by their
        SliceLocation value.

        Args
        ----
        file_path : str
            The absolute path of the dicom folder.

        Returns
        -------
        CT_array : ndarray
            The image matrix.

        aspect_ratio : dict
            Dictionary of values that preserve the
            aspect ratio of the different slices.
        """

        dicom_files = glob.glob(os.path.join(file_path, "*"))
        dicom_slices = [ pydicom.read_file(fname) for fname in dicom_files]
        slices = [dcm_slice for dcm_slice in dicom_slices if hasattr(dcm_slice, 'SliceLocation')]
        slices.sort(key=lambda x: x.SliceLocation)
        shape = slices[0].Rows, slices[0].Columns, len(slices)
        CT_array = np.zeros(shape, dtype=slices[0].pixel_array.dtype)
        for i, dcm in enumerate(slices):
            CT_array[:, :, i] = dcm.pixel_array
        CT_array = np.flip(CT_array, axis=1)
        x, y, z = *slices[0].PixelSpacing, slices[0].SliceThickness
        aspect_ratio = {
            'axial': y/x,
            'sagittal': y/z,
            'coronal': x/z
        }
        # Only for brain images
        #CT_array = np.swapaxes(CT_array, 0, 2)
        #CT_array = np.rot90(CT_array[..., 117 - 10 : 117 + 10], axes=(1,0))
        # Return to HU units
        CT_array = CT_array[168 : 415, 284 : 460, 267 - 10 : 267 + 10] - 1024.
        #ax = plt.subplot(1, 1, 1)
        #ax.imshow(CT_array[:, :, 10], cmap="gray")
        #ax.set_aspect(aspect_ratio['axial'])
        return CT_array, aspect_ratio


if __name__ == "__main__":
    matrix_dim = [512, 512, 512]
    imagedata = ImageData(matrix_dim)
    CT_arr = ImageData.image_from_file("D:/train_data/Training/CASE01")[0]
    plt.show()
    plt.hist(np.ravel(CT_arr), bins=300)
    plt.show()

