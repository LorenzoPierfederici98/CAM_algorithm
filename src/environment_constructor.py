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


class ImageData:


    """Class handling the image matrix and the pheromone map.

    Attributes
    ----------
    matrix_dimensions : list[int]
        The shape of the image matrix

    Methods
    -------
    create_cube : creates a cube as the image matrix.

    create_sphere_ellipsoid : creates a sphere or ellipsoid as the image matrix.

    create_donut : creates a donut as the image matrix.

    initialize_pheromone_map : initializes the pheromone map.

    image_from_file : classmethod, alternative constructor which provides an image
    from a dicom folder.
    """


    def __init__(self, matrix_dimensions):
        self.matrix_dimensions = matrix_dimensions


    def create_cube(self, center, length):
        """Creates a cube with a certain length from the center-coordinates
        given by the user.

        Args
        ------
        center : list[int]
            The coordinates of the center of the cube.

        length : int
            The length of the cube, which must be less than matrix dimensions.

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
        return image_matrix


    def create_sphere_ellipsoid(self, center, radius, semi_axes):
        """Creates a sphere or an ellipsoid with center and radius given by the user.
        If semi_axes is [1, 1, 1] a sphere is returned.

        Args
        ----
        center : list[int]
            The coordinates of the center.

        semi_axes : list[float]
            The semi-axes lengths.

        radius : float
            The radius of the figure.

        Returns
        -------
        image_matrix : ndarray
            The image matrix of the sphere/ellipsoid.
        """

        x = np.linspace(0, self.matrix_dimensions[0] - 1, self.matrix_dimensions[0])
        y = np.linspace(0, self.matrix_dimensions[1] - 1, self.matrix_dimensions[1])
        z = np.linspace(0, self.matrix_dimensions[2] - 1, self.matrix_dimensions[2])
        x, y, z = np.meshgrid(x, y, z)
        image_matrix = np.sqrt(
            (x - center[0]) ** 2 / semi_axes[0] ** 2
            + (y - center[1]) ** 2 / semi_axes[1] ** 2
            + (z - center[2]) ** 2 / semi_axes[2] ** 2
        )
        image_matrix = np.where(image_matrix <= radius, 5, 0)
        return image_matrix


    def create_donut(self, center, radius):
        """Creates a sphere with a concetric hole of radius radius/2.

        Args
        ----
        center : list[int]
            The coordinates of the center.

        radius : float
            The external radius.

        Returns
        -------
        image_matrix : ndarray
            The image matrix of the donut.
        """

        x = np.linspace(0, self.matrix_dimensions[0] - 1, self.matrix_dimensions[0])
        y = np.linspace(0, self.matrix_dimensions[1] - 1, self.matrix_dimensions[1])
        z = np.linspace(0, self.matrix_dimensions[2] - 1, self.matrix_dimensions[2])
        x, y, z = np.meshgrid(x, y, z)
        image_matrix = np.sqrt(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
        )
        image_matrix = np.where(
            (image_matrix <= radius) & (image_matrix >= radius // 2), 5, 0
        )
        return image_matrix


    def initialize_pheromone_map(self):
        """Initializes the pheromone map. The first 3 dimensions store
        the voxels values of the image matrix, the fourth dimension stores
        0 for every voxel i.e it isn't occupied by an ant.

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
    def image_from_file(cls, file_path, extrema):
        """Loads an image from a dicom folder whose path is
        given by the user. The output image is the result of
        the stacked slices of the dicom folder, sorted by their
        SliceLocation value; the voxels values are scaled back to the
        Hounsfield units.

        Args
        ----
        file_path : str
            The absolute path of the dicom folder.

        extrema : list[int]
            The values to crop the image matrix as
            image_matrix[extrema[0] : extrema[1], extrema[2] : extrema[3], extrema[4] : extrema[5]]

        Returns
        -------
        CT_array : ndarray
            The image matrix.

        aspect_ratio : dict
            Dictionary of values that preserve the
            aspect ratio of the different slices.
        """

        dicom_files = glob.glob(os.path.join(file_path, "*"))
        dicom_slices = [pydicom.read_file(fname) for fname in dicom_files]
        slices = [
            dcm_slice
            for dcm_slice in dicom_slices
            if hasattr(dcm_slice, "SliceLocation")
        ]
        slices.sort(key=lambda x: x.SliceLocation)
        shape = slices[0].Rows, slices[0].Columns, len(slices)
        CT_array = np.zeros(shape, dtype=slices[0].pixel_array.dtype)
        for i, dcm in enumerate(slices):
            CT_array[:, :, i] = dcm.pixel_array
        CT_array = np.flip(CT_array, axis=1)
        x, y, z = *slices[0].PixelSpacing, slices[0].SliceThickness
        aspect_ratio = {"axial": y / x, "sagittal": y / z, "coronal": x / z}
        # Crop and return to HU units
        CT_array = (
            CT_array[
                extrema[0] : extrema[1],
                extrema[2] : extrema[3],
                extrema[4] : extrema[5],
            ]
            - 1024.0
        )
        return CT_array, aspect_ratio
