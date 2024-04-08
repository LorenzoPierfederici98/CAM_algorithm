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


"""Module implementing the region growing algorithm from
scikit-image which defines the ground truth for the CAM algorithm."""


import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, filters
from environment_constructor import ImageData


def ground_truth(image_matrix):
    """Defines the ground truth i.e the voxels part of the aerial trees using the flood
    method from skimage.segmentation, quoting 'starting at a specific seed_point,
    connected points equal or within tolerance of the seed value are found.'

    Args
    ----
    image_matrix : ndarray
        The image matrix.

    Returns
    -------
    segm_mask_array : ndarray
        The boolean matrix which defines the voxels part of the aerial trees.

    ground_truth_vox : ndarray
        The coordinates of the voxels part of the aerial trees.

    thresh_mean : float
        The mean of the threshold value given by threshold_otsu which
        distinguishes foreground and background, for all the z-slices.
    """

    segm_mask_arr = np.empty(image_matrix.shape, image_matrix.dtype)
    thresh_mean = 0

    for i in range(image_matrix.shape[2]):

        smooth = filters.butterworth(image_matrix[..., i])
        thresh_value = filters.threshold_otsu(smooth)
        thresh_mean += thresh_value
        fill = smooth > thresh_value
        mask = segmentation.disk_level_set(
            image_matrix[..., i].shape, radius=90
        ).astype(bool)
        fill[mask == 0] = 0
        segm_mask_arr[..., i] = fill

    thresh_mean = thresh_mean / image_matrix.shape[2]

    x_truth = np.nonzero(segm_mask_arr)[0]
    y_truth = np.nonzero(segm_mask_arr)[1]
    z_truth = np.nonzero(segm_mask_arr)[2]
    ground_truth_vox = np.stack((x_truth, y_truth, z_truth)).transpose()
    return segm_mask_arr, ground_truth_vox, thresh_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module implementing the region growing algorithm from scikit-image."
    )
    parser.add_argument(
        "seed_coordinates",
        help="The seed voxel position.",
        type=int,
        nargs=3,
        metavar="seed_coordinate",
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="The absolute path of the image directory.",
        metavar="file_path",
    )
    args = parser.parse_args()

    extrema = [168, 415, 284, 460, 257, 277]
    image, aspect_ratio = ImageData.image_from_file(args.file_path, extrema)

    segm_mask_array, ground_truth_voxels, _ = ground_truth(image)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_aspect(aspect_ratio["axial"])
    plot_0 = ax[0].imshow(image[:, :, args.seed_coordinates[2]], cmap="gray")
    ax[0].set_title("Original image")
    plot_1 = ax[1].imshow(segm_mask_array[:, :, args.seed_coordinates[2]])
    for plot, ax in zip([plot_0, plot_1], [ax[0], ax[1]]):
        plt.colorbar(plot, ax=ax)
    plt.tight_layout()
    plt.show()
