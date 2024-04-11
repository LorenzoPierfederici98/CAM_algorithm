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


"""Module implementing the segmentation of the internal lung region
with the watershed algorithm. The code is a slight modification of
that one present in
https://www.kaggle.com/code/ankasor/improved-lung-segmentation-using-watershed
"""

import argparse
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import segmentation, filters, measure, morphology
from environment_constructor import ImageData


def generate_markers(image_matrix):
    """Generates the internal, external and watershed markers
    needed for the watershed algorithm.

    Args
    ----
    image_matrix : ndarray
        The image matrix from the DICOM directory.

    Returns
    -------
    marker_internal : ndarray
        Labeled array of the internal lung region.

    marker_watershed : ndarray
        Superposition of internal and external labels.
    """

    # Creation of the internal Marker
    marker_internal = image_matrix < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    marker_watershed = np.zeros(
        (image_matrix.shape[0], image_matrix.shape[1]), dtype="int32"
    )
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    return marker_internal, marker_watershed


def seperate_lungs(image_matrix):
    """Segments the lung area with the watershed algorithm, applied to
    the image filtered with a Sobel filter and with markers given by
    the generate_markers function. A black top hat operation is applied
    to include the voxels at the border.

    Args
    ----
    image_matrix : ndarray
        The image matrix from the DICOM directory.

    Returns
    -------
    segmented : ndarray
        The internal lung region.
    """

    # Creation of the markers
    marker_internal, marker_watershed = generate_markers(image_matrix)

    # Creation of the Sobel-Gradient
    sobel_gradient = filters.sobel(image_matrix, axis=[0, 1])
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    # Watershed algorithm
    watershed = segmentation.watershed(sobel_gradient, marker_watershed)

    # Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3, 3)).astype(bool)

    # Performing Black-Tophat Morphology for reinclusion
    blackhat_struct = morphology.disk(radius=24).astype(bool)
    # Perform the Black-Hat
    outline += morphology.black_tophat(outline, footprint=blackhat_struct)

    # Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    # Close holes in the lungfilter
    # fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = morphology.binary_closing(lungfilter, footprint=np.ones((12, 12)))

    # Apply the lungfilter (note the filtered areas being assigned -1024 HU)
    segmented = np.where(
        lungfilter == 1,
        image_matrix,
        -1024 * np.ones((image_matrix.shape[0], image_matrix.shape[1])),
    )
    return segmented


def image_segmenter(image_matrix):
    """Returns the segmented image from a DICOM folder.

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

    segmented = np.empty(image_matrix.shape, image_matrix.dtype)

    for i in range(image_matrix.shape[2]):
        segmented[..., i] = seperate_lungs(image_matrix[..., i])

    return segmented


def ground_truth(segmented_image):
    """Defines the ground truth i.e the voxels part of the aerial trees in
    the region segmented with the watershed algorithm.
    
    Args
    ----
    segmented_image : ndarray
        The image segmented with the watershed algorithm

    Returns
    -------
    ground_truth_vox : ndarray
        The coordinates of the voxels part of the aerial trees.

    thresh_mean : float
        The mean of the threshold value given by threshold_otsu which
        distinguishes foreground and background, for all the z-slices.
    """

    segm_mask_arr = np.empty(segmented_image.shape, segmented_image.dtype)
    thresh_mean = 0

    for i in range(segmented_image.shape[2]):
        thresh = filters.threshold_otsu(segmented_image[..., i])
        thresh_mean += thresh
        thresh_image = segmented_image[..., i] > thresh
        segm_mask_arr[..., i] = thresh_image

    thresh_mean /= segmented_image.shape[2]
    x_truth = np.nonzero(segm_mask_arr)[0]
    y_truth = np.nonzero(segm_mask_arr)[1]
    z_truth = np.nonzero(segm_mask_arr)[2]
    ground_truth_vox = np.stack((x_truth, y_truth, z_truth)).transpose()
    return ground_truth_vox, thresh_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module implementing the region growing algorithm from scikit-image."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="The absolute path of the image directory.",
        metavar="file_path",
    )
    args = parser.parse_args()

    extrema = [138, 475, 234, 600, 257, 277]
    image, aspect_ratio = ImageData.image_from_file(args.file_path, extrema)

    segmented_im = image_segmenter(image)
    _, thresh = ground_truth(segmented_im)
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(image[..., 10], cmap="gray")
    ax[1].imshow(segmented_im[..., 10], cmap="gray")
    plt.show()
