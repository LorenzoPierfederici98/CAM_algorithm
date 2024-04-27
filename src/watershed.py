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

This module also implements the region growing flood algorithm from
scikit image used as a benchmark for the CAM algorithm.
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
    border = segmentation.find_boundaries(lungfilter)
    # Apply the lungfilter (note the filtered areas being assigned -1024 HU)
    segmented = np.where(
        (lungfilter == 1) & (border == 0),
        image_matrix,
        -1024.0 * np.ones((image_matrix.shape[0], image_matrix.shape[1])),
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
    cropped : ndarray
        The segmented and cropped lung ROI.
    """

    segmented = np.empty(image_matrix.shape, image_matrix.dtype)
    xl = np.empty(image_matrix.shape[2], dtype=int)
    xr = np.empty(image_matrix.shape[2], dtype=int)
    yl = np.empty(image_matrix.shape[2], dtype=int)
    yr = np.empty(image_matrix.shape[2], dtype=int)

    for i in range(image_matrix.shape[2]):
        segmented[..., i] = seperate_lungs(image_matrix[..., i])
        x, y = np.nonzero(segmented[..., i] + 1024.0)
        xl[i], xr[i] = x.min(), x.max()
        yl[i], yr[i] = y.min(), y.max()

    x_left, x_right = xl.min(), xr.max()
    y_left, y_right = yl.min(), yr.max()
    cropped = segmented[x_left : x_right + 1, y_left : y_right + 1, :]
    return cropped


def ground_truth(segmented_image):
    """Defines the ground truth i.e the voxels part of the aerial trees in
    the region segmented with the watershed algorithm.

    Args
    ----
    segmented_image : ndarray
        The image segmented with the watershed algorithm.

    Returns
    -------
    segm_mask_arr : ndarray
        The ground truth image, containing the voxels classified
        as aerial tree.

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
        thresh_image = segmented_image[..., i] > thresh
        thresh_mean += thresh
        segm_mask_arr[..., i] = thresh_image

    thresh_mean /= segmented_image.shape[2]
    x_truth = np.nonzero(segm_mask_arr)[0]
    y_truth = np.nonzero(segm_mask_arr)[1]
    z_truth = np.nonzero(segm_mask_arr)[2]
    ground_truth_vox = np.stack((x_truth, y_truth, z_truth)).transpose()
    return segm_mask_arr, ground_truth_vox, thresh_mean


def region_growing(seed, segmented_image):
    """Applies the region growing flood segmentation algorithm
    from skimage.segmentation, to be compared with the CAM
    algorithm. The seed corresponds to the anthill voxel coordinates;
    it has to be a voxel with high intensity (100-200 HU).

    Args
    ----
    seed : list[int]
        The anthill voxel coordinates from which the
        segmentation starts.

    segmented_image : ndarray
        The image segmented with the watershed algorithm.

    Returns
    -------
    flood : ndarray
        The image-result of the region growing segmentation.

    flood_voxels : ndarray
        The voxels segmented with the region growing flood
        algorithm.
    """

    seed = tuple(seed)
    flood = segmentation.flood(
        segmented_image,
        seed_point=seed,
        footprint=np.ones((21, 21, 21)),
        tolerance=400,
    )
    x_flood = np.nonzero(flood)[0]
    y_flood = np.nonzero(flood)[1]
    z_flood = np.nonzero(flood)[2]
    flood_voxels = np.stack((x_flood, y_flood, z_flood)).transpose()
    return flood, flood_voxels


def plot_display(
    image_matrix, a_ratio, segmented_image, ground_truth_image_, seed
):
    """Displays the plots of the original image matrix, the ROI segmtented
    and cropped with the watershed algorithm, the image of the ground truth
    obtained with Otsu thresholding and the result of the region growing flood
    segmentation.

    Args
    ----
    image_matrix : ndarray
        The image matrix.

    a_ratio : float
        Value that preserves the aspect ratio of the axial slices.

    segmented_image : ndarray
        The image segmented with the watershed algorithm.

    ground_truth_image : ndarray
        The ground truth image defined with Otsu thresholding.

    seed : list[int] or None
        The seed from which the flood region growing algorithm starts
        the segmentation. If the user doesn't specify this argument it
        defaults to None, that happens the first time the user runs this
        module in order to assess th anthill voxel position on the segmented
        and cropped lung ROI.
    """

    cmap = "gray"
    axial_slice = image_matrix.shape[2] // 2
    if seed is not None:
        region_growing_image_, _ = region_growing(seed, segmented_image)
        _, ax = plt.subplots(2, 2, figsize=(8, 7))
        plot_1 = ax[0][0].imshow(image_matrix[..., axial_slice], cmap=cmap)
        ax[0][0].set_aspect(a_ratio)
        ax[0][0].set_title("Original image, axial view")
        plot_2 = ax[0][1].imshow(segmented_image[..., axial_slice], cmap=cmap)
        ax[0][1].set_title("Segmented and cropped ROI with watershed")
        plot_3 = ax[1][0].imshow(ground_truth_image_[..., axial_slice], cmap=cmap)
        ax[1][0].set_title("Ground truth, defined with thresholding")
        plot_4 = ax[1][1].imshow(region_growing_image_[..., axial_slice], cmap=cmap)
        ax[1][1].set_title("Segmentation with flood region growing")
        for plot, ax in zip(
            [plot_1, plot_2, plot_3, plot_4], [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
        ):
            plt.colorbar(plot, ax=ax)
    else:
        _, ax = plt.subplots(1, 3, figsize=(15, 6))
        plot_1 = ax[0].imshow(image_matrix[..., axial_slice], cmap=cmap)
        ax[0].set_aspect(a_ratio)
        plot_2 = ax[1].imshow(segmented_image[..., axial_slice], cmap=cmap)
        ax[1].set_title("Segmented and cropped ROI with watershed")
        plot_3 = ax[2].imshow(ground_truth_image_[..., axial_slice], cmap=cmap)
        ax[2].set_title("Ground truth, defined with thresholding")
        for plot, ax in zip(
            [plot_1, plot_2, plot_3], [ax[0], ax[1], ax[2]]
        ):
            plt.colorbar(plot, ax=ax, fraction=0.05, pad=0.04)
    plt.tight_layout()
    plt.savefig("../results/watershed_results.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Module implementing the segmentation"
        " of the lung ROI with the watershed algorithm and the"
        " region growing flood algorithm from scikit-image."
    )
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        help="The absolute path of the image directory.",
        metavar="str",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        nargs=3,
        help="The seed voxel coordinates from which"
        " the region growing segmentation starts."
        " Defaults to None if the user doesn't specify this argument.",
        default=None,
        metavar="int",
    )
    parser.add_argument(
        "-e",
        "--extrema",
        help="The values to crop the image matrix"
        " as image_matrix[extrema[0] : extrema[1], extrema[2] : extrema[3],"
        " extrema[4] : extrema[5]]",
        type=int,
        nargs=6,
        metavar="int",
    )
    args = parser.parse_args()

    image, aspect_ratio = ImageData.image_from_file(args.file_path, args.extrema)

    segmented_im = image_segmenter(image)
    ground_truth_image, _, _ = ground_truth(segmented_im)

    plot_display(
        image, aspect_ratio, segmented_im, ground_truth_image, args.seed
    )
    plt.show()
