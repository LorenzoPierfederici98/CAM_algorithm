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

import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import flood_fill, flood
from environment_constructor import ImageData

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
image_matrix, aspect_ratio = ImageData.image_from_file(args.file_path)
segmented_image = flood_fill(
    image_matrix,
    tuple(args.seed_coordinates),
    image_matrix[args.seed_coordinates[0], args.seed_coordinates[1], args.seed_coordinates[2]],
)
segm_mask = flood(
    image_matrix[:, :, args.seed_coordinates[2]],
    (args.seed_coordinates[0], args.seed_coordinates[1]),
    #connectivity=500,
    tolerance=700,
)

print(np.nonzero(segm_mask))

print(image_matrix[args.seed_coordinates[0], args.seed_coordinates[1], args.seed_coordinates[2]])
fig, ax = plt.subplots(1, 3)
ax[0].set_aspect(aspect_ratio["axial"])
plot_0 = ax[0].imshow(image_matrix[:, :, args.seed_coordinates[2]], cmap="gray")
ax[0].set_title("Original image")
plot_1 = ax[1].imshow(segmented_image[:, :, args.seed_coordinates[2]], cmap="gray")
ax[1].plot(args.seed_coordinates[1], args.seed_coordinates[0], "ro", label="Seed point")
ax[1].legend()
ax[1].set_title("Segmented image")
plot_2 = ax[2].imshow(segm_mask)
for plot, ax in zip([plot_0, plot_1, plot_2], [ax[0], ax[1], ax[2]]):
    plt.colorbar(plot, ax=ax)
plt.tight_layout()
plt.show()
