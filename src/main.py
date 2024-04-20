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
# pylint: disable=W1203


"""Module implementing the CAM algorithm."""


import argparse
import logging
import time
import multiprocessing
import warnings
import numpy as np
import matplotlib.pyplot as plt
import CAM_functions as cam


warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s:%(levelname)s: %(message)s",
    filename="../results/log_results.txt",
    filemode="w",
    level=logging.INFO,
    encoding="utf-8",
)


parser = argparse.ArgumentParser(description="Module implementing the CAM algorithm.")
parser.add_argument(
    "-a",
    "--anthill_coordinates",
    help="The anthill voxel position.",
    type=int,
    nargs=3,
    metavar="int",
)
parser.add_argument(
    "-n",
    "--n_iteration",
    help="Number of iterations before stopping.",
    type=int,
    metavar="int",
)
subparsers = parser.add_subparsers(help="sub-command help", dest="cmd")
parser_path = subparsers.add_parser(
    "dicom", help="Returns an image from a DICOM folder."
)
parser_path.add_argument(
    "-f",
    "--file_path",
    help="The DICOM folder path.",
    type=str,
    metavar="str",
)
parser_cube = subparsers.add_parser("cube", help="Returns a cube as the image matrix.")
parser_cube.add_argument(
    "-m",
    "--matrix_dimensions",
    help="Image matrix dimensions.",
    type=int,
    nargs=3,
    metavar="int",
)
parser_cube.add_argument(
    "-c",
    "--center_coordinates",
    help="The cube center.",
    type=int,
    nargs=3,
    metavar="int",
)
parser_cube.add_argument(
    "-l", "--cube_length", help="The cube length.", type=int, metavar="int"
)
parser_curve = subparsers.add_parser(
    "sphere/ellipsoid", help="Returns a sphere/ellipsoid as the image matrix."
)
parser_curve.add_argument(
    "-m",
    "--matrix_dimensions",
    help="Image matrix dimensions.",
    type=int,
    nargs=3,
    metavar="int",
)
parser_curve.add_argument(
    "-c",
    "--center_coordinates",
    help="The center of the figure.",
    type=int,
    nargs=3,
    metavar="int",
)
parser_curve.add_argument(
    "-s",
    "--semi_axes",
    help="The semi-axes lengths.",
    type=float,
    nargs=3,
    metavar="float",
)
parser_curve.add_argument(
    "-r",
    "--radius",
    help="The radius of the figure.",
    type=int,
    metavar="int",
)
parser_donut = subparsers.add_parser(
    "donut",
    help="Returns a donut as the image matrix i.e a sphere with a concentric hole with half external radius as the internal radius.",
)
parser_donut.add_argument(
    "-m",
    "--matrix_dimensions",
    help="Image matrix dimensions.",
    type=int,
    nargs=3,
    metavar="int",
)
parser_donut.add_argument(
    "-c",
    "--center_coordinates",
    help="The center of the donut.",
    type=int,
    nargs=3,
    metavar="int",
)
parser_donut.add_argument(
    "-r",
    "--radius",
    help="The external radius of the donut.",
    type=int,
    metavar="int",
)
args = parser.parse_args()


if __name__ == "__main__":

    image, aspect_ratio, pheromone_map_init, pheromone_shared, pheromone_map = (
        cam.set_image_and_pheromone(args)
    )
    image_voxels, thresh_value = cam.cam_ground_truth(args, image)
    # copies pheromone_map_init into pheromone_map
    np.copyto(pheromone_map, pheromone_map_init)

    n_iteration = 0
    ENERGY_DEATH = 1.
    ENERGY_REPRODUCTION = 1.3
    ant_number = []
    pheromone_mean_sum = 0
    pheromone_mean_list = []
    colony_length = 0

    ant_colony, anthill_position = cam.set_colony(args.anthill_coordinates, image, thresh_value)
    start_time_local = time.perf_counter()

    while len(ant_colony) != 0 and n_iteration <= args.n_iteration:
        print(f"Iter:{n_iteration}\t#ants:{len(ant_colony)}\n")

        chunksize = max(2, len(ant_colony) // 6)
        with multiprocessing.Pool(
            processes=6,
            initializer=cam.pool_initializer,
            initargs=(pheromone_shared, pheromone_map.shape),
        ) as pool:
            released_pheromone = pool.map(
                cam.update_pheromone_map, ant_colony, chunksize=chunksize
            )
            next_voxel = pool.map(
                cam.find_next_voxel, ant_colony, chunksize=chunksize
            )

        colony_length += len(ant_colony)
        pheromone_mean_sum += sum(released_pheromone)
        pheromone_mean = pheromone_mean_sum / colony_length
        pheromone_mean_list.append(pheromone_mean)

        for i, ant in enumerate(ant_colony):
            if len(next_voxel[i]) == 0:
                del ant_colony[i]
                del released_pheromone[i]
                continue
            ant.voxel_coordinates = next_voxel[i]
            ant.update_energy(released_pheromone[i] / pheromone_mean)

        for i, ant in enumerate(ant_colony):
            if ant.energy < ENERGY_DEATH:
                del ant_colony[i]
                if len(ant_colony) == 0:
                    print("No ants left.\n")
                continue
            if ant.energy > ENERGY_REPRODUCTION:
                ant.energy = 1.0 + ant.alpha
                first_neighbours = ant.find_first_neighbours()
                valid_index = np.where(
                    pheromone_map[
                        first_neighbours[:, 0],
                        first_neighbours[:, 1],
                        first_neighbours[:, 2],
                        1,
                    ]
                    == 0
                )[0]
                valid_neighbours = np.array(
                    [first_neighbours[index] for index in valid_index]
                )
                if valid_neighbours.shape[0] == 0:
                    continue

                second_neighbours = ant.find_second_neighbours()
                image_second_neigh = ant_colony[i].pheromone_release(
                    second_neighbours
                )

                image_second_neigh_mean = image_second_neigh.mean()
                image_second_max = np.amax(image_second_neigh)
                image_second_min = np.amin(image_second_neigh)

                try:
                    n_offspring = min(
                        int(
                            26
                            * (image_second_neigh_mean - image_second_min)
                            / (image_second_max - image_second_min)
                        ),
                        valid_neighbours.shape[0],
                    )
                except (ValueError, OverflowError):
                    n_offspring = valid_neighbours.shape[0]

                for neigh in valid_neighbours[:n_offspring]:
                    ant_colony.append(cam.Ant(image, list(neigh), thresh_value))
                pheromone_map[
                    valid_neighbours[:, 0],
                    valid_neighbours[:, 1],
                    valid_neighbours[:, 2],
                    1,
                ] = 1
                continue

        ant_number.append(len(ant_colony))
        n_iteration += 1

    logging.info(f"Image dimensions: {image.shape}\n")
    logging.info(f"Anthill coordinates: {args.anthill_coordinates}\n")
    logging.info(f"Energy death: {ENERGY_DEATH}\n")
    logging.info(f"Energy reproduction: {ENERGY_REPRODUCTION}\n")
    logging.info(f"# iterations: {n_iteration}\n")
    logging.info(
        f"Elapsed time: {(time.perf_counter() - start_time_local) / 60:.3f} min\n"
    )

    cam.statistics(
        ant_number,
        pheromone_mean_list,
        image,
        pheromone_map[..., 0],
        image_voxels,
    )
    cam.plot_display(aspect_ratio, image, pheromone_map[..., 0], anthill_position)
    plt.show()
