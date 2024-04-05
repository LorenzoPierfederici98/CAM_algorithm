.. CAM algorithm documentation master file, created by
   sphinx-quickstart on Fri Mar 15 12:55:24 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CAM algorithm's documentation!
=========================================

Repository of the project for the CMEPDA course implementing a Channeler Ant Model (CAM) algorithm in order to segment aerial trees, following the model described by Cerello et al. in the paper cited in the *References* section of the `github repository <https://github.com/LorenzoPierfederici98/CAM_algorithm>`_ .


Overview
^^^^^^^^
The CAM algorithm exploits virtual ant colonies whose behaviour mimics the cooperation strategies put in place by real ants, which is here used in image processing. The ant colony lives in the digital habitat of the image voxels, in which ants move and deposit pheromone in order to build a pheromone map. 

The life cycle of the ants is discretized in iterations: starting from the anthill voxel (chosen by the user) and all its first-order neighbours, the ants deposit pheromone values, corresponding to the respective image voxels intensities, and evaluate the next voxel destination among all the first-order neighbouring voxels not occupied by an ant. The evaluation of the destination voxel is made by computing a probability for all the free neighbouring voxels, which depends on their pheromone values; the next voxel is chosen with a roulette wheel algorithm, in order to find a balance between random paths and "directional" paths given by the pheromone trails.

The ants lifespan is reuglated by the energy parameter: all the ants are assigned with a default value which varies with every iteration, depending on the pheromone value released by the ant and the pheromone mean per iteration released by the ant colony since the first iteration. Whenever an ant has energy greater than a reproduction value it generates :math:`N_{offspring}\in[0, 26]` ants, related to the local properties of the enviornment, which are placed in the free first-order neighbouring voxels; if the energy is lower than a certain value or if the ant has no possible voxel destination it dies. Following those rules the ants build the pheromone map, which is deployed to segment bronchial and vascular trees in lung CT images.


Moving rules
^^^^^^^^^^^^
The moving rules take into account both randomness and the colony global knowledge of the environment, provided by the pheromone values stored in the first-neighbouring voxels :math:`\sigma_j` : a large amount of pheromone in a voxel corresponds to a higher probability to choose that one as a possibile destination for an ant. Starting from an ant at a voxel :math:`v_i` it is computed the probability to choose a first-neighbouring voxel :math:`v_j`, which must be ant-free and must have been visited a number of times lower than :math:`N_V` (defined later), as follows

.. math:: P_{ij}(v_i\to v_j) = \frac{W(\sigma_j)}{\sum_{n_{neigh}}W(\sigma_n)}

Where :math:`W(\sigma_j) = (1 + \frac{\sigma_j}{1 + \delta \sigma_j})^{\beta}` depends on the *osmotro-potaxic sensitivity* :math:`\beta = 3.5` representing the pheromone trail influence in choosing the voxel destination and the *sensory capacity* :math:`1/\delta = 1/0.2` which determines a decrease in the ant sensitivity relatively to the pheromone if its concentration is too high in a voxel. A random float :math:`rand` between 0 and :math:`\max{P_{ij}}` is then extracted and the first voxel for which :math:`P_{ij}\geq rand` is chosen to be the destination.


Pheromone laying rule
^^^^^^^^^^^^^^^^^^^^^
The pheromone value an ant deposits into a voxel :math:`v_i` before leaving is

.. math:: T = \eta + \Delta_{ph}\quad \Delta_{ph} = 10 I(v_i)

With :math:`I(v_i)` the intensity of the corresponding image voxel and :math:`\eta = 0.01` a quantity that an ant would leave even into a voxel with zero intensity, certifying that it was visited. See the reference article for other laying rules.


Life cycle
^^^^^^^^^^
The life cycle of the ants is regulated by the energy parameter with a default value of :math:`\varepsilon_0 = 1 + \alpha` with :math:`\alpha = 0.2`. The energy update of an ant takes into account the amount of pheromone deposited by it :math:`\Delta_{ph}` and the average amount of pheromone per iteration released by the colony since the start :math:`<\Delta_{ph}>`

.. math:: \Delta\varepsilon = -\alpha (1 - \frac{\Delta_{ph}}{<\Delta_{ph}>})

An ant dies whenever :math:`\varepsilon < \varepsilon_D = 0.3` and gives birth whenever :math:`\varepsilon > \varepsilon_R = 1.25`.

The number of ants generated when a reproduction takes place :math:`N_{offspring}` is a function of the local properties of the environment, which are evaluated replacing :math:`T` with :math:`T_5` the pheromone releasing rule considering the intensity :math:`I_5` as the image intensity averaged on the second-order neighbours of the ant current voxel.

.. math:: N_{offspring} = 26 \frac{T_5 - T_{5,min}}{T_{5,max} - T_{5,min}}

:math:`T_{5,min}` :math:`T_{5,max}` being the minimum and maximum pheromone releases in the second-order neighbours. If :math:`N_{offspring}` is greater than the number of free first-order neighbours, it is set to this latter.


Number of visits per voxel
^^^^^^^^^^^^^^^^^^^^^^^^^^
The number of visits a voxel V can receive :math:`N_V` is voxel and pheromone dependent: in areas with small pheromone deposition a larger number of visits is allowed, vice versa in areas with larger pheromone deposition. The pheromone deposition ranges from :math:`T_{min}` to :math:`T_{max}`, so that for the voxel V:

.. math:: N_V = 40 + 80 \frac{T - T_{max}}{T_{min} - T_{max}}

:math:`N_V` ranges from 40 to 120. A voxel which has reached the maximum number cannot be further visited by the colony.


Evaluation metrics
^^^^^^^^^^^^^^^^^^
CAM performances are evaluated defining threshold values above which voxels can be considered as segmented. The following quantities are defined:

* Sensitivity :math:`S = N_R/N_O` the ratio between the number of segmented voxels which are also part of the image voxels and the number of voxels in these latter.
* Exploration level :math:`E = N_S/N_O` the ratio between the number of all segmented voxels and the number of voxels in the original image objects.
* Contamination level :math:`C = N_C/N_O` the ratio between the number of segmented voxels which are not part of the image objects and these latter. It corresponds to :math:`C = E - S`.

These quantities are evaluated as functions of the pheromone threshold.



.. toctree::
   :maxdepth: 5
   :caption: Contents:

   antconstr
   envconstr
   main_doc
   license
   help

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
