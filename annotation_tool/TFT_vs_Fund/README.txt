Pose estimation of 3 views based on the trifocal tensor and the fundamental matrix.

This MATLAB(R) directory contains the code implemented for the PSIVT 2017 paper "A Critical Review of the Trifocal Tensor Estimation" by Laura F. Julia and Pascal Monasse.

Laura F. Julia, <laura.fernandez-julia@enpc.fr>, Univ. Paris Est, LIGM, ENPC, France

Version 0.1, July 2017

Future releases and updates:
https://github.com/LauraFJulia/TFT_vs_Fund.git


Files and folders
-----------------

TFT_methods/          - Pose estimation methods based on the TFT and auxiliary functions
F_methods/            - Pose estimation methods based on the F and auxiliary functions
Optimization/         - Optimization functions: Gauss_Helmert.m and BundleAdjustment.m
auxiliar_functions/   - Auxiliary functuions for triangulation, error computation, etc
Data/                 - Data necessary for experiments with EPFL datasets
README.txt            - This file
LICENSE.txt           - License file
example.m             - example script
experiments.m         - synthetic experiments script
experiments_real.m    - experiments script for real EPFL datasets


Setup and Usage
---------------

1. You must have MATLAB software installed on your computer.
2. The folowing toolboxes are required:
  * statistics_toolbox
  * optimization_toolbox
3. Copy/move the 'TFT_vs_Fund' folder to the MATLAB 'work' directory.
4. Run the example script example.m to easily understand how to use the code or run the scripts experiments.m and experiments_real.m to recreate the experiments mentionned in the paper.


Copyright and Licence
---------------------

Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>
All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

