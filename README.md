# Polynomial fitting of decomposed energy for accelerating MD simulation

## Compile and Input

This repository is for preform the MLMD with polynomial fitting, the `History version` is include the history version and latest version. The `source` is the source code for OpenMx which include ML.c, you can type:

```
make
```

The example input file is given in Methane.dat, the parameter for ML need to be modified in `ML` section:

```
#
# ML
#

ML.status           on
ML.force_status     off
ML.Max_order        3
ML.Min_order        -3
ML.Train_iter       50
ML.Correction_iter  10
ML.Lammda_1         0.7
ML.Lammda_2         0.0001
```
The `ML.status` is to control whether start the ML process.

The `ML.force_status` is to control whether replace the atomic force by model force.

The `ML.Max_order` and `ML.Min_order` are max order and min order of fitting polynomial.

The `ML.Train_iter` is the training iteration number of on-the-fly preparation.

The `ML.Correction_iter` is the correction iteration number, which means after `ML.Correction_iter` the code will do 1 DFT calculation to retrain the model.

The `ML.Lammda_1` and `ML.Lammda_2` are hyperparameter of the loss function.

## Visualization.py

Visualization.py is a python script for plotting the visualization of model energy, model force and error respect to the MD iteration. 

To use Visualization.py first copy the Plot.py to work folder and then execute:
```
python3 Visualization.py
```
Following folders should be generated:
```
energy fig
force fig
error fig
```
following is an example of Methane.dat:

1. Energy visualization 
![H1_energy](https://user-images.githubusercontent.com/66453357/164589202-6d1ff07f-c4e2-4f69-b362-433b40c8cb66.jpg)
The figure shows the comparison of the model energy (2-body, 3-body, total) and DFT energy with the MD iteration (x axis).

2. Force visualization
![H1_force_x](https://user-images.githubusercontent.com/66453357/164589221-498ca81d-0e5c-4ce9-bf66-23511f2d27fb.jpg)
The figure shows the comparison of the model force (C in x direction) and DFT force with the MD iteration (x axis). Atomic force in x, y, z direction are all plotted.

3. Error visualization
![H_energy_error](https://user-images.githubusercontent.com/66453357/164589235-0417c632-8096-4143-945f-aacf6bd2ff00.jpg)

![H_force_error](https://user-images.githubusercontent.com/66453357/164589244-9b706e45-d4e2-4645-a461-980f85025f77.jpg)
The x,y axis are respect to error and MD iteration. The black dash line is for indicating the end of training. The color curve is to show the error tendency. The error figures are organized by element type.
