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
ML.Max_order        3
ML.Min_order        -3
ML.Train_iter       50
ML.Correction_iter  10
ML.Lammda_1         0.7
ML.Lammda_2         0.0001
```
The `ML.status` is to control whether start the ML process.

The `ML.Max_order` and `ML.Min_order` are max order and min order of fitting polynomial.

The `ML.Train_iter` is the training iteration number of on-the-fly preparetion.

The `ML.Correction_iter` is the correction iteration number, which means after `ML.Correction_iter` the code will do 1 DFT calculation to retrain the model.

The `ML.Lammda_1` and `ML.Lammda_2` are hyperparameter of the loss function.

## Plot.py

Plot.py is a python script for ploting the error respect to each MD iteration, following is an example of Methane.dat:
![H](https://user-images.githubusercontent.com/66453357/155910297-5621015a-f73b-4837-8331-360e44703ddb.jpg)
The x,y axis are respect to error and MD iteration. The black dash line is for indicating the end of training. The color curve is to show the error tendency.

To use Plot.py first copy the Plot.py to work folder and then execute:
```
python3 Plot.py
```
Then you can select the plot for energy error or force error by entering:
```
force/energy
```
Following figures should be generated:
```
C.jpg
H.jpg
```
The error figures are organized by element type.
