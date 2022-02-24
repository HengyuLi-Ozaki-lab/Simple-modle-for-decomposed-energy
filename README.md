# Polynomial fitting of decomposed energy for accelerating MD simulation

This repository is for preform the MLMD with polynomial fitting, the `History version` is include the history version and latest version. The `source` is the source code for OpenMx which include ML.c, you can type:

```
make
```

The example input file is given in Methane.dat, the parameter for ML need to be modified in `ML` section:

```
#
# ML
#

ML.status             on
ML.Max_order     3
ML.Min_order      -3
ML.Train_iter       50
ML.Correction_iter 10
ML.Lammda_1    0.7
ML.Lammda_2    0.0001
```
The `ML.status` is to control whether start the ML process.

The `ML.Max_order` and `ML.Min_order` are max order and min order of fitting polynomial.

The `ML.Train_iter` is the training iteration number of on-the-fly preparetion.

The `ML.Correction_iter` is the correction iteration number, which means after `ML.Correction_iter` the code will do 1 DFT calculation to retrain the model.

The `ML.Lammda_1` and `ML.Lammda_2` are hyperparameter of the loss function.
