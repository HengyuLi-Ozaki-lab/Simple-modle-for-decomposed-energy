# Polynomial fitting of decomposed energy for accelerating MD simulation

This repository is for preform the MLMD with polynomial fitting, the `History version` is include the history version and latest version. The `source` is the source code for OpenMx which include ML.c, you can type:

```
make
```

To generate the executable file `openmx`, the training iteration number and minimum order, maximum order of polynomial are setting at the front of ML.c, from line 22:

```c
/*******************************************************
     Hyperparameter for fitting decomposed energy
*******************************************************/

float lammda1 = 0.8, lammda2 = 0.0001;
int Max_order = 3;
int Min_order = -3;
int train_iter = 50;
```

Every time you change the hyperparameter you need to build again.
