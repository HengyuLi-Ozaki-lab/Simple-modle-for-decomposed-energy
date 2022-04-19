/**********************************************************************
  ML.c:

     MLDFT.c is a subroutine to perform ML prediction of atomic force

  Log of ML.c:

     10/Sep/2021  Added by Hengyu Li

     ver 1.0 train every 10 MD step for error testing

***********************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "openmx_common.h"
#include "lapack_prototypes.h"

#define Pi 3.141592654

/*******************************************************
    Arry for distance,angular,matrix A B C A' B' C'
*******************************************************/

static int *angular_num;
static double **Dec_tot;
static double **dis_nei;
static double **ang_nei;
static double ***matrix_a;
static double ***matrix_b;
static double ***matrix_a_;
static double ***matrix_b_;
static double ***twobody_ene;
static double ***threebody_ene;
static double **numerical_force;
static double **matrix_c;
static double **matrix_c_;
static double **fitted_energy;
static double **model_force;
static double **total_force;
static double **loss;
static double **force_error;
static double **constant_matrix;
static double **parameter_matrix;
static double **current_model;

/*******************************************************
                Subfunction of ML
*******************************************************/

/* Compute three-body combination number */

int factorial(int m, int n)
{
  int i,j;
	int ans = 1;
  
	if(m < n-m) m = n-m;
	for(i = m+1; i <= n; i++) ans *= i;
	for(j = 1; j <= n - m; j++) ans /= j;

	return ans;

}

/* Compute distance for each two-body combination */

void cal_dis(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{
  char filelast[YOUSO10] = ".cal_dis";
  int i,j,k;
  FILE *fp;

  fnjoint(filepath,filename,filelast);
  fp = fopen(filelast,"a");

  fprintf(fp,"Distance at MD iter =%d\n",iter);

  for (i=1; i<=atomnum; i++){
    for (j=1; j<=FNAN[i]; j++){
      k = natn[i][j];
      dis_nei[i][j] = sqrt((Gxyz[k][1]-Gxyz[i][1])*(Gxyz[k][1]-Gxyz[i][1])+(Gxyz[k][2]-Gxyz[i][2])*(Gxyz[k][2]-Gxyz[i][2])+(Gxyz[k][3]-Gxyz[i][3])*(Gxyz[k][3]-Gxyz[i][3]));
      fprintf(fp,"%8.8f ",dis_nei[i][j]);
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
}

/* Compute angular for each three-body combination */

void cal_ang(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{
  char filelast[YOUSO10] = ".cal_ang";
  int i,j,k,nei_gnum1,nei_gnum2,nei_num,count;
  FILE *fp;

  fnjoint(filepath,filename,filelast);
  fp = fopen(filelast,"a");

  fprintf(fp,"Angular at MD iter =%d\n",iter);

  for (i=1; i<=atomnum; i++){
    nei_num = FNAN[i];
    count = 1;
    for (j=1; j<=nei_num-1; j++){
      nei_gnum1 = natn[i][j];
      for (k=j+1; k<=nei_num; k++){
        nei_gnum2 = natn[i][k];
        ang_nei[i][count] = ((Gxyz[nei_gnum1][1]-Gxyz[i][1])*(Gxyz[nei_gnum2][1]-Gxyz[i][1])+\
        (Gxyz[nei_gnum1][2]-Gxyz[i][2])*(Gxyz[nei_gnum2][2]-Gxyz[i][2])+\
        (Gxyz[nei_gnum1][3]-Gxyz[i][3])*(Gxyz[nei_gnum2][3]-Gxyz[i][3]))/(dis_nei[i][j]*dis_nei[i][k]);
        fprintf(fp,"%8.8f ",ang_nei[i][count]);
        count ++;
      }
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
}

/* Compute cutoff coefficient */

double cut_off(double distance,double r_cut,int grad)
{
  double coefficient;

  /* Output normal cutoff function */

  if (grad == 0){
    if (distance <= r_cut){
      coefficient = 0.5*(cos(Pi*distance/r_cut)+1);
      return coefficient;
    }
    else{
      return 0;
    }
  }

  /* Output normal cutoff function */
  
  else if (grad == 1){
    if (distance <= r_cut){
      coefficient = 0.5*Pi*sin(Pi*distance/r_cut)/r_cut; // (rjx-rix)/r_ij is not inclued
      return coefficient;
    }
    else{
      return 0;
    }
  }

  /* Input error */

  else{
    printf("Check grad control");
    return 0;
  }
}

/* Get decomposed energy repect to atom on each processor */

void Get_decomposed_ene(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{

  int i, j, spin, local_num, global_num, ID, species,tag=999,myid;
  double energy;

  /* MPI */
  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);

  for (global_num=1; global_num<=atomnum; global_num++){
    species = WhatSpecies[global_num];
    ID = G2ID[global_num];
    energy = 0;

    if (myid==ID){

      local_num = F_G2M[global_num];

      if (SpinP_switch==0){
        for (i=0;i<Spe_Total_CNO[species];i++){
          energy += 2*DecEkin[0][local_num][i]; 
          energy += 2*DecEv[0][local_num][i];
          energy += 2*DecEcon[0][local_num][i];
          energy += 2*DecEscc[0][local_num][i];
          energy += 2*DecEvdw[0][local_num][i];
        }
      }

      printf("Add test pass for atom %d at processor %d\n",global_num,ID);

      if (SpinP_switch==1 || SpinP_switch==3){
        for (i=0;i<Spe_Total_CNO[species];i++){
          for (spin=0;spin<=1;spin++){
            energy += DecEkin[spin][local_num][i];
            energy += DecEv[spin][local_num][i];
            energy += DecEcon[spin][local_num][i];
            energy += DecEscc[spin][local_num][i];
            energy += DecEvdw[spin][local_num][i];
          }
        }
      }

      if (myid!=Host_ID){
        MPI_Send(&energy,1,MPI_DOUBLE,Host_ID,tag,mpi_comm_level1);
        printf("Send test pass for atom %d at processor %d\n",global_num,ID);        
      }

    }

    if (myid == Host_ID){

      if (ID!=Host_ID){
        MPI_Recv(&energy, 1, MPI_DOUBLE, ID, tag, mpi_comm_level1, &status);
        printf("Receive test Pass for atom %d at processor %d\n",global_num,ID);        
      }

      Dec_tot[global_num][iter] = energy;
      printf("Append test Pass of atom %d at processor %d\n",global_num,ID);

    }
  }

}

/* Allocate working array and matrice for ML */

void ML_allocate()
{
  int i,j,k,nei_num;

  angular_num = (int*)malloc(sizeof(int)*(atomnum+1));
  memset(angular_num,0,(atomnum+1)*sizeof(int));  

  printf("Angular number array allocate Pass\n");

  for (i=1;i<=atomnum;i++){
    angular_num[i] = factorial(2,FNAN[i]);
  }

  dis_nei = (double**)malloc(sizeof(double*)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    dis_nei[i] = (double*)malloc(sizeof(double)*(FNAN[i]+1));
    memset(dis_nei[i],0,(FNAN[i]+1)*sizeof(double));
  }

  printf("Distance calculation Pass\n");

  ang_nei = (double**)malloc(sizeof(double*)*atomnum+1);
  for(i=1;i<=atomnum;i++){
    ang_nei[i] = (double*)malloc(sizeof(double)*(angular_num[i]+1));
    memset(ang_nei[i],0,(angular_num[i]+1)*sizeof(double));
  }

  printf("Angular calculation Pass\n");

  matrix_a = (double***)malloc(sizeof(double**)*(atomnum+1));
  for (i=1; i<=atomnum; i++){
    matrix_a[i] = (double**)malloc(sizeof(double*)*(FNAN[i]*(Max_order-Min_order+1)+1)); 
    for (j=1; j<=(FNAN[i]*(Max_order-Min_order+1)); j++){
      matrix_a[i][j] = (double*)malloc(sizeof(double)*(FNAN[i]*(Max_order-Min_order+1)+1)); 
      memset(matrix_a[i][j],0,(FNAN[i]*(Max_order-Min_order+1)+1)*sizeof(double));
    }
  }

  printf("Matrix A allocate Pass\n");

  matrix_b = (double***)malloc(sizeof(double**)*(atomnum+1));
  for (i=1; i<=atomnum; i++){
    matrix_b[i] = (double**)malloc(sizeof(double*)*(FNAN[i]*(Max_order-Min_order+1)+1)); 
    for (j=1; j<=(FNAN[i]*(Max_order-Min_order+1)); j++){
      matrix_b[i][j] = (double*)malloc(sizeof(double)*((Max_order-Min_order+1)*angular_num[i]+1));
      memset(matrix_b[i][j],0,((Max_order-Min_order+1)*angular_num[i]+1)*sizeof(double));
    }
  }

  printf("Matrix B allocate Pass\n");

  matrix_c = (double**)malloc(sizeof(double*)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    matrix_c[i] = (double*)malloc(sizeof(double)*(FNAN[i]*(Max_order-Min_order+1)+1));
    memset(matrix_c[i],0,(FNAN[i]*(Max_order-Min_order+1)+1)*sizeof(double));
  }

  printf("Matrix C allocate Pass\n");

  matrix_a_ = (double***)malloc(sizeof(double**)*(atomnum+1));
  for (i=1; i<=atomnum; i++){
    matrix_a_[i] = (double**)malloc(sizeof(double*)*((Max_order-Min_order+1)*angular_num[i]+1)); 
    for (j=1; j<=((Max_order-Min_order+1)*angular_num[i]); j++){
      matrix_a_[i][j] = (double*)malloc(sizeof(double)*(FNAN[i]*(Max_order-Min_order+1)+1));
      memset(matrix_a_[i][j],0,(FNAN[i]*(Max_order-Min_order+1)+1)*sizeof(double));
    }
  }

  printf("Matrix A' allocate Pass\n");

  matrix_b_ = (double***)malloc(sizeof(double**)*(atomnum+1));
  for (i=1; i<=atomnum; i++){
    matrix_b_[i] = (double**)malloc(sizeof(double*)*((Max_order-Min_order+1)*angular_num[i]+1)); 
    for (j=1; j<=((Max_order-Min_order+1)*angular_num[i]); j++){
      matrix_b_[i][j] = (double*)malloc(sizeof(double)*((Max_order-Min_order+1)*angular_num[i]+1));
      memset(matrix_b_[i][j],0,((Max_order-Min_order+1)*angular_num[i]+1)*sizeof(double)); 
    }
  }

  printf("Matrix B' allocate Pass\n");

  matrix_c_ = (double**)malloc(sizeof(double*)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    matrix_c_[i] = (double*)malloc(sizeof(double)*((Max_order-Min_order+1)*angular_num[i]+1));
    memset(matrix_c_[i],0,((Max_order-Min_order+1)*angular_num[i]+1)*sizeof(double)); 
  }

  printf("Matrix C' allocate Pass\n");

  parameter_matrix = (double**)malloc(sizeof(double*)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    nei_num = FNAN[i];
    parameter_matrix[i] = (double*)malloc(sizeof(double)*(pow((nei_num+angular_num[i])*(Max_order-Min_order+1),2)+1));
    memset(parameter_matrix[i],0,(pow((nei_num+angular_num[i])*(Max_order-Min_order+1),2)+1)*sizeof(double));
  }

  printf("Parameter array allocate Pass\n");

  constant_matrix = (double**)malloc(sizeof(double*)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    nei_num = FNAN[i];
    constant_matrix[i] = (double*)malloc(sizeof(double)*((nei_num+angular_num[i])*(Max_order-Min_order+1)+1));
    memset(constant_matrix[i],0,((nei_num+angular_num[i])*(Max_order-Min_order+1)+1)*sizeof(double));
  }

  printf("Constant array allocate Pass\n");

  current_model = (double**)malloc(sizeof(double*)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    nei_num = FNAN[i];
    current_model[i] = (double*)malloc(sizeof(double)*((nei_num+angular_num[i])*(Max_order-Min_order+1)+1));
    memset(current_model[i],0,((nei_num+angular_num[i])*(Max_order-Min_order+1)+1)*sizeof(double));
  }

  printf("Final model array allocate Pass\n");

  Dec_tot = (double**)malloc(sizeof(double)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    Dec_tot[i] = (double*)malloc(sizeof(double)*(MD_IterNumber+1));
    memset(Dec_tot[i],0,(MD_IterNumber+1)*sizeof(double));
  }

  printf("Decomposed energy array allocate Pass\n");

  fitted_energy = (double**)malloc(sizeof(double)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    fitted_energy[i] = (double*)malloc(sizeof(double)*(MD_IterNumber+1));
    memset(fitted_energy[i],0,(MD_IterNumber+1)*sizeof(double));
  }

  printf("Model fitting energy array allocate Pass\n");

  twobody_ene = (double***)malloc(sizeof(double)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    nei_num = FNAN[i];
    twobody_ene[i] = (double**)malloc(sizeof(double)*(MD_IterNumber+1));
    for (j=1;j<=MD_IterNumber;j++){
      twobody_ene[i][j] = (double*)malloc(sizeof(double)*(nei_num+1));
      memset(twobody_ene[i][j],0,(nei_num+1)*sizeof(double));
    }
  }

  printf("Twobody energy array allocate Pass\n");

  threebody_ene = (double***)malloc(sizeof(double)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    threebody_ene[i] = (double**)malloc(sizeof(double)*(MD_IterNumber+1));
    for (j=1;j<=MD_IterNumber;j++){
      threebody_ene[i][j] = (double*)malloc(sizeof(double)*(angular_num[i]+1));
      memset(threebody_ene[i][j],0,(angular_num[i]+1)*sizeof(double));
    }
  }

  printf("Threebody energy array allocate Pass\n");

  loss = (double**)malloc(sizeof(double)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    loss[i] = (double*)malloc(sizeof(double)*(MD_IterNumber+1));
    memset(loss[i],0,(MD_IterNumber+1)*sizeof(double));
  }

  printf("Fitting energy error array allocate Pass\n");

  force_error = (double**)malloc(sizeof(double)*(atomnum+1));
  for(i=1;i<=atomnum;i++){
    force_error[i] = (double*)malloc(sizeof(double)*(MD_IterNumber+1));
    memset(force_error[i],0,(MD_IterNumber+1)*sizeof(double));
  }

  printf("Fitting energy error array allocate Pass\n");

  model_force = (double**)malloc(sizeof(double)*(atomnum+1));
  for (i=1;i<=atomnum;i++){
    model_force[i] = (double*)malloc(sizeof(double)*4);
    memset(model_force[i],0,4*sizeof(double));
  }

  printf("Model force array allocate Pass\n");

  numerical_force = (double**)malloc(sizeof(double)*(atomnum+1));
  for (i=1;i<=atomnum;i++){
    numerical_force[i] = (double*)malloc(sizeof(double)*4);
    memset(numerical_force[i],0,4*sizeof(double));
  }

  printf("Numerical force array allocate Pass\n");

  total_force = (double**)malloc(sizeof(double)*4);
  for (i=1;i<=3;i++){
    total_force[i] = (double*)malloc(sizeof(double)*(MD_IterNumber+1));
    memset(total_force[i],0,(MD_IterNumber+1)*sizeof(double));
  }

  printf("Total force array allocate Pass\n");

  printf("All array allocate Pass\n");

}

/* Output information */

void ML_output(int iter, char filepath[YOUSO10], char filename[YOUSO10], char keyword[YOUSO10])
{
  int i,j,k,nei_num;
  char target_file[YOUSO10];
  FILE *fp;

  strcpy(target_file, keyword);

  fnjoint(filepath,filename,target_file);

  fp = fopen(target_file,"a");
  
  if (keyword==".matrix"){

    fprintf(fp,"Matrices at MD iter =%d\n",iter);

    /* Output matrix A */

    fprintf(fp,"Matrix A\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix A for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*FNAN[i]; j++){
        for (k=1; k<=(Max_order-Min_order+1)*FNAN[i]; k++){
          fprintf(fp,"%8.8f ",matrix_a[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }
    fprintf(fp,"\n");

    /* Output matrix C */

    fprintf(fp,"Matrix C\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix C for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*FNAN[i]; j++){
        fprintf(fp,"%8.8f ",matrix_c[i][j]);
      }
      fprintf(fp,"\n");
    }
    fprintf(fp,"\n");

    /* Output matrix B */

    fprintf(fp,"Matrix B\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix B for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*FNAN[i]; j++){
        for (k=1; k<=(Max_order-Min_order+1)*angular_num[i]; k++){
          fprintf(fp,"%8.8f ",matrix_b[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }
    fprintf(fp,"\n");

    /* Output matrix A' */

    fprintf(fp,"Matrix A'\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix A' for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*angular_num[i]; j++){
        for (k=1; k<=(Max_order-Min_order+1)*FNAN[i]; k++){
          fprintf(fp,"%8.8f ",matrix_a_[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }
    fprintf(fp,"\n");

    /* Output matrix C' */

    fprintf(fp,"Matrix C'\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix C' for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*angular_num[i]; j++){
        fprintf(fp,"%8.8f ",matrix_c_[i][j]);
      }
      fprintf(fp,"\n");
    }
    fprintf(fp,"\n");

    /* Output matrix B' */

    fprintf(fp,"Matrix B'\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix B' for atom %d\n", i);
      for (j=1; j<=(Max_order-Min_order+1)*angular_num[i]; j++){
        for (k=1; k<=(Max_order-Min_order+1)*angular_num[i]; k++){
          fprintf(fp,"%8.8f ",matrix_b_[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }

    fclose(fp);

    printf("Out matrix Pass\n");

  }

  else if (keyword==".solver_input"){

    fprintf(fp,"Parameter and constant array at MD iter =%d\n",iter);

    for (i=1;i<=atomnum;i++){
      nei_num = FNAN[i];
      fprintf(fp,"Parameter matrix for atom %d\n",i);
      for (j=1;j<=pow((nei_num+angular_num[i])*(Max_order-Min_order+1),2);j++){
        fprintf(fp,"%8.8f ",parameter_matrix[i][j]); 
      }
      fprintf(fp,"\n");
      fprintf(fp,"Constant matrix for atom %d\n",i);
      for (k=1;k<=(nei_num+angular_num[i])*(Max_order-Min_order+1);k++){ 
        fprintf(fp,"%8.8f ",constant_matrix[i][k]);
      }
      fprintf(fp,"\n");
    }

    fclose(fp);

    printf("Out solver input Pass\n");

  }

  else if (keyword==".fitted_parameter"){

    fprintf(fp,"Fitted parameters for MD %d\n",iter);

    for (i=1;i<=atomnum;i++){
      nei_num =FNAN[i];
      fprintf(fp,"atom %d\n",i);
      for (j=1;j<=(nei_num+angular_num[i])*(Max_order-Min_order+1);j++){
        fprintf(fp,"%8.8f ",constant_matrix[i][j]);
      }
      fprintf(fp,"\n");
    }

    fclose(fp);

    printf("Out fitted parameter Pass\n");

  }
  
  else if (keyword==".error"){

    if (iter==MD_IterNumber){
      fprintf(fp,"Fitting error for each atom\n");
      for (i=1;i<=atomnum;i++){
        fprintf(fp,"Atom %d ",i);
        for (j=1;j<=MD_IterNumber;j++){
          fprintf(fp,"%8.8f ",loss[i][j]);
        }
        fprintf(fp,"\n");
      }
      fclose(fp);

      printf("Out fitting error Pass\n");
    }
  }

  else if (keyword==".force_error"){

    if (iter==MD_IterNumber){
      fprintf(fp,"Fitting error for each atom\n");
      for (i=1;i<=atomnum;i++){
        fprintf(fp,"Atom %d ",i);
        for (j=1;j<=MD_IterNumber;j++){
          fprintf(fp,"%8.8f ",force_error[i][j]);
        }
        fprintf(fp,"\n");
      }
      
      fprintf(fp,"\n");
      fclose(fp);

      printf("Out fitting force error Pass\n");
    }
  }
  
  else if (keyword==".fitted_energy"){

    if (iter==MD_IterNumber){
      fprintf(fp,"Fitting energy for each atom\n");
      for (i=1;i<=atomnum;i++){
        fprintf(fp,"Atom %d ",i);
        for (j=1;j<=MD_IterNumber;j++){
          fprintf(fp,"%8.8f ",fitted_energy[i][j]);
        }
        fprintf(fp,"\n");
      }

      fclose(fp);      
    }

    printf("Out fitted energy Pass\n");

  }

  else if (keyword==".twobody_energy"){

    if (iter==MD_IterNumber){
      fprintf(fp,"Twobody energy for each atom\n");
      for (i=1;i<=atomnum;i++){
        nei_num = FNAN[i];
        fprintf(fp,"Atom %d ",i);
        for (j=1;j<=MD_IterNumber;j++){
          for (k=1;k<=nei_num;k++){
            fprintf(fp,"%8.8f ",twobody_ene[i][j][k]);
          }
          fprintf(fp,",");
        }
        fprintf(fp,"\n");
      }

      fclose(fp);      
    }

    printf("Out Twobody energy Pass\n");

  }

  else if (keyword==".threebody_energy"){

    if (iter==MD_IterNumber){
      fprintf(fp,"Twobody energy for each atom\n");
      for (i=1;i<=atomnum;i++){
        fprintf(fp,"Atom %d ",i);
        for (j=1;j<=MD_IterNumber;j++){
          for (k=1;k<=angular_num[i];k++){
            fprintf(fp,"%8.8f ",threebody_ene[i][j][k]);
          }
          fprintf(fp,",");
        }
        fprintf(fp,"\n");
      }

      fclose(fp);      
    }

    printf("Out Threebody energy Pass\n");

  }

  else if (keyword==".ref_energy"){

    if (iter==MD_IterNumber){
      fprintf(fp,"Reference energy for each atom\n");
      for (i=1;i<=atomnum;i++){
        fprintf(fp,"Atom %d ",i);
        for (j=1;j<=MD_IterNumber;j++){
          fprintf(fp,"%8.8f ",Dec_tot[i][j]);
        }
        fprintf(fp,"\n");
      }
      fclose(fp);      
    }

    printf("Out reference decomposed energy Pass\n");

  }

  else if (keyword==".fitted_force"){

    fprintf(fp,"Fitted force at iter = %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d ",i);

      for (j=1;j<=3;j++){
        fprintf(fp,"%8.8f ",model_force[i][j]);
      }

      fprintf(fp,"\n");

    }

    fclose(fp);

    printf("Out fitted force Pass\n");

  }

  else if (keyword==".ref_force"){

    fprintf(fp,"Reference force at iter = %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d ",i);

      for (j=17;j<=19;j++){
        fprintf(fp,"%8.8f ",Gxyz[i][j]);
      }

      fprintf(fp,"\n");
    }

    fclose(fp);

    printf("Out reference force Pass\n");

  }

  else if (keyword==".total_force"){

    fprintf(fp,"Total force for x,y,z\n");

    fprintf(fp,"x: ");
    for (i=1;i<=MD_IterNumber;i++){
      fprintf(fp,"%8.8f ", total_force[1][i]);
    }
    fprintf(fp,"\n");
    fprintf(fp,"y: ");
    for (i=1;i<=MD_IterNumber;i++){
      fprintf(fp,"%8.8f ", total_force[2][i]);
    }
    fprintf(fp,"\n");
    fprintf(fp,"z: ");
    for (i=1;i<=MD_IterNumber;i++){
      fprintf(fp,"%8.8f ", total_force[3][i]);
    }

  }

  else if (keyword==".numerical_force"){

    fprintf(fp,"Numerical force at iter = %d\n",iter);

    for (i=1;i<=atomnum;i++){

      fprintf(fp,"Atom %d ",i);

      for (j=1;j<=3;j++){
        fprintf(fp,"%8.8f ",numerical_force[i][j]);
      }

      fprintf(fp,"\n");
    }

    fclose(fp);

    printf("Out Numerical force Pass\n");

  }

  else{
    printf("Check output keyword \n");
  }

}

/* Generate the matrice for linear solver */

void ML_matrix_gen(int iter, char filepath[YOUSO10], char filename[YOUSO10])
{
  int i,j,k,p,j_1,j_2,k_1,p_1,p_2;
  int count_ang,count_ang1,count_ang2,row,column,column1,column2,parameter_count;
  int species,nei_num;
  double r_cut;

  for (i=1;i<=atomnum;i++){
    //r_cut = 7;
    row = 1;
    species = WhatSpecies[i];
    r_cut = Spe_Atom_Cut1[species];
    nei_num = FNAN[i];

    for (j=1;j<=nei_num;j++){
      for (p=Min_order;p<=Max_order;p++){
        column1 = 1;

        /* Generation of A */

        for (j_1=1;j_1<=nei_num;j_1++){
          for (p_1=Min_order;p_1<=Max_order;p_1++){
            matrix_a[i][row][column1] += 2*lammda1*pow(dis_nei[i][j],p)*pow(dis_nei[i][j_1],p_1)*\
            cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][j_1],r_cut,0);
            if (iter==1){
              if (j==j_1 && p==p_1){
                matrix_a[i][(j-1)*(Max_order-Min_order+1)+p-Min_order+1][(j_1-1)*(Max_order-Min_order+1)+p_1-Min_order+1] += 2*lammda2;
              }                  
            }
            column1 += 1;
          }
        }

        /* Generation of B */

        column2 = 1;
        count_ang = 1;
        for (j_2=1;j_2<=nei_num-1;j_2++){
          for (k=j_2+1;k<=nei_num;k++){
            for (p_2=Min_order;p_2<=Max_order;p_2++){
              matrix_b[i][row][column2] += 2*lammda1*pow(ang_nei[i][count_ang],p_2)*pow(dis_nei[i][j],p)*\
              cut_off(dis_nei[i][j_2],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*cut_off(dis_nei[i][j],r_cut,0);
              column2 ++;
            }
            count_ang ++;
          }
        }

        /* Generation of C */

        matrix_c[i][row] += 2*lammda1*pow(dis_nei[i][j],p)*cut_off(dis_nei[i][j],r_cut,0)*Dec_tot[i][iter];          
        row += 1;
      }
    }
  }

  printf("Generate A B C Pass\n");

  /* Generation of matrix A' B' C' */

  for (i=1;i<=atomnum;i++){
    //r_cut = 7;
    species = WhatSpecies[i];
    r_cut = Spe_Atom_Cut1[species];
    nei_num = FNAN[i];
    row = 1;
    count_ang1 = 1;
    for (j=1;j<=nei_num-1;j++){
      for (k=j+1;k<=nei_num;k++){
        for (p=Min_order;p<=Max_order;p++){

          /* Generation of A' */

          column1 = 1;

          for (j_1=1;j_1<=nei_num;j_1++){
            for (p_1=Min_order;p_1<=Max_order;p_1++){
              matrix_a_[i][row][column1] += 2*lammda1*pow(ang_nei[i][count_ang1],p)*pow(dis_nei[i][j_1],p_1)*\
              cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*cut_off(dis_nei[i][j_1],r_cut,0);
            }
            column1 += 1;
          }

          /* Generation of B' */

          column2 = 1;
          count_ang2 = 1;

          for (j_2=1;j_2<=nei_num-1;j_2++){
            for (k_1=j_2+1;k_1<=nei_num;k_1++){
              for (p_2=Min_order;p_2<=Max_order;p_2++){
                matrix_b_[i][row][column2] += 2*lammda1*pow(ang_nei[i][count_ang1],p)*pow(ang_nei[i][count_ang2],p_2)*\
                cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*cut_off(dis_nei[i][j_2],r_cut,0)*cut_off(dis_nei[i][k_1],r_cut,0);
                if (iter==1){
                  if (j==j_2 && k==k_1 && p==p_2){
                    matrix_b_[i][row][column2] += 2*lammda2;
                  }                    
                }              
                column2 ++;
              }
              count_ang2 ++;
            }
          }            

          /* Generation of C' */ 

          matrix_c_[i][row] += 2*lammda1*pow(ang_nei[i][count_ang1],p)*cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*Dec_tot[i][iter];

          row ++;

        }
        count_ang1 ++;
      }
    }
  }

  printf("Generate A' B' C' Pass\n");

  ML_output(iter,filepath,filename,".matrix");
  
  printf("Out put matrix Pass\n");

  /* Transform matrice to array for solver */

  for (i=1;i<=atomnum;i++){
    nei_num = FNAN[i];
    parameter_count = 1;
    for (j=1;j<=nei_num*(Max_order-Min_order+1);j++){
      for (k=1;k<=nei_num*(Max_order-Min_order+1);k++){
        parameter_matrix[i][parameter_count] = matrix_a[i][j][k];
        parameter_count ++;
      }
      if (atomnum!=2){
        for (p=1;p<=(Max_order-Min_order+1)*angular_num[i];p++){
          parameter_matrix[i][parameter_count] = matrix_b[i][j][p];
          parameter_count ++;
        }        
      }
    }

    if (atomnum!=2){
      for (j=1;j<=(Max_order-Min_order+1)*angular_num[i];j++){
        for (k=1;k<=nei_num*(Max_order-Min_order+1);k++){
          parameter_matrix[i][parameter_count] = matrix_a_[i][j][k];
          parameter_count ++;
        }
        for (p=1;p<=(Max_order-Min_order+1)*angular_num[i];p++){
          parameter_matrix[i][parameter_count] = matrix_b_[i][j][p];
          parameter_count ++;
        }
      }      
    }

  }

  printf("Transform para array Pass\n");

  for (i=1;i<=atomnum;i++){
    nei_num = FNAN[i];
    row = 1;
    for (j=1;j<=nei_num*(Max_order-Min_order+1);j++){
      constant_matrix[i][row] = matrix_c[i][j];
      row ++;
    }

    if (atomnum!=2){
      for (k=1;k<=(Max_order-Min_order+1)*angular_num[i];k++){
        constant_matrix[i][row] = matrix_c_[i][k];
        row ++;
      }      
    }
  }

  printf("Transform const array Pass\n");

  ML_output(iter,filepath,filename,".solver_input");

}

/* Linear solver for fitting */

void ML_DSYSV_solver(int iter, char filepath[YOUSO10], char filename[YOUSO10])
{
  int i,j,nei_num;
  int n,nrhs,lda,ldb,info,lwork;
  double *work;
  int *ipiv;
  char filelast[YOUSO10] = ".solver_info";
  FILE *fp;

  fnjoint(filepath,filename,filelast);
  fp = fopen(filelast,"a");

  /* DSYSV Solver */

  printf("Start Solver\n");

  for (i=1;i<=atomnum;i++){

    nei_num = FNAN[i];

    if (atomnum==2){
      n = nei_num*(Max_order-Min_order+1);
      nrhs = 1;
      lda = nei_num*(Max_order-Min_order+1);
      ldb = nei_num*(Max_order-Min_order+1);
      lwork = nei_num*(Max_order-Min_order+1);
    }

    else{
      n = (nei_num+angular_num[i])*(Max_order-Min_order+1);
      nrhs = 1;
      lda = (nei_num+angular_num[i])*(Max_order-Min_order+1); 
      ldb = (nei_num+angular_num[i])*(Max_order-Min_order+1);
      lwork = (nei_num+angular_num[i])*(Max_order-Min_order+1);      
    }


    /* Allocate the work array */

    work= (double*)malloc(sizeof(double)*lwork);
    memset(work,0,lwork*sizeof(double));

    ipiv = (int*)malloc(sizeof(int)*n);
    memset(ipiv,0,n*sizeof(int));

    /* Call LAPACK solver DSYCV */

    F77_NAME(dsysv,DSYSV)("L", &n, &nrhs, &parameter_matrix[i][1], &lda, ipiv, &constant_matrix[i][1], &ldb, work, &lwork, &info);
 
    /* Free work array */

    free(ipiv);
    ipiv = NULL;

    free(work);
    work = NULL;
    
    if (info!=0) {
      fprintf(fp,"info=%d for atom %d at MD iter %d\n",info,i,iter);
    }
  }

  fclose(fp);

  ML_output(iter,filepath,filename,".fitted_parameter");

  printf("Solver pass at all atom\n");

  if (atomnum==2){
    for (i=1;i<=atomnum;i++){
      nei_num = FNAN[i];
      for (j=1;j<=nei_num*(Max_order-Min_order+1);j++){
        current_model[i][j] = constant_matrix[i][j];
      }
    }
  }

  else{
    for (i=1;i<=atomnum;i++){
      nei_num = FNAN[i];
      for (j=1;j<=(nei_num+angular_num[i])*(Max_order-Min_order+1);j++){
        current_model[i][j] = constant_matrix[i][j];
      }
    }    
  }
  
  printf("Record model pass\n");

}

/* Calculate the model energy and loss respect to current model */

double ML_model_energy(int iter)
{
  int i,j,j_1,k,p,p_1,count_para,count_ang,nei_num,species;
  double r_cut,test,pre_ene;

  /* Rebuild model energy */ 

  for (i=1;i<=atomnum;i++){
    //r_cut = 7;
    nei_num = FNAN[i];
    species = WhatSpecies[i];
    r_cut = Spe_Atom_Cut1[species];
    count_para = 1;

    for (j=1;j<=nei_num;j++){
      for (p=Min_order;p<=Max_order;p++){
        pre_ene = current_model[i][count_para]*cut_off(dis_nei[i][j],r_cut,0)*pow(dis_nei[i][j],p);
        twobody_ene[i][iter][j] += pre_ene;
        fitted_energy[i][iter] += pre_ene;
        count_para ++;
      }
    }
    
    if (atomnum!=2){
      count_ang = 1;
      for (j_1=1;j_1<=nei_num-1;j_1++){
        for (k=j_1+1;k<=nei_num;k++){
          for (p_1=Min_order;p_1<=Max_order;p_1++){
            pre_ene = current_model[i][count_para]*cut_off(dis_nei[i][j_1],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*pow(ang_nei[i][count_ang],p_1);
            threebody_ene[i][iter][count_ang] += pre_ene;
            fitted_energy[i][iter] += pre_ene;
            count_para ++;
          }
          count_ang ++;
        }
      }      
    }

  }

  /* Compute error */

  for (i=1;i<=atomnum;i++){
    loss[i][iter] = fitted_energy[i][iter]-Dec_tot[i][iter];
  }

}

/* Calculate force */

void ML_force(int iter, char filepath[YOUSO10], char filename[YOUSO10])
{

  int i,i_,j,j_,k,k_,p,axis,species;
  int nei_num1,nei_num2,nei_gnum1,nei_gnum2,nei_gnum3,count_para,count_ang,r_cut,nei_num_;
  double test,ref_energy,energy_plus,numerical_precision,dis_shift1,dis_shift2,ang_shift,force_pre;

  numerical_precision = 0.000001;

  /* Initial force array */

  for (i=1;i<=atomnum;i++){
    for (j=1;j<=3;j++){
      model_force[i][j] = 0;
    }
  }

  /* Central contribution */

  for (i=1;i<=atomnum;i++){

    species = WhatSpecies[i];
    r_cut = Spe_Atom_Cut1[species];

    nei_num1 = FNAN[i];
    count_para = 1;

    /* Two-body contribution */

    for (j=1;j<=nei_num1;j++){

      nei_gnum1 = natn[i][j];

      for (p=Min_order;p<=Max_order;p++){

        for (axis=1;axis<=3;axis++){

          model_force[i][axis] -= current_model[i][count_para]*pow(dis_nei[i][j],p-2)*(Gxyz[nei_gnum1][axis]-Gxyz[i][axis])*\
          (p*cut_off(dis_nei[i][j],r_cut,0)-dis_nei[i][j]*cut_off(dis_nei[i][j],r_cut,1));

        }
        count_para ++;
      }

    }

    /* Three-body contribution */
    
    count_ang = 1;
    
    for (j=1;j<=nei_num1-1;j++){

      nei_gnum1 = natn[i][j];

      for (k=j+1;k<=nei_num1;k++){

        nei_gnum2 = natn[i][k];

        for (p=Min_order;p<=Max_order;p++){
          for (axis=1;axis<=3;axis++){

            force_pre = current_model[i][count_para]*p*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*(Gxyz[nei_gnum2][axis]-Gxyz[i][axis])/pow(dis_nei[i][k],2)\
                        +current_model[i][count_para]*p*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*(Gxyz[nei_gnum1][axis]-Gxyz[i][axis])/pow(dis_nei[i][j],2)\
                        +current_model[i][count_para]*p*pow(ang_nei[i][count_ang],p-1)*cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*(2*Gxyz[i][axis]-Gxyz[nei_gnum1][axis]-Gxyz[nei_gnum2][axis])/(dis_nei[i][j]*dis_nei[i][k])\
                        +current_model[i][count_para]*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei[i][j],r_cut,1)*cut_off(dis_nei[i][k],r_cut,0)*(Gxyz[nei_gnum1][axis]-Gxyz[i][axis])/dis_nei[i][j]\
                        +current_model[i][count_para]*pow(ang_nei[i][count_ang],p)*cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,1)*(Gxyz[nei_gnum2][axis]-Gxyz[i][axis])/dis_nei[i][k];
            printf("force pre at centra = %d axis = %d %8.12f\n",i,axis,force_pre);
            model_force[i][axis] -= force_pre;
                      
          }
          count_para ++;
        }
        count_ang ++;
      }
    }
    
  }

  printf("Central atom force pass\n");

  /* Neighbor atom contribution */

  for (i=1;i<=atomnum;i++){

    nei_num1 = FNAN[i];
    species = WhatSpecies[i];
    r_cut = Spe_Atom_Cut1[species];

    /* Two-body contribution */

    for (j=1;j<=nei_num1;j++){
      
      nei_num2 = FNAN[j];
      nei_gnum1 = natn[i][j];
      count_para = 1;

      for (k=1;k<=nei_num2;k++){
        
        nei_gnum2 = natn[nei_gnum1][k];

        if (nei_gnum2==i){
          for (p=Min_order;p<=Max_order;p++){
            for (axis=1;axis<=3;axis++){

              model_force[i][axis] += current_model[nei_gnum1][count_para]*pow(dis_nei[nei_gnum1][k],p-2)*(Gxyz[nei_gnum2][axis]-Gxyz[nei_gnum1][axis])*\
              (p*cut_off(dis_nei[nei_gnum1][k],r_cut,0)-dis_nei[nei_gnum1][k]*cut_off(dis_nei[nei_gnum1][k],r_cut,1));

            }
            count_para ++;
          }
          printf("2body add at central local = %d golbal = %d, neighbor local = %d golbal = %d\n",j,nei_gnum1,k,nei_gnum2);
        }

        else{
          count_para += Max_order-Min_order+1;
        }

      }
    }

    /* Three-body contribution */

    for (j=1;j<=nei_num1;j++){

      nei_num2 = FNAN[j];
      nei_gnum1 = natn[i][j];
      count_ang = 1;
      count_para = nei_num2*(Max_order-Min_order+1)+1;

      for (k=1;k<=nei_num2-1;k++){

        nei_gnum2 = natn[nei_gnum1][k];

        if (nei_gnum2!=i){
          for (k_=k+1;k_<=nei_num2;k_++){

            nei_gnum3 = natn[nei_gnum1][k_];

            if (nei_gnum3==i){
              for (p=Min_order;p<=Max_order;p++){
                for (axis=1;axis<=3;axis++){
                  force_pre = current_model[nei_gnum1][count_para]*pow(ang_nei[nei_gnum1][count_ang],p-1)*p*(Gxyz[nei_gnum2][axis]-Gxyz[nei_gnum1][axis])*cut_off(dis_nei[nei_gnum1][k],r_cut,0)*cut_off(dis_nei[nei_gnum1][k_],r_cut,0)/(dis_nei[nei_gnum1][k]*dis_nei[nei_gnum1][k_])\
                              -current_model[nei_gnum1][count_para]*pow(ang_nei[nei_gnum1][count_ang],p)*p*(Gxyz[nei_gnum3][axis]-Gxyz[nei_gnum1][axis])*cut_off(dis_nei[nei_gnum1][k],r_cut,0)*cut_off(dis_nei[nei_gnum1][k_],r_cut,0)/pow(dis_nei[nei_gnum1][k_],2)\
                              +current_model[nei_gnum1][count_para]*pow(ang_nei[nei_gnum1][count_ang],p)*(Gxyz[nei_gnum3][axis]-Gxyz[nei_gnum1][axis])*cut_off(dis_nei[nei_gnum1][k],r_cut,0)*cut_off(dis_nei[nei_gnum1][k_],r_cut,1)/dis_nei[nei_gnum1][k_];
                  model_force[i][axis] -= force_pre;
                  printf("force pre at nei = %8.12f\n",force_pre);
                }
                count_para ++;
              }
              printf("3body add at central local = %d golbal = %d, 1st neighbor local = %d golbal = %d, 2nd neighbor local = %d golbal = %d\n",j,nei_gnum1,k,nei_gnum2,k_,nei_gnum3);
              count_ang ++;
            }

            else{
              count_ang ++;
              count_para += Max_order-Min_order+1;
            }
          }
        }

        else{
          for (k_=k+1;k_<=nei_num2;k_++){

            nei_gnum3 = natn[nei_gnum1][k_];
            
            for (p=Min_order;p<=Max_order;p++){
              for (axis=1;axis<=3;axis++){
                force_pre = current_model[nei_gnum1][count_para]*pow(ang_nei[nei_gnum1][count_ang],p-1)*p*(Gxyz[nei_gnum3][axis]-Gxyz[nei_gnum1][axis])*cut_off(dis_nei[nei_gnum1][k],r_cut,0)*cut_off(dis_nei[nei_gnum1][k_],r_cut,0)/(dis_nei[nei_gnum1][k]*dis_nei[nei_gnum1][k_])\
                            -current_model[nei_gnum1][count_para]*pow(ang_nei[nei_gnum1][count_ang],p)*p*(Gxyz[nei_gnum2][axis]-Gxyz[nei_gnum1][axis])*cut_off(dis_nei[nei_gnum1][k],r_cut,0)*cut_off(dis_nei[nei_gnum1][k_],r_cut,0)/pow(dis_nei[nei_gnum1][k],2)\
                            +current_model[nei_gnum1][count_para]*pow(ang_nei[nei_gnum1][count_ang],p)*(Gxyz[nei_gnum2][axis]-Gxyz[nei_gnum1][axis])*cut_off(dis_nei[nei_gnum1][k],r_cut,1)*cut_off(dis_nei[nei_gnum1][k_],r_cut,0)/dis_nei[nei_gnum1][k];
                model_force[i][axis] -= force_pre;
                printf("force pre at nei = %8.12f\n",force_pre);
              }
              count_para ++;
            }
            printf("3body add at central local = %d golbal = %d, 1st neighbor local = %d golbal = %d, 2nd neighbor local = %d golbal = %d\n",j,nei_gnum1,k,nei_gnum2,k_,nei_gnum3);
            count_ang ++;
          }
        }
      }
    }

  }

  printf("Neighbor atom force pass\n");

  /* Calculate force error */

  for (i=1;i<=atomnum;i++){
    force_error[i][iter] = pow((pow(model_force[i][1],2)+pow(model_force[i][2],2)+pow(model_force[i][3],2)),0.5)\
    -pow((pow(Gxyz[i][17],2)+pow(Gxyz[i][18],2)+pow(Gxyz[i][19],2)),0.5);
  }

  /* Calculate total force */

  for (axis=1;axis<=3;axis++){
    for (i=1;i<=atomnum;i++){
      total_force[axis][iter] += model_force[i][axis];
    }
  }

  /* Calculate numerical force */

  for (i=1;i<=atomnum;i++){
    for (axis=1;axis<=3;axis++){

      energy_plus = 0;
      ref_energy = 0;
      Gxyz[i][axis] += numerical_precision;

      for (i_=1;i_<=atomnum;i_++){

        count_para = 1;
        nei_num1 = FNAN[i_];
        species = WhatSpecies[i_];
        r_cut = Spe_Atom_Cut1[species];

        for (j=1;j<=nei_num1;j++){

          nei_gnum1 = natn[i_][j];
          dis_shift1 = sqrt(pow((Gxyz[nei_gnum1][1]-Gxyz[i_][1]),2)+pow((Gxyz[nei_gnum1][2]-Gxyz[i_][2]),2)+pow((Gxyz[nei_gnum1][3]-Gxyz[i_][3]),2));

          for (p=Min_order;p<=Max_order;p++){
            energy_plus += current_model[i_][count_para]*cut_off(dis_shift1,r_cut,0)*pow(dis_shift1,p);
            count_para += 1;
          }
        }

        for (j=1;j<=nei_num1;j++){

          nei_gnum1 = natn[i_][j];
          dis_shift1 = sqrt(pow((Gxyz[nei_gnum1][1]-Gxyz[i_][1]),2)+pow((Gxyz[nei_gnum1][2]-Gxyz[i_][2]),2)+pow((Gxyz[nei_gnum1][3]-Gxyz[i_][3]),2));

          for (k=j+1;k<=nei_num1;k++){

            nei_gnum2 = natn[i_][k];
            dis_shift2 = sqrt(pow((Gxyz[nei_gnum2][1]-Gxyz[i_][1]),2)+pow((Gxyz[nei_gnum2][2]-Gxyz[i_][2]),2)+pow((Gxyz[nei_gnum2][3]-Gxyz[i_][3]),2));
            ang_shift = ((Gxyz[nei_gnum1][1]-Gxyz[i_][1])*(Gxyz[nei_gnum2][1]-Gxyz[i_][1])+(Gxyz[nei_gnum1][2]-Gxyz[i_][2])*(Gxyz[nei_gnum2][2]-Gxyz[i_][2])+\
            (Gxyz[nei_gnum1][3]-Gxyz[i_][3])*(Gxyz[nei_gnum2][3]-Gxyz[i_][3]))/(dis_shift1*dis_shift2);

            for (p=Min_order;p<=Max_order;p++){
              energy_plus += current_model[i_][count_para]*cut_off(dis_shift1,r_cut,0)*cut_off(dis_shift2,r_cut,0)*pow(ang_shift,p);
              count_para += 1;
            }
          }
        }
        ref_energy += fitted_energy[i_][iter];
      }
      Gxyz[i][axis] -= numerical_precision;
      numerical_force[i][axis] = -(ref_energy-energy_plus)/numerical_precision;
    }
  }

  /* Output force */

  ML_output(iter,filepath,filename,".fitted_force");
  ML_output(iter,filepath,filename,".ref_force");
  ML_output(iter,filepath,filename,".numerical_force");

  /* Replace the DFT force */
  
  if (ML_force_status==1){
    if (iter>Train_iter && (iter-Train_iter)%Correction_iter!=0){
      for (i=1;i<=atomnum;i++){
        for (axis=17;axis<=19;axis++){
          Gxyz[i][axis] = model_force[i][axis-16];
        }
      }
    }
  }

}

/* Free working array and matrice */

void ML_free_array()
{
  int i,j,nei_num;

  /* Free Matrix A */

  for (i=1;i<=atomnum;i++){
    nei_num = FNAN[i];
    for (j=1;j<=(Max_order-Min_order+1)*nei_num;j++){
      free(matrix_a[i][j]);
    }
  }

  for (i=1;i<=atomnum;i++){
    free(matrix_a[i]);
  }

  free(matrix_a);
  matrix_a = NULL;

  /* Free Matrix B */
  
  for (i=1;i<=atomnum;i++){
    nei_num = FNAN[i];
    for (j=1;j<=(Max_order-Min_order+1)*nei_num;j++){
      free(matrix_b[i][j]);
    }
    free(matrix_b[i]);
  }
  free(matrix_b);

  matrix_b = NULL;

  /* Free Matrix C */
  
  for (i=1;i<=atomnum;i++){
    free(matrix_c[i]);
  }
  free(matrix_c);

  matrix_c = NULL;

  /* Free Matrix A' */
  
  for (i=1;i<=atomnum;i++){
    for (j=1;j<=(Max_order-Min_order+1)*angular_num[i];j++){
      free(matrix_a_[i][j]);
    }
    free(matrix_a_[i]);
  }
  free(matrix_a_);

  matrix_a_ = NULL;

  /* Free Matrix B' */
  
  for (i=1;i<=atomnum;i++){
    for (j=1;j<=(Max_order-Min_order+1)*angular_num[i];j++){
      free(matrix_b_[i][j]);
    }
    free(matrix_b_[i]);
  }
  free(matrix_b_);

  matrix_b_ = NULL;

  /* Free Matrix C' */
  
  for (i=1;i<=atomnum;i++){
    free(matrix_c_[i]);
  }
  free(matrix_c_);

  matrix_c_ = NULL;

  printf("Free matrix pass\n");

  /* Free Para matrix */

  for (i=1;i<=atomnum;i++){
    free(parameter_matrix[i]);
  }
  free(parameter_matrix);

  parameter_matrix = NULL;

  /* Free Const matrix */

  for (i=1;i<=atomnum;i++){
    free(constant_matrix[i]);
  }
  free(constant_matrix);

  constant_matrix = NULL;

  printf("Free para & const array pass\n");

  /* Free Distance & Angular array */

  for (i=1;i<=atomnum;i++){
    free(dis_nei[i]);
  }
  free(dis_nei);

  dis_nei = NULL;

  for (i=1;i<=atomnum;i++){
    free(ang_nei[i]);
  }
  free(ang_nei);

  ang_nei = NULL;

  printf("Free dis & ang pass\n");

  /* Free Decomposed energy,fitted energy,loss,current model array */

  for (i=1;i<=atomnum;i++){
    free(Dec_tot[i]);
  }
  free(Dec_tot);

  Dec_tot = NULL;

  for (i=1;i<=atomnum;i++){
    free(fitted_energy[i]);
  }
  free(fitted_energy);

  fitted_energy = NULL;

  for (i=1;i<=atomnum;i++){
    for (j=1;j<=MD_IterNumber;j++){
      free(twobody_ene[i][j]);
    }
  }

  for (i=1;i<=atomnum;i++){
    free(twobody_ene[i]);
  }

  twobody_ene = NULL;

  for (i=1;i<=atomnum;i++){
    for (j=1;j<=MD_IterNumber;j++){
      free(threebody_ene[i][j]);
    }
  }
  
  for (i=1;i<=atomnum;i++){
    free(threebody_ene[i]);
  }

  threebody_ene = NULL;

  for (i=1;i<=atomnum;i++){
    free(loss[i]);
  }
  free(loss);
  
  loss = NULL;

  for (i=1;i<=atomnum;i++){
    free(force_error[i]);
  }
  free(force_error);
  
  force_error = NULL;

  for (i=1;i<=atomnum;i++){
    free(current_model[i]);
  }
  free(current_model);

  current_model = NULL;

  printf("Free Dec model loss pass\n");

  free(angular_num);
  angular_num = NULL;

  printf("free angular number\n");

  for (i=1;i<=atomnum;i++){
    free(model_force[i]);
  }
  free(model_force);

  model_force = NULL;

  printf("free model force\n");

  for (i=1;i<=3;i++){
    free(total_force[i]);
  }
  free(total_force);

  total_force = NULL;

  printf("free total force\n");

  for (i=1;i<=atomnum;i++){
    free(numerical_force[i]);
  }

  free(numerical_force);

  numerical_force = NULL;

  printf("free numerical force\n");

}

/*******************************************************
                Main function of ML
*******************************************************/

void ML_main(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{

  int myid;

  /* MPI */
  MPI_Status status;
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){

    /* Allocate array */
    if (iter==1){
      ML_allocate();
    }

  }
  
  /* Get decomposed energy */

  Get_decomposed_ene(iter,filepath,filename);

  if (myid==Host_ID){

    if (iter<=Train_iter){

      /* Calculate distance and angular for central atom */

      cal_dis(iter,filepath,filename);
      cal_ang(iter,filepath,filename);

      /* Generate parameter matrix and constant matrix */

      ML_matrix_gen(iter,filepath,filename);

      /* Run linear solver to fit the polynomial */

      ML_DSYSV_solver(iter,filepath,filename);

      /* Compute model energy and error */

      ML_model_energy(iter);

      ML_force(iter,filepath,filename);
      
    }

    else if (iter>Train_iter && (iter-Train_iter)%Correction_iter==0){

      /* Calculate distance and angular for central atom */

      cal_dis(iter,filepath,filename);
      cal_ang(iter,filepath,filename);

      /* Generate parameter matrix and constant matrix */

      ML_matrix_gen(iter,filepath,filename);

      /* Run linear solver to fit the polynomial */

      ML_DSYSV_solver(iter,filepath,filename);

      /* Compute model energy and error */

      ML_model_energy(iter);

      ML_force(iter,filepath,filename);

    }

    else if (iter>Train_iter && (iter-Train_iter)%Correction_iter!=0){

      cal_dis(iter,filepath,filename);
      cal_ang(iter,filepath,filename);

      ML_model_energy(iter);

      ML_force(iter,filepath,filename);

    }

    /* Free array */

    if (iter==MD_IterNumber){

      ML_output(iter,filepath,filename,".fitted_energy");
      ML_output(iter,filepath,filename,".total_force");
      ML_output(iter,filepath,filename,".threebody_energy");
      ML_output(iter,filepath,filename,".twobody_energy");
      ML_output(iter,filepath,filename,".ref_energy");
      ML_output(iter,filepath,filename,".error");
      ML_output(iter,filepath,filename,".force_error");

      ML_free_array();

    }
  }

}