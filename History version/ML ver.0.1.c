/**********************************************************************
  ML.c:

     MLDFT.c is a subroutine to perform ML prediction of atomic force

  Log of ML.c:

     10/Sep/2021  Added by Hengyu Li

***********************************************************************/

#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "openmx_common.h"
#include "lapack_prototypes.h"

#define Pi 3.141592654

float lammda1 = 0.8, lammda2 = 0.001;
int Max_order = 3;

/*******************************************************
 Double dis_nei[][]; 
 Distance of central atom and neighbor
  size: dis_nei[atomnum][len(natn[atomnum])+1]
  allocation: call as cal_dis in iterout.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double dis_nei[10][10];

/*******************************************************
 Double dis_ana[][]; 
 Angular of central atom and neighbor
  size: dis_ana[atomnum][len(natn[atomnum])+1]
  allocation: call as cal_ana in iterout.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double ang_nei[10][20];

/*******************************************************
 Double matrix_a[][]; 
 Parameter matrix of pure two-body terms
  size: matrix_a[n_power*nei_num][n_power*nei_num]
  allocation: call as cal_ana in iterout.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double matrix_a[10][20][20];

/*******************************************************
 Double matrix_c[][]; 
 Parameter matrix of constant of pure two-body terms
  size: matrix_a[n_power*nei_num][n_power*nei_num]
  allocation: call as cal_ana in iterout.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double matrix_c[10][20];

/*******************************************************
 Double matrix_c[][]; 
 Parameter matrix of constant of pure two-body terms
  size: matrix_a[n_power*nei_num][n_power*nei_num]
  allocation: call as cal_ana in iterout.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double matrix_b[10][20][20];

/*******************************************************
 Double matrix_c[][]; 
 Parameter matrix of constant of pure two-body terms
  size: matrix_a[n_power*nei_num][n_power*nei_num]
  allocation: call as cal_ana in iterout.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double matrix_c_[10][20];

/*******************************************************
 Double matrix_c[][]; 
 Parameter matrix of constant of pure two-body terms
  size: matrix_a[n_power*nei_num][n_power*nei_num]
  allocation: call as cal_ana in iterout.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double matrix_a_[10][20][20];

/*******************************************************
 Double matrix_c[][]; 
 Parameter matrix of constant of pure two-body terms
  size: matrix_a[n_power*nei_num][n_power*nei_num]
  allocation: call as cal_ana in iterout.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
double matrix_b_[10][20][20];

/*******************************************************
 Double matrix_c[][]; 
 Parameter matrix of constant of pure two-body terms
  size: matrix_a[n_power*nei_num][n_power*nei_num]
  allocation: call as cal_ana in iterout.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
static double Dec_tot[10];

/*******************************************************
 Double matrix_c[][]; 
 Parameter matrix of constant of pure two-body terms
  size: matrix_a[n_power*nei_num][n_power*nei_num]
  allocation: call as cal_ana in iterout.c
  free:       call as Free_Arrays(0) in openmx.c
*******************************************************/
//static double parameter_matrix[5][920];
// static double test_matrix[5][31][31];
//static double constant_matrix[5][30];
//double model_energy[10];

void cut_off(double,double,int);
void Get_decomposed_ene(int,char,char);
void cal_dis(int,char,char);
void cal_ang(int,char,char);

void output_nei(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{

  int i,j,k,myid;
  char filelast[YOUSO10] = ".test";
  FILE *fp;
    
  fnjoint(filepath,filename,filelast);
  fp = fopen(filelast,"a");

  MPI_Comm_rank(mpi_comm_level1,&myid);
  if (myid!=Host_ID) return;
  
  fprintf(fp,"Test at MD iter =%d\n",iter);
	  
  for (i=1; i<=atomnum; i++){
    fprintf(fp,"Neighbor of atom %d\n",i);
    for (j=1; j<=FNAN[i]; j++){
      k = natn[i][j];
      fprintf(fp,"%s %d %8.6f %8.6f %8.6f\n","Global Num",k,Gxyz[k][1]*BohrR,Gxyz[k][2]*BohrR,Gxyz[k][3]*BohrR);  
    }
  }
  fprintf(fp,"\n");
  fclose(fp);
}

void cal_dis(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{
  
  char filelast[YOUSO10] = ".cal_dis";
  int i,j,k;
  FILE *fp;

  fnjoint(filepath,filename,filelast);
  fp = fopen(filelast,"a");

  //MPI_Comm_rank(mpi_comm_level1,&myid);
  //if (myid!=Host_ID) return;

  fprintf(fp,"Distance at MD iter =%d\n",iter);

  for (i=1; i<=atomnum; i++){
    // fprintf(fp,"%8.6f %8.6f %8.6f",Gxyz[i][1]*BohrR,Gxyz[i][2]*BohrR,Gxyz[i][3]*BohrR);
    for (j=1; j<=FNAN[i]; j++){
      k = natn[i][j];
      dis_nei[i][j] = sqrt((Gxyz[k][1]-Gxyz[i][1])*(Gxyz[k][1]-Gxyz[i][1])+(Gxyz[k][2]-Gxyz[i][2])*(Gxyz[k][2]-Gxyz[i][2])+(Gxyz[k][3]-Gxyz[i][3])*(Gxyz[k][3]-Gxyz[i][3]));
      fprintf(fp,"%8.6f ",dis_nei[i][j]);
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
}

void cal_ang(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{
  char filelast[YOUSO10] = ".cal_ang";
  int i,j,k,nei_gnum1,nei_gnum2,nei_num,count;
  FILE *fp;

  fnjoint(filepath,filename,filelast);
  fp = fopen(filelast,"a");

  //MPI_Comm_rank(mpi_comm_level1,&myid);
  //if (myid!=Host_ID) return;

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
        fprintf(fp,"%8.6f ",ang_nei[i][count]);
        count ++;
      }
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
}

double cut_off(double distance,double r_cut,int grad)
{
  double coefficient;

  if (grad == 0){
    if (distance <= r_cut){
      coefficient = 0.5*(cos(Pi*distance/r_cut)+1);
      return coefficient;
    }
    else{
      return 0;
    }
  }

  else{
    if (distance <= r_cut){
      coefficient = -0.25*Pi*sin(Pi*distance/r_cut);
      return coefficient;
    }
    else{
      return 0;
    }
  }
}

void para_matrix_gen(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{

  char filelast[YOUSO10] = ".matrix",filelast1[YOUSO10] = ".parameter",filelast2[YOUSO10] = ".result",filelast3[YOUSO10] = ".loss",filelast4[YOUSO10] = ".info";
  char filelast5[YOUSO10] = ".test";
  int i,j,k,p,j_1,j_2,k_1,p_1,p_2,nei_num,count_ang,count_ang1,count_ang2,row,row_mpi,column,column_mpi,species,myid,ID,tag=999;
  int parameter_count,constant_count,proc_count,numprocs,matrix_count,test,count_para;
  double r_cut,loss,energy_test;
  static double model_energy[10];
  static double constant_matrix[6][35];
  static double parameter_matrix[6][920];

  /* Lapack varibles */
  int n, nrhs, lda, ldb, info, lwork;
  double wkopt;
  double *work;
  int *ipiv;

  /* Output file */
  FILE *fp;
  FILE *fp1;
  FILE *fp2;
  FILE *fp3;
  FILE *fp4;
  FILE *fp5;

  /* MPI */
  MPI_Status status;

  fnjoint(filepath,filename,filelast);
  fnjoint(filepath,filename,filelast1);
  fnjoint(filepath,filename,filelast2);
  fnjoint(filepath,filename,filelast3);
  fnjoint(filepath,filename,filelast4);
  fnjoint(filepath,filename,filelast5);
  fp = fopen(filelast,"a");
  fp1 = fopen(filelast1,"a");
  fp2 = fopen(filelast2,"a");
  fp3 = fopen(filelast3,"a");
  fp4 = fopen(filelast4,"a");
  fp5 = fopen(filelast5,"a");

  /* MPI preparation */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* Preparae the decomposed energy */

  Get_decomposed_ene(iter,filepath,filename);

  if (myid==Host_ID){

    fprintf(fp,"Matrix_a at MD iter =%d\n",iter);

    /* Preparae the angular,distance */ 

    cal_dis(iter,filepath,filename);
    cal_ang(iter,filepath,filename);

    /* Generation of matrix A B C */

    for (i=1;i<=atomnum;i++){
      species = WhatSpecies[i];
      r_cut = Spe_Atom_Cut1[species];
      nei_num = FNAN[i];
      for (j=1;j<=nei_num;j++){
        for (p=1;p<=Max_order;p++){
          /* Generation of A */
          for (j_1=1;j_1<=nei_num;j_1++){
            for (p_1=1;p_1<=Max_order;p_1++){
              matrix_a[i][(j-1)*Max_order+p][(j_1-1)*Max_order+p_1] += 2*lammda1*pow(dis_nei[i][j],p)*pow(dis_nei[i][j_1],p_1)*\
              cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][j_1],r_cut,0); 
              if (j==j_1 && p==p_1){
                matrix_a[i][(j-1)*Max_order+p][(j_1-1)*Max_order+p_1] += 2*lammda2;
              }
            }
          }
          /* Generation of B */
          column = 1;
          count_ang = 1;
          for (j_2=1;j_2<=nei_num-1;j_2++){
            for (k=j_2+1;k<=nei_num;k++){
              for (p_2=1;p_2<=Max_order;p_2++){ //p = 1 or = 0 
                matrix_b[i][(j-1)*Max_order+p][column] += 2*lammda1*pow(ang_nei[i][count_ang],p_2)*pow(dis_nei[i][j],p)*\
                cut_off(dis_nei[i][j_2],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*cut_off(dis_nei[i][j],r_cut,0);
                column ++;
              }
              count_ang ++;
            }
          }
          /* Generation of C */
          matrix_c[i][(j-1)*Max_order+p] += 2*lammda1*pow(dis_nei[i][j],p)*cut_off(dis_nei[i][j],r_cut,0)*Dec_tot[i];
        }
      }
    }

    /* Generation of matrix A' B' C' */

    for (i=1;i<=atomnum;i++){
      species = WhatSpecies[i];
      r_cut = Spe_Atom_Cut1[species];
      nei_num = FNAN[i];
      row = 1;
      count_ang1 = 1;
      for (j=1;j<=nei_num-1;j++){
        for (k=j+1;k<=nei_num;k++){
          for (p=1;p<=Max_order;p++){
            /* Generation of A' */
            for (j_1=1;j_1<=nei_num;j_1++){
              for (p_1=1;p_1<=Max_order;p_1++){
                matrix_a_[i][row][(j_1-1)*Max_order+p_1] += 2*lammda1*pow(ang_nei[i][count_ang1],p)*pow(dis_nei[i][j_1],p_1)*\
                cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*cut_off(dis_nei[i][j_1],r_cut,0);
              }
            }
            /* Generation of B' */
            column = 1;
            count_ang2 = 1;
            for (j_2=1;j_2<=nei_num-1;j_2++){
              for (k_1=j_2+1;k_1<=nei_num;k_1++){
                for (p_2=1;p_2<=Max_order;p_2++){
                  matrix_b_[i][row][column] += 2*lammda1*pow(ang_nei[i][count_ang1],p)*pow(ang_nei[i][count_ang2],p_2)*\
                  cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*cut_off(dis_nei[i][j_2],r_cut,0)*cut_off(dis_nei[i][k_1],r_cut,0);
                  if (j==j_2 && k==k_1 && p==p_2){
                    matrix_b_[i][row][column] += 2*lammda2;
                  }
                  column ++;
                }
                count_ang2 ++;
              }
            }
            /* Generation of C' */ 
            matrix_c_[i][row] += 2*lammda1*pow(ang_nei[i][count_ang1],p)*cut_off(dis_nei[i][j],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*Dec_tot[i];
            row ++;
          }
          count_ang1 ++;
        }
      }
    }

    /* Output matrix A */

    fprintf(fp,"Matrix A\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix A for atom %d\n", i);
      for (j=1; j<=3*FNAN[i]; j++){
        for (k=1; k<=3*FNAN[i]; k++){
          fprintf(fp,"%8.6f ",matrix_a[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }
    fprintf(fp,"\n");

    /* Output matrix C */

    fprintf(fp,"Matrix C\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix C for atom %d\n", i);
      for (j=1; j<=3*FNAN[i]; j++){
        fprintf(fp,"%8.6f ",matrix_c[i][j]);
      }
      fprintf(fp,"\n");
    }
    fprintf(fp,"\n");

    /* Output matrix B */

    fprintf(fp,"Matrix B\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix B for atom %d\n", i);
      for (j=1; j<=3*FNAN[i]; j++){
        for (k=1; k<=18; k++){
          fprintf(fp,"%8.6f ",matrix_b[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }
    fprintf(fp,"\n");

    /* Output matrix A' */

    fprintf(fp,"Matrix A'\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix A' for atom %d\n", i);
      for (j=1; j<=18; j++){
        for (k=1; k<=3*FNAN[i]; k++){
          fprintf(fp,"%8.6f ",matrix_a_[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }
    fprintf(fp,"\n");

    /* Output matrix C' */

    fprintf(fp,"Matrix C'\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix C' for atom %d\n", i);
      for (j=1; j<=18; j++){
        fprintf(fp,"%8.6f ",matrix_c_[i][j]);
      }
      fprintf(fp,"\n");
    }
    fprintf(fp,"\n");

    /* Output matrix B' */

    fprintf(fp,"Matrix B'\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix B' for atom %d\n", i);
      for (j=1; j<=18; j++){
        for (k=1; k<=18; k++){
          fprintf(fp,"%8.6f ",matrix_b_[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }

    fclose(fp);

    for (i=1;i<=atomnum;i++){
      nei_num = FNAN[i];
      parameter_count = 1;
      for (j=1;j<=nei_num*Max_order;j++){
        for (k=1;k<=nei_num*Max_order;k++){
          parameter_matrix[i][parameter_count] = matrix_a[i][j][k];
          parameter_count ++;
        }
        for (p=1;p<=18;p++){
          parameter_matrix[i][parameter_count] = matrix_b[i][j][p];
          parameter_count ++;
        }
      }

      for (j=1;j<=18;j++){
        for (k=1;k<=nei_num*Max_order;k++){
          parameter_matrix[i][parameter_count] = matrix_a_[i][j][k];
          parameter_count ++;
        }
        for (p=1;p<=18;p++){
          parameter_matrix[i][parameter_count] = matrix_b_[i][j][p];
          parameter_count ++;
        }
      }
    }

    for (i=1;i<=atomnum;i++){
      nei_num = FNAN[i];
      row = 1;
      for (j=1;j<=nei_num*Max_order;j++){
        constant_matrix[i][row] = -matrix_c[i][j];
        row ++;
      }
      for (k=1;k<=30;k++){
        constant_matrix[i][row] = -matrix_c_[i][k];
        row ++;
      }
    }

    for (i=1;i<=atomnum;i++){
      fprintf(fp1,"Parameter matrix for atom %d\n",i);
      for (j=1;j<=900;j++){
        fprintf(fp1,"%8.6f ",parameter_matrix[i][j]);
      }
      fprintf(fp1,"\n");
      fprintf(fp1,"Constant matrix for atom %d\n",i);
      for (test=1;test<=30;test++){
        fprintf(fp1,"%8.8f ",constant_matrix[i][test]);
      }
      fprintf(fp1,"\n");
    }

    fclose(fp1);

    n = 30;
    nrhs = 1;
    lda = 30;
    ldb = 30;
    lwork = 30;
    work= (double*)malloc(sizeof(double)*lwork);
    ipiv = (int*)malloc(sizeof(int)*(n));

    fprintf(fp2,"Result for MD %d\n",iter);
    //printf("Test In\n");
    for (i=1;i<=atomnum;i++){
      //printf("%8.6f\n",Dec_tot[i]);
      F77_NAME(dsysv,DSYSV)("L", &n, &nrhs, &parameter_matrix[i][0], &lda, ipiv, &constant_matrix[i][0], &ldb, work, &lwork, &info); // Report [1,1] is singular need fix (Fixed)
      //printf("%8.6f\n",Dec_tot[i]);
      if (info!=0) {
        fprintf(fp4,"info=%d for atom %d at MD iter %d\n",i,info,iter);
      }
      fprintf(fp2,"atom %d\n",i);
      for (test=1;test<=30;test++){
        fprintf(fp2,"%8.8f ",constant_matrix[i][test]);
      }
      fprintf(fp2,"\n");
    }
    fclose(fp4);
    fclose(fp2);

    /* Rebuild the decomposed energy and compute loss */

    printf("Test\n");

    for (i=1;i<=atomnum;i++){
      nei_num = FNAN[i];
      species = WhatSpecies[i];
      r_cut = Spe_Atom_Cut1[species];
      count_para = 1;

      for (j=1;j<=nei_num;j++){
        for (p=1;p<=Max_order;p++){
          energy_test = constant_matrix[i][count_para]*cut_off(dis_nei[i][j],r_cut,0)*pow(dis_nei[i][j],p);
          model_energy[i] += energy_test;
          count_para ++;
          printf("%8.6f ",energy_test);
        }
      }
      count_ang = 1;
      for (j_1=1;j_1<=nei_num-1;j_1++){
        for (k=j_1+1;k<=nei_num;k++){
          for (p_1=1;p_1<=Max_order;p_1++){
            energy_test = constant_matrix[i][count_para]*cut_off(dis_nei[i][j_1],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*pow(ang_nei[i][count_ang],p_1);
            model_energy[i] += energy_test;
            count_para ++;
            printf("%8.6f ",energy_test);
          }
          count_ang ++;
        }
      }
      printf("\n");
      printf("\n");
    }

    fprintf(fp3,"Model energy and loss\n");
    fprintf(fp3,"Model energy\n");

    for (i=1;i<=atomnum;i++){
      fprintf(fp3,"Atom %d %8.6f\n",i,model_energy[i]);
    }
    
    fprintf(fp3,"loss\n");
    for (i=1;i<=atomnum;i++){
      loss = model_energy[i]-Dec_tot[i];
      fprintf(fp3,"Atom %d = %8.6f\n",i,loss);
    }
    fclose(fp3);
  }
}

/* Get decomposed energy repect to atom on each processor */

void Get_decomposed_ene(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{
  char filelast[YOUSO10] = ".mpi";
  int i, j, spin, local_num, global_num, ID, numprocs, species,tag=999,myid;
  double energy;
  FILE *fp;
  MPI_Status status;

  fnjoint(filepath,filename,filelast);
  fp = fopen(filelast,"a");

  /* MPI */
  MPI_Comm_size(mpi_comm_level1,&numprocs);
  MPI_Comm_rank(mpi_comm_level1,&myid);

  if (myid==Host_ID){
    fprintf(fp,"Decomposed energy respect to atom at MD = %d\n",iter);
  }

  for (global_num=1; global_num<=atomnum; global_num++){
    species = WhatSpecies[global_num];
    ID = G2ID[global_num];
    if (myid==ID){
      local_num = F_G2M[global_num];
      energy = 0;
      if (SpinP_switch==0){
        for (i=0;i<Spe_Total_CNO[species];i++){
          energy += 2*DecEkin[0][local_num][i]; // Dec_tot[global_num] += DecEkin[0][local_num][i];
          energy += 2*DecEv[0][local_num][i]; // Dec_tot[global_num] += DecEv[0][local_num][i];
          energy += 2*DecEcon[0][local_num][i]; // Dec_tot[global_num] += DecEcon[0][local_num][i];
          energy += 2*DecEscc[0][local_num][i]; // Dec_tot[global_num] += DecEscc[0][local_num][i];
          energy += 2*DecEvdw[0][local_num][i]; // Dec_tot[global_num] += DecEvdw[0][local_num][i];
        }
      }
      if (SpinP_switch==1 || SpinP_switch==3){
        for (i=0;i<Spe_Total_CNO[species];i++){
          for (spin=0;spin<=1;spin++){
            energy += DecEkin[spin][local_num][i]; //Dec_tot[global_num] += DecEkin[spin][local_num][i];
            energy += DecEv[spin][local_num][i]; //Dec_tot[global_num] += DecEv[spin][local_num][i];
            energy += DecEcon[spin][local_num][i]; //Dec_tot[global_num] += DecEcon[spin][local_num][i];
            energy += DecEscc[spin][local_num][i]; //Dec_tot[global_num] += DecEscc[spin][local_num][i];
            energy += DecEvdw[spin][local_num][i]; //Dec_tot[global_num] += DecEvdw[spin][local_num][i];
          }
        }
      }
      if (myid!=Host_ID){
        MPI_Send(&energy,1,MPI_DOUBLE,Host_ID,tag,mpi_comm_level1);
      }
    }

    if (myid==Host_ID){
      if (ID!=Host_ID){
        MPI_Recv(&energy, 1, MPI_DOUBLE, ID, tag, mpi_comm_level1, &status);
      }
      Dec_tot[global_num] = energy;
      fprintf(fp,"%s ",SpeName[species]);
      fprintf(fp,"%8.6f\n", Dec_tot[global_num]);
    }
  }
  fclose(fp);
}