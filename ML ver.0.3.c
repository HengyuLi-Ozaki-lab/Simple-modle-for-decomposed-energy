/**********************************************************************
  ML.c:

     ML.c is a subroutine to perform ML prediction of atomic force

  Log of ML.c:

     10/Sep/2021  Added by Hengyu Li

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
     Hyperparameter for fitting decomposed energy
*******************************************************/

float lammda1 = 0.8, lammda2 = 0.001;
int Max_order = 3;

/*******************************************************
    Arry for distance,angular,matrix A B C A' B' C'
*******************************************************/

static double *Dec_tot;
static double **dis_nei;
static double **ang_nei;

/*******************************************************/

int factorial(int m, int n)
{
	int ans = 1;
	if(m < n-m) m = n-m;
	for(int i = m+1; i <= n; i++) ans *= i;
	for(int j = 1; j <= n - m; j++) ans /= j;
	return ans;
}

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
  static int i,j,k,p,j_1,j_2,k_1,p_1,p_2,nei_num,count_ang,count_ang1,count_ang2,row,column,species,myid,ID;
  static int parameter_count,constant_count,matrix_count,test,count_para,matrix_test;
  double r_cut,loss,energy_test;

  static double ***matrix_a;
  static double ***matrix_b;
  static double ***matrix_a_;
  static double ***matrix_b_;
  static double **matrix_c;
  static double **matrix_c_;
  static double *model_energy;
  static int *angular_num;
  static double **constant_matrix;
  static double **parameter_matrix;
  //static double parameter_matrix[6][901];
  //static int angular_num[6]; //= {6,6,6,6,6,6};

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

  /* MPI */
  MPI_Status status;

  fnjoint(filepath,filename,filelast);
  fnjoint(filepath,filename,filelast1);
  fnjoint(filepath,filename,filelast2);
  fnjoint(filepath,filename,filelast3);
  fnjoint(filepath,filename,filelast4);
  fp = fopen(filelast,"a");
  fp1 = fopen(filelast1,"a");
  fp2 = fopen(filelast2,"a");
  fp3 = fopen(filelast3,"a");
  fp4 = fopen(filelast4,"a");

  /* MPI preparation */
  MPI_Comm_rank(mpi_comm_level1,&myid);

  /* Allocate array */

  if (myid==Host_ID){

    printf("Start\n");

    
    if (iter==1){
      angular_num = (int*)malloc(sizeof(int)*(atomnum+1));
      memset(angular_num,0,(atomnum+1)*sizeof(int));      
    }

    for (i=1;i<=atomnum;i++){
      angular_num[i] = factorial(2,FNAN[i]);
      printf("%d\n",angular_num[i]);
    }
    

    dis_nei = (double**)malloc(sizeof(double*)*(atomnum+1));
    for(j=1;j<=atomnum;j++){
      dis_nei[j] = (double*)malloc(sizeof(double)*(FNAN[j]+1));
      memset(dis_nei[j],0,(FNAN[j]+1)*sizeof(double));
    }

    ang_nei = (double**)malloc(sizeof(double*)*atomnum+1);
    for(j=1;j<=atomnum;j++){
      ang_nei[j] = (double*)malloc(sizeof(double)*(angular_num[j]+1));
      memset(ang_nei[j],0,(angular_num[j]+1)*sizeof(double));
    }

    if (iter==1){

      matrix_a = (double***)malloc(sizeof(double**)*(atomnum+1));
      for (i=1; i<=atomnum; i++){
        matrix_a[i] = (double**)malloc(sizeof(double*)*(FNAN[i]*Max_order+1)); 
        for (j=1; j<=(FNAN[i]*Max_order); j++){
          matrix_a[i][j] = (double*)malloc(sizeof(double)*(FNAN[i]*Max_order+1)); 
          memset(matrix_a[i][j],0,(FNAN[i]*Max_order+1)*sizeof(double));
        }
      }

      matrix_b = (double***)malloc(sizeof(double**)*(atomnum+1));
      for (i=1; i<=atomnum; i++){
        matrix_b[i] = (double**)malloc(sizeof(double*)*(FNAN[i]*Max_order+1)); 
        for (j=1; j<=(FNAN[i]*Max_order); j++){
          matrix_b[i][j] = (double*)malloc(sizeof(double)*(Max_order*angular_num[i]+1));
          memset(matrix_b[i][j],0,(Max_order*angular_num[i]+1)*sizeof(double));
        }
      }

      matrix_c = (double**)malloc(sizeof(double*)*(atomnum+1));
      for(j=1;j<=atomnum;j++){
        matrix_c[j] = (double*)malloc(sizeof(double)*(FNAN[j]*Max_order+1));
        memset(matrix_c[j],0,(FNAN[j]*Max_order+1)*sizeof(double));
      }

      matrix_a_ = (double***)malloc(sizeof(double**)*(atomnum+1));
      for (i=1; i<=atomnum; i++){
        matrix_a_[i] = (double**)malloc(sizeof(double*)*(Max_order*angular_num[i]+1)); 
        for (j=1; j<=(Max_order*angular_num[i]); j++){
          matrix_a_[i][j] = (double*)malloc(sizeof(double)*(FNAN[i]*Max_order+1));
          memset(matrix_a_[i][j],0,(FNAN[i]*Max_order+1)*sizeof(double));
        }
      }

      matrix_b_ = (double***)malloc(sizeof(double**)*(atomnum+1));
      for (i=1; i<=atomnum; i++){
        matrix_b_[i] = (double**)malloc(sizeof(double*)*(Max_order*angular_num[i]+1)); 
        for (j=1; j<=(Max_order*angular_num[i]); j++){
          matrix_b_[i][j] = (double*)malloc(sizeof(double)*(Max_order*angular_num[i]+1));
          memset(matrix_b_[i][j],0,(Max_order*angular_num[i]+1)*sizeof(double)); 
        }
      }

      matrix_c_ = (double**)malloc(sizeof(double*)*(atomnum+1));
      for(j=1;j<=atomnum;j++){
        matrix_c_[j] = (double*)malloc(sizeof(double)*(Max_order*angular_num[j]+1));
        memset(matrix_c_[j],0,(Max_order*angular_num[j]+1)*sizeof(double)); 
      }

      Dec_tot = (double*)malloc(sizeof(double)*(atomnum+1));
      memset(Dec_tot,0,(atomnum+1)*sizeof(double));

      model_energy = (double*)malloc(sizeof(double)*(atomnum+1));
      memset(model_energy,0,(atomnum+1)*sizeof(double));

      printf("Matrix Pass\n");

      parameter_matrix = (double**)malloc(sizeof(double*)*(atomnum+1));
      for(j=1;j<=atomnum;j++){
        nei_num = FNAN[j];
        parameter_matrix[j] = (double*)malloc(sizeof(double)*(pow((nei_num*Max_order+Max_order*angular_num[j]),2)+1)); // (pow((nei_num*Max_order+Max_order*angular_num[j]),2)+1)
        memset(parameter_matrix[j],0,(pow((nei_num*Max_order+Max_order*angular_num[j]),2)+1)*sizeof(double));
      }

      constant_matrix = (double**)malloc(sizeof(double*)*(atomnum+1));
      for(j=1;j<=atomnum;j++){
        nei_num = FNAN[j];
        constant_matrix[j] = (double*)malloc(sizeof(double)*(nei_num*Max_order+Max_order*angular_num[j]+1));
        memset(constant_matrix[j],0,(nei_num*Max_order+Max_order*angular_num[j]+1)*sizeof(double));
      }

      printf("Para Const Pass\n");

    }

  }
  
  /* Preparae the decomposed energy */

  Get_decomposed_ene(iter,filepath,filename);

  printf("Get decomposed_ene Pass\n");

  if (myid==Host_ID){

    /* Preparae the angular,distance */ 

    cal_dis(iter,filepath,filename);
    cal_ang(iter,filepath,filename);

    printf("Dis & Ang Pass\n");

    fprintf(fp,"Matrix_a at MD iter =%d\n",iter);

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

    printf("A B C Pass\n");

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

    printf("A' B' C' Pass\n");

    /* Output matrix A */

    fprintf(fp,"Matrix A\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix A for atom %d\n", i);
      for (j=1; j<=Max_order*FNAN[i]; j++){
        for (k=1; k<=Max_order*FNAN[i]; k++){
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
      for (j=1; j<=Max_order*FNAN[i]; j++){
        fprintf(fp,"%8.6f ",matrix_c[i][j]);
      }
      fprintf(fp,"\n");
    }
    fprintf(fp,"\n");

    /* Output matrix B */

    fprintf(fp,"Matrix B\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix B for atom %d\n", i);
      for (j=1; j<=Max_order*FNAN[i]; j++){
        for (k=1; k<=Max_order*angular_num[i]; k++){
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
      for (j=1; j<=Max_order*angular_num[i]; j++){
        for (k=1; k<=Max_order*FNAN[i]; k++){
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
      for (j=1; j<=Max_order*angular_num[i]; j++){
        fprintf(fp,"%8.6f ",matrix_c_[i][j]);
      }
      fprintf(fp,"\n");
    }
    fprintf(fp,"\n");

    /* Output matrix B' */

    fprintf(fp,"Matrix B'\n");
    for (i=1; i<=atomnum; i++){
      fprintf(fp,"Matrix B' for atom %d\n", i);
      for (j=1; j<=Max_order*angular_num[i]; j++){
        for (k=1; k<=Max_order*angular_num[i]; k++){
          fprintf(fp,"%8.6f ",matrix_b_[i][j][k]);
        }
        fprintf(fp,"\n");
      }
    }

    fclose(fp);

    printf("Out matrix Pass\n");
    
    for (i=1;i<=atomnum;i++){
      nei_num = FNAN[i];
      parameter_count = 1;
      for (j=1;j<=nei_num*Max_order;j++){
        for (k=1;k<=nei_num*Max_order;k++){
          parameter_matrix[i][parameter_count] = matrix_a[i][j][k];
          parameter_count ++;
        }
        for (p=1;p<=Max_order*angular_num[i];p++){
          parameter_matrix[i][parameter_count] = matrix_b[i][j][p];
          parameter_count ++;
        }
      }

      for (j=1;j<=Max_order*angular_num[i];j++){
        for (k=1;k<=nei_num*Max_order;k++){
          parameter_matrix[i][parameter_count] = matrix_a_[i][j][k];
          parameter_count ++;
        }
        for (p=1;p<=Max_order*angular_num[i];p++){
          parameter_matrix[i][parameter_count] = matrix_b_[i][j][p];
          parameter_count ++;
        }
      }
    }

    printf("Tran para matrix Pass\n");

    for (i=1;i<=atomnum;i++){
      nei_num = FNAN[i];
      row = 1;
      for (j=1;j<=nei_num*Max_order;j++){ //nei_num*Max_order
        constant_matrix[i][row] = -matrix_c[i][j];
        row ++;
      }
      for (k=1;k<=Max_order*angular_num[i];k++){ //Max_order*angular_num[i]
        constant_matrix[i][row] = -matrix_c_[i][k];
        row ++;
      }
    }

    printf("Tran matrix Pass\n");

    /* Free Matrix A B C A' B' C' after last MD iter */

    if (iter==MD_IterNumber){

      /* Free Matrix A */

      for (i=1;i<=atomnum;i++){
        nei_num = FNAN[i];
        //printf("i = %d\n",i);
        for (j=1;j<=Max_order*nei_num;j++){
          free(matrix_a[i][j]);
          //printf("%d j pass\n",j);
        }
      }

      for (i=1;i<=atomnum;i++){
        free(matrix_a[i]);
        //printf("%d i pass\n",i);
      }

      //printf("a1 Pass\n");

      free(matrix_a);
      //printf("a2 Pass\n");
      matrix_a = NULL;

      //printf("a3 Pass\n");

      /* Free Matrix B */
      
      for (i=1;i<=atomnum;i++){
        nei_num = FNAN[i];
        for (j=1;j<=Max_order*nei_num;j++){
          free(matrix_b[i][j]);
        }
        free(matrix_b[i]);
      }
      free(matrix_b);

      printf("b Pass\n");

      matrix_b = NULL;

      /* Free Matrix C */
      
      for (i=1;i<=atomnum;i++){
        free(matrix_c[i]);
      }
      free(matrix_c);

      matrix_c = NULL;

      printf("c Pass\n");

      /* Free Matrix A' */
      
      for (i=1;i<=atomnum;i++){
        printf("i = %d\n",i);
        for (j=1;j<=Max_order*angular_num[i];j++){
          printf("j = %d\n",j);
          printf("%8.6f\n",matrix_a_[i][j][1]);
          free(matrix_a_[i][j]);
          printf("%d j pass\n",j);
        }
        free(matrix_a_[i]);
        printf("%d i pass\n",i);
      }
      free(matrix_a_);
      printf("1 pass\n");

      matrix_a_ = NULL;

      printf("a' Pass\n");

      /* Free Matrix B' */
      
      for (i=1;i<=atomnum;i++){
        for (j=1;j<=Max_order*angular_num[i];j++){
          free(matrix_b_[i][j]);
        }
        free(matrix_b_[i]);
      }
      free(matrix_b_);

      matrix_b_ = NULL;

      printf("b' Pass\n");

      /* Free Matrix C' */
      
      for (i=1;i<=atomnum;i++){
        free(matrix_c_[i]);
      }
      free(matrix_c_);

      matrix_c_ = NULL;

      printf("c' Pass\n");

      printf("Free matrix pass\n");

    }

    for (i=1;i<=atomnum;i++){
      nei_num = FNAN[i];
      fprintf(fp1,"Parameter matrix for atom %d\n",i);
      for (j=1;j<=pow((nei_num*Max_order+Max_order*angular_num[i]),2);j++){
        fprintf(fp1,"%8.6f ",parameter_matrix[i][j]); 
      }
      fprintf(fp1,"\n");
      fprintf(fp1,"Constant matrix for atom %d\n",i);
      for (test=1;test<=(nei_num*Max_order+Max_order*angular_num[i]);test++){ 
        fprintf(fp1,"%8.8f ",constant_matrix[i][test]);
      }
      fprintf(fp1,"\n");
    }

    fclose(fp1);

    n = 30; // nei_num*Max_order+Max_order*angular_num[i]; This part should be included in for i< atomnum, nei_num should change with i
    nrhs = 1;
    lda = 30; // nei_num*Max_order+Max_order*angular_num[i];
    ldb = 30; //nei_num*Max_order+Max_order*angular_num[i];
    lwork = 31; // nei_num*Max_order+Max_order*angular_num[i];

    work= (double*)malloc(sizeof(double)*lwork);
    memset(work,0,lwork*sizeof(double));

    ipiv = (int*)malloc(sizeof(int)*n);
    memset(ipiv,0,n*sizeof(int));

    printf("Start Solver\n");

    fprintf(fp2,"Result for MD %d\n",iter);

    for (i=1;i<=atomnum;i++){
      nei_num = FNAN[i];
      F77_NAME(dsysv,DSYSV)("L", &n, &nrhs, &parameter_matrix[i][0], &lda, ipiv, &constant_matrix[i][0], &ldb, work, &lwork, &info); // Report [1,1] is singular need fix (Fixed)
      if (info!=0) {
        fprintf(fp4,"info=%d for atom %d at MD iter %d\n",i,info,iter);
      }
      fprintf(fp2,"atom %d\n",i);
      for (test=1;test<=(nei_num*Max_order+Max_order*angular_num[i]);test++){
        fprintf(fp2,"%8.8f ",constant_matrix[i][test]);
      }
      fprintf(fp2,"\n");
    }

    fclose(fp4);
    fclose(fp2);


    free(ipiv);
    ipiv = NULL;

    free(work);
    work = NULL;

    printf("Solver Pass\n");


    if (iter==MD_IterNumber){
      
      for (i=1;i<=atomnum;i++){
        free(parameter_matrix[i]);
      }
      free(parameter_matrix);

      parameter_matrix = NULL;

      printf("Free test 2\n");
    }
    
    /* Rebuild the decomposed energy and compute loss */

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
          // printf("%8.6f ",energy_test);
        }
      }

      count_ang = 1;
      for (j_1=1;j_1<=nei_num-1;j_1++){
        for (k=j_1+1;k<=nei_num;k++){
          for (p_1=1;p_1<=Max_order;p_1++){
            energy_test = constant_matrix[i][count_para]*cut_off(dis_nei[i][j_1],r_cut,0)*cut_off(dis_nei[i][k],r_cut,0)*pow(ang_nei[i][count_ang],p_1);
            model_energy[i] += energy_test;
            count_para ++;
            // printf("%8.6f ",energy_test);
          }
          count_ang ++;
        }
      }
      // printf("\n");
    }

    if (iter==MD_IterNumber){
      for (i=1;i<=atomnum;i++){
        free(constant_matrix[i]);
      }
      free(constant_matrix);

      constant_matrix = NULL;
    }

    printf("Rebuild Pass\n");

    fprintf(fp3,"Model energy and loss\n");
    fprintf(fp3,"Model energy\n");

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

    printf("Free test 1\n");

    for (i=1;i<=atomnum;i++){
      fprintf(fp3,"Atom %d %8.6f\n",i,model_energy[i]);
    }

    printf("Sub test 1\n");

    fprintf(fp3,"loss\n");
    for (i=1;i<=atomnum;i++){
      loss = model_energy[i]-Dec_tot[i];
      fprintf(fp3,"Atom %d = %8.6f\n",i,loss);
    }
    fclose(fp3);

    printf("Sub test 2\n");

    if (iter==MD_IterNumber){

      free(Dec_tot);
      Dec_tot = NULL;

      free(model_energy);
      model_energy = NULL;

      printf("Free Dec model pass\n");

      /*
      free(angular_num);
      angular_num = NULL;
      printf("free ang\n");
      */
    }
  }
}

/* Get decomposed energy repect to atom on each processor */

void Get_decomposed_ene(int iter,char filepath[YOUSO10],char filename[YOUSO10])
{
  char filelast[YOUSO10] = ".mpi";
  int i, j, spin, local_num, global_num, ID, species,tag=999,myid;
  double energy;
  FILE *fp;
  MPI_Status status;

  fnjoint(filepath,filename,filelast);
  fp = fopen(filelast,"a");

  /* MPI */
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
          energy += 2*DecEkin[0][local_num][i]; 
          energy += 2*DecEv[0][local_num][i];
          energy += 2*DecEcon[0][local_num][i];
          energy += 2*DecEscc[0][local_num][i];
          energy += 2*DecEvdw[0][local_num][i];
        }
      }
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
