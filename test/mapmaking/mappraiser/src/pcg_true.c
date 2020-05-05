// Midapack library
// mapmaking code example using the Midapack library - release 1.2b, Nov 2012
// PCG routine applied to the mapmaking equation
// This can use the diagonal or the block-diagonal jacobi preconditionners

/** @file   pcg_true.c
    @author Frederic Dauvergne
    @date   November 2012
    @Last_update February 2019 by Hamza El Bouhargani*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include "midapack.h"
#include "mappraiser.h"



int PCG_GLS_true(char *outpath, char *ref, Mat *A, Tpltz Nm1, double *x, double *b, double *noise, double *cond, int *lhits, double tol, int K)
{
  int 		i, j, k ;			// some indexes
  int		m, n, rank, size;
  double 	localreduce;			//reduce buffer
  double	st, t;				//timers
  double	solve_time;
  double	res, res0, *res_rel;
  double *tmp;
  FILE *fp;

  res_rel = (double *) malloc(1*sizeof(double));

  MPI_Comm comm = A->comm;

  m=A->m;					//number of local time samples
  n=A->lcount;					//number of local pixels
  MPI_Comm_rank(comm, &rank);		//
  MPI_Comm_size(comm, &size);		//


  double *_g, *ACg, *Ah, *Nm1Ah ;		// time domain vectors
  double *g, *Cg, *h ;		      		// map domain vectors
  double *AtNm1Ah ;                    		// map domain
  double ro, gamma, coeff ;			// scalars
  double g2pix, g2pixp;
  double norm2b;

/*
  printf("n=%d, m=%d, A->nnz=%d \n", n, m, A->nnz );
*/

//Init CG descent
  // double *c;
  // c = (double *) malloc(n*sizeof(double));
  Mat BJ;


// for no preconditionner:
 // for(j=0; j<n; j++)                    //
 //   c[j]=1.;

  st=MPI_Wtime();
  // precondjacobilike( A, Nm1, lhits, cond, c);
 // precondjacobilike_avg( A, Nm1, c);
 // Compute preconditioner and process degenerate pixels
  precondblockjacobilike(A, Nm1, &BJ, b, cond, lhits);

  /* for (i = 0; i < Nm1.local_V_size; ++i) { */
  /*   for (j = 0; j < A->nnz; ++j) { */
  /*     printf("Is nan ? ----> %i\n", isnan(A->values[i*(A->nnz)+j])); */
  /*   } */
  /* } */

  
// Redefine number of pixels in the map
  n=A->lcount-(A->nnz)*(A->trash_pix);
// Reallocate memory for well-conditioned map
  tmp = realloc(x, n * sizeof(double));
  if(tmp !=NULL){
    x = tmp;
  }




  
  int i0,i1,i2,i3;
  
  if (rank == 0) {
    printf("######## Start building ALS ##############\n");
  }
  fflush(stdout);
  MPI_Barrier(comm);

  
  int nb_defl = 20; // To give as argument of PCG_GLS_true later on
  int nb_blocks_loc;
  nb_blocks_loc = Nm1.nb_blocks_loc;

  
  double *Z1;
  Z1 = (double *) malloc(sizeof(double)*n*nb_defl*nb_blocks_loc);

  /* if (rank == 0) { */
  /*   printf("Adress of Z :\n"); */
  /* } */
  
  /* printf("r: %i, %p\n", rank,Z1); */
  /* fflush(stdout); */
  /* MPI_Barrier(comm); */
  
  /* printf("r: %i, %i\n",rank,n/A->nnz); */

  /* if (Z1 != NULL) { */
  /*   printf("Fuck\n"); */
  /*   fflush(stdout); */
  /* } */

  /* printf("Number of pixels : %i\n",n/A->nnz); */
  
  // Build the unorthogonalized coarse space of the blocks on a proc
  Build_ALS(A,Nm1,Z1,nb_defl,n,rank);
  
  if (rank == 0) {
    printf("######## ALS builed ##############\n");
  }
  fflush(stdout);
  
  double tol_svd;
  int new_size;
  tol_svd = 1e-10; // To give as argument of PCG_GLS_true later on

  if (rank == 0) {
    printf("######## Start Ortho ##############\n");
  }
  MPI_Barrier(comm);
  fflush(stdout);

  /* if (rank == 0) { */
  /*   printf("adress of Z1 : %p\n",&Z1); */
  /*   printf("value of Z1 : %p\n", Z1); */
  /* } */
  /* MPI_Barrier(comm); */
  /* fflush(stdout); */

  double *Z2 = NULL;
  
  // Orthogonalize the coarse space Z on a proc
  Z2 = Orthogonalize_Space_loc(Z1,n,nb_defl*nb_blocks_loc,&new_size,tol_svd,rank);

  /* if (rank == 0) { */
  /*   printf("Adress of Z :\n"); */
  /* } */

  /* printf("r: %i, %p\n", rank,Z2); */
  /* fflush(stdout); */
  /* MPI_Barrier(comm); */
    
  /* printf("r: %i, %i\n", rank, new_size); */
  
  /* MPI_Barrier(comm); */
  /* fflush(stdout); */
  /* MPI_Barrier(comm); */

  if (Z1 != NULL) {
    free(Z1);
  }
  else {
    printf("r: %i, Z1 NULL\n", rank);
  }

  /* double * Z_tmp; */
  /* Z_tmp = (double *) realloc(Z,n*new_size); */
  /* if (Z_tmp == NULL) { */
  /*   free(Z); */
  /*   printf("r: %i, FUCK\n", rank); */
  /* } */
  /* else { */
  /*   Z = Z_tmp; */
  /* } */

  
  if (rank == 0) {
    printf("######## End Ortho ##############\n");
  }
  MPI_Barrier(comm);
  fflush(stdout);

  /* printf("r: %i, CHECKING !\n", rank); */
  /* for (i = 0; i < new_size*n; ++i) { */
  /*   if (isnan(Z[i]) || (1-isfinite(Z[i])) ) { */
  /*     printf("r: %i, FUCK !\n", rank); */
  /*   } */
  /* } */

  /* printf("SOMETHING !\n"); */
  /* fflush(stdout); */
  /* MPI_Barrier(comm); */
  

  
  /* printf("r: %i, %i\n",rank,new_size); */
  
  /* // ###### Not mandatory, just to have an idea of the size os the CS locally, what's needed is the total size, ie the sum */
  /* int *size_CS; */
  /* size_CS = (int *) calloc(nb_defl*nb_blocks_loc,sizeof(int)); */

  /* ++size_CS[new_size-1]; */

  /* /\* printf("r: %i\n",rank); *\/ */
  /* /\* for (i = 0; i < nb_defl*nb_blocks_loc; ++i) { *\/ */
  /* /\*   printf("%i ", size_CS[i]); *\/ */
  /* /\* } *\/ */
  /* /\* printf("\n"); *\/ */

  /* int *size_CS_reduce; */
  /* size_CS_reduce = (int *) calloc(nb_defl*nb_blocks_loc,sizeof(int)); */

  /* MPI_Allreduce(size_CS,size_CS_reduce,nb_defl*nb_blocks_loc,MPI_INT,MPI_SUM,comm); */

  /* free(size_CS); */
  
  /* printf("r: %i\n",rank); */
  /* for (i = 0; i < nb_defl*nb_blocks_loc; ++i) { */
  /*   printf("%i ", size_CS_reduce[i]); */
  /* } */
  /* printf("\n"); */

  // ######


  /* int fake_n = 2; */
  /* double *fake_CS; */
  /* fake_CS = (double *) malloc((rank+1)*fake_n*sizeof(double)); */

  /* /\* printf("r: %i, %i \n",rank,(rank+1)*fake_n); *\/ */

  /* for (i = 0; i < rank+1; ++i) { */
  /*   for (j = 0; j < fake_n; ++j) { */
  /*     fake_CS[i*fake_n+j] = i+1; */
  /*     // printf("%i ",i+1); */
  /*   } */
  /* } */

  /* printf("\n"); */
  
  if (rank == 0) {
    printf("######## Start Communicate CS ##############\n");
  }
  MPI_Barrier(comm);
  fflush(stdout);
  
  double *Z3 = NULL;

  int tot_size_CS;

  /* tot_size_CS = Communicate_sizes(new_size,comm); */

  /* printf("r: %i, %i\n", rank,tot_size_CS); */
  
  /* Z3 = (double *) malloc(sizeof(double)*n*tot_size_CS); */

  /* if(rank == 15) */
  /*   printf("r: %i, %i\n",rank,n); */

  int nb_proc = size;
  
  // Array of zeros containing the size of the CS of proc rank at position rank
  int *size_on_proc;
  size_on_proc = (int *) calloc(nb_proc,sizeof(int));
  size_on_proc[rank] = new_size;

  // size_on_proc_reduce is the sum of the size_on_proc array element wise
  int * size_on_proc_reduce;
  size_on_proc_reduce = (int *) calloc(nb_proc,sizeof(int));
  MPI_Allreduce(size_on_proc,size_on_proc_reduce,nb_proc,MPI_INT,MPI_SUM,comm);

  free(size_on_proc);

  /* if(rank == 15) */
  /* printf("r: %i\n",rank); */

  tot_size_CS = 0;
  
  // Compute the total size of the CS
  for (i0 = 0; i0 < nb_proc; ++i0) {
    // if(rank == 15)
    // printf("%i ", size_on_proc_reduce[i0]*n);
    tot_size_CS += size_on_proc_reduce[i0];
  }

  /* if(rank == 15) */
  /* printf("\n"); */

  /* fflush(stdout); */
  
  /* MPI_Barrier(comm); */

  /* printf("r: %i, %i\n",rank,tot_size_CS); */

  Z3 = (double *) malloc(sizeof(double)*n*(tot_size_CS));
  if (Z3 == NULL) {
    printf("PROBLEM FOR r: %i\n",rank);
  }
  
  int *location;
  int *recvcount;
  location = (int *) calloc(nb_proc,sizeof(int));
  recvcount = (int *) calloc(nb_proc,sizeof(int));

  /* int tmp = 0; */
  for (i0 = 1; i0 < nb_proc; ++i0) {
    location[i0] = size_on_proc_reduce[i0-1]*n + location[i0-1];
    // tmp = location[i0];
  }

  // if(rank == 15)
    // printf("r: %i\n",rank);
  
  for (i0 = 0; i0 < nb_proc; ++i0) {
    recvcount[i0] = size_on_proc_reduce[i0]*n;
    // if(rank == 15)
      // printf("%i ",location[i0]);
    // printf("%i ",recvcount[i0]/n);
  }

  /* if(rank == 15) */
  /* printf("\n"); */

  /* if (rank == 0) { */
  /*   printf("-----> HERE : r: %i\n",rank); */
  /* } */
  
  /* fflush(stdout); */

  /* MPI_Barrier(comm); */
  
  MPI_Allgatherv(Z2,new_size*n,MPI_DOUBLE,Z3,recvcount,location,MPI_DOUBLE,comm);

  /* printf("r: %i ------> HERE !!\n", rank); */
  /* fflush(stdout); */
  
  /* int i; */
  /* if (rank == 0) { */
  /*   for (i = 0; i < (tot_size_CS)*n; ++i) { */
  /*     printf("%f ",Z2[i]); */
  /*   } */
  /*   printf("\n"); */
  /* } */



  
  /* Z3 = Communicate_CS(Z2,new_size,&tot_size_CS,comm,n); */

  /* if (rank == 0) { */
  /*   printf("Adress of Z :\n"); */
  /* } */
  
  /* printf("r: %i, %p\n", rank,Z3); */
  /* fflush(stdout); */
  /* MPI_Barrier(comm); */
  
  /* printf("r: %i, %p\n",rank,Z3); */

  /* printf("r: %i, %i\n",rank,tot_size_CS); */
  
  /* double norm = 0; */
  /* int i0, i1; */
  /* for (i0 = 0; i0 < tot_size_CS; ++i0) { */
  /*   for (i1 = 0; i1 < n; ++i1) { */
  /*     norm += Z3[i0*n+i1]*Z3[i0*n+i1]; */
  /*   } */
  /* } */
  /* printf("r: %i, %f\n", rank,sqrt(norm)); */
  /* fflush(stdout); */


  if (Z2 != NULL) {
    free(Z2);
  }
  else {
    printf("r: %i, Z2 NULL\n", rank);
  }
  
  MPI_Barrier(comm);
  
  if (rank == 0) {
    printf("######## End Communicate CS ##############\n");
  }
  fflush(stdout);
  MPI_Barrier(comm);
  
  /* printf("r: %i\n", rank); */
  
  /* for (i = 0; i < 5; ++i) { */
  /*   printf("%f ", Z3[i]); */
  /* } */
  /* printf("\n"); */
  
  /* double *Z4 = NULL; */

  if (rank == 3) {
    printf("Adress of Z : %p\n", Z3);
  }
  fflush(stdout);
  MPI_Barrier(comm);

  /* new_size = Orthogonalize_Space_loc(Z3,n,tot_size_CS,Z4,tol_svd,rank); */
  
  /* if (Z3 != NULL) { */
  /*   free(Z3); */
  /* } */
  /* else { */
  /*   printf("r: %i, Z3 NULL\n", rank); */
  /* } */

  /* if (rank == 0) { */
  /*   printf("Adress of Z :\n"); */
  /* } */

  /* printf("r: %i, %p\n", rank,Z3); */
  /* fflush(stdout); */
  /* MPI_Barrier(comm); */
  

  /* printf("r: %i, %i\n", rank,new_size); */

  

  /* printf("r: %i, %i\n", rank,new_size); */
  /* fflush(stdout); */
  /* MPI_Barrier(comm); */
  
  /* tot_size_CS = Communicate_CS(fake_CS,rank+1,comm,fake_n); */

  /* if (rank == 0) { */
  /*   for (i = 0; i < tot_size_CS*fake_n; ++i) { */
  /*     printf("%f ",fake_CS[i]); */
  /*   } */
  /*   printf("\n"); */
  /* } */



  /* // Orthogonalize the coarse space CS on a proc */
  /* new_size = Orthogonalize_Space_loc(CS,n,tot_size_CS,tol_svd,rank); */
  
  /* double *E; */
  /* E = (double *) malloc(new_size*new_size*sizeof(double)); */

  /* Factorize_CS(A,Nm1,CS,E); */



  

  printf("Ok for me : %i\n", rank);
  fflush(stdout);
  MPI_Barrier(comm);
  
  //map domain
  h = (double *) malloc(n * sizeof(double));      //descent direction
  g = (double *) malloc(n * sizeof(double));
  AtNm1Ah = (double *) malloc(n * sizeof(double));

  //time domain
  Ah = (double *) malloc(m * sizeof(double));
  
  _g = Ah;
  Cg = AtNm1Ah;
  Nm1Ah = Ah;
  
  double *pixpond;
  pixpond = (double *) malloc(n * sizeof(double));

  //compute pixel share ponderation
  get_pixshare_pond(A, pixpond);

  //if we want to use the true norm to compute the residual
  int TRUE_NORM=1;  //0: No ; 1: Yes


  t=MPI_Wtime();

   if(rank==0)
     printf("Init PCG   t=%lf \n", t-st);
   fflush(stdout);
   
   st=MPI_Wtime();
   
   MatVecProd(A, x, _g, 0);		// _g = A*x
   // for(i=0; i<50; i++){//
   //     printf("MatVecProd: _g[%d] = %f\n",i,_g[i]);
   // }
  
   for(i=0; i<m; i++)	//
     _g[i] = b[i] + noise[i] - _g[i];		// _g = b+n-A*x
   // for(i=0; i<50; i++){
   //   printf("b-_g:_g[%d] = %f\n",i,_g[i]);
   // }
   
   stbmmProd(Nm1, _g);		// _g = N*b+N*n-N*Ax)
   // for(i=0; i<50; i++){//
   //     printf("Nm1*_g: _g[%d] = %f\n",i,_g[i]);
   // }

   
   TrMatVecProd(A, _g, g, 0);		//  g = P.T N (b+n)-(P.T N P)x
   // for(i=0; i<50; i++){			//
   //     printf("At*_g: index = %d, g[%d] = %.18f\n", A->lindices[i], i, g[i]);
   // }






   /* Apply_ADEF1(BJ,A,Nm1,CS,E,g,Cg); */
   
   MatVecProd(&BJ, g, Cg, 0);
   // for(j=0; j<n; j++)                    //
   //   Cg[j]=c[j]*g[j]; 			//  Cg = M1*P.T*N*(b+n)-M1*(P.T N P)x
   // for(i=3360; i<3380; i++){
   //     printf("index = %d , Cg[%d]=%.18f\n", A->lindices[i], i, Cg[i]);
   // }
   

   for(j=0; j<n; j++)   		        //  h = -Cg
     h[j]=Cg[j];
   
   
   g2pix=0.0;                               //g2 = "res"
   localreduce=0.0;
   for(i=0; i<n; i++)                    //  g2 = (Cg , g)
     localreduce+= Cg[i] * g[i] * pixpond[i];
   
   MPI_Allreduce(&localreduce, &g2pix, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
   t=MPI_Wtime();
   
   solve_time += (t-st);
   //Just to check with the true norm:
   if (TRUE_NORM==1) {
     res=0.0;                               //g2 = "res"
     localreduce=0.0;
     for(i=0; i<n; i++)                    //  g2 = (Cg , g)
       localreduce+= g[i] * g[i] * pixpond[i];
     
     MPI_Allreduce(&localreduce, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   }
   else {
     
     res = g2pix;
     
   }//end if
   
   double g2pixB = g2pix;
   double tol2rel = tol*tol*res;//tol*tol*g2pixB;//*g2pixB;//tol;//*tol*g2;
   res0=res;
   //Test if already converged
   if(rank==0) {
     *res_rel = sqrt(res)/sqrt(res0);
     printf("res=%e \n", res);
     printf("k=%d res_g2pix=%e res_g2pix_rel=%e res_rel=%e t=%lf\n", 0, g2pix , sqrt(g2pix)/sqrt(g2pixB), sqrt(res)/sqrt(res0), t-st);
     char filename[256];
     sprintf(filename,"%s/pcg_residuals_%s.dat",outpath, ref);
     fp=fopen(filename, "wb");
     fwrite(res_rel, sizeof(double), 1, fp);
   }
   
   
   if(res<tol){                         //
     if(rank==0)                      //
       printf("--> converged (%e < %e)\n", res, tol);
     k=K;//to not enter inside the loop
   }
   
   st=MPI_Wtime();
   
   fflush(stdout);
   
   // PCG Descent Loop
   for(k=1; k<K ; k++){
     
     
     MatVecProd(A, h, Ah, 0);		// Ah = A h
     // for(i=0; i<8; i++){//
     //     printf("MatVecProd: Ah[%d] = %f\n",i,Ah[i]);
     // }
     stbmmProd(Nm1, Nm1Ah);		// Nm1Ah = Nm1 Ah   (Nm1Ah == Ah)
     // for(i=0; i<8; i++){//
     //     printf("Nm1Ah: Nm1Ah[%d] = %f\n",i,Nm1Ah[i]);
     // }
     TrMatVecProd(A, Nm1Ah, AtNm1Ah, 0); //  AtNm1Ah = At Nm1Ah
     // for(i=n-1; i>n-9; i--){//
     //     printf("lhs: AtNm1Ah[%d] = %f\n",i,AtNm1Ah[i]);
     // }
     coeff=0.0;
     localreduce=0.0;
     for(i=0; i<n; i++)
       localreduce+= h[i]*AtNm1Ah[i] * pixpond[i] ;
     MPI_Allreduce(&localreduce, &coeff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     // printf("Coeff = %f\n",coeff);
/*
  ro=0.0;
  localreduce=0.0;
  for(i=0; i<n; i++)
  localreduce+= g[i]*Cg[i] *pixpond[i] ;
  MPI_Allreduce(&localreduce, &ro, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
*/
     ro = g2pix;
     
     ro = ro/coeff;
     // printf("ro = %f\n",ro);
     
     for(j=0; j<n; j++)			//  x = x + ro*h
       x[j] = x[j] + ro*h[j];		//
     
     
     for(j=0; j<n; j++)                  //   g = g + ro * (At Nm1 A) h
       g[j] = g[j] - ro * AtNm1Ah[j] ;
    
     MatVecProd(&BJ, g, Cg, 0);
     // for(j=0; j<n; j++)                  //
     //   Cg[j]=c[j]*g[j];                       //  Cg = C g  with C = Id
     // for(i=n-1; i>n-9; i--){//
     //     printf("g[%d] = %.34f\n",i,g[i]);
     //     printf("Cg[%d] = %.34f\n",i,Cg[i]);
     // }


     g2pixp=g2pix;                               // g2p = "res"
     localreduce=0.0;
     for(i=0; i<n; i++)                    // g2 = (Cg , g)
       localreduce+= Cg[i] * g[i] *pixpond[i];

     MPI_Allreduce(&localreduce, &g2pix, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     
     t=MPI_Wtime();                       //stop timer
     solve_time += (t-st);
     
     //Just to check with the true norm:
     if (TRUE_NORM==1) {
       localreduce=0.0;
       for(i=0; i<n; i++)                    // g2 = (Cg , g)
	 localreduce+= g[i] * g[i] *pixpond[i];
      
       MPI_Allreduce(&localreduce, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     }
     else{
       res = g2pix;
     }//end if
     
     
     if(rank==0){                          //print iterate info
       *res_rel = sqrt(res)/sqrt(res0);
       printf("k=%d res_g2pix=%e res_g2pix_rel=%e res_rel=%e t=%lf \n", k, g2pix, sqrt(g2pix)/sqrt(g2pixB), sqrt(res)/sqrt(res0), t-st);
       fwrite(res_rel, sizeof(double), 1, fp);
     }
     //   if(g2pix<tol2rel){                         //
     fflush(stdout);
     
     if(res<tol2rel){
       if(rank ==0) {                     //
	 printf("--> converged (%e < %e) \n", res, tol2rel);
	 printf("--> i.e. \t (%e < %e) \n", sqrt(res/res0), tol);
	 printf("--> solve time = %lf \n", solve_time);
	 fclose(fp);
       }
       break;
     }
     if(g2pix>g2pixp){                         //
       if(rank ==0)                      //
	 printf("--> g2pix>g2pixp pb (%e > %e) \n", g2pix, g2pixp);
       //    break;                        //
     }
     
     st=MPI_Wtime();
     
     
     gamma = g2pix/g2pixp ;
     
     for(j=0; j<n; j++)			// h = h * gamma - Cg
       h[j] = h[j] * gamma + Cg[j] ;
   }  //End loop

   
   
   if(k==K){				//check unconverged
     if(rank==0){
       printf("--> unconverged, max iterate reached (%lf > %lf)\n", g2pix, tol2rel);
       fclose(fp);
     }
   }

   if(rank ==0)
     printf("--> res_g2pix=%e  \n", g2pix);
   
   
   free(h);
   free(Ah);
   free(g);
   free(AtNm1Ah);
   free(res_rel);
   
   
   return 0;
}
