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
#include <mkl.h>



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
  int info;
  
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

  printf("r: %i, local length of the pixel domains : %i\n", rank,n);
  fflush(stdout);

  if (A->nnz > 3) {
    printf("r: %i, A->nnz = %i\n", rank,A->nnz);
    fflush(stdout);
  }
  
  
// Reallocate memory for well-conditioned map
  tmp = realloc(x, n * sizeof(double));
  if(tmp !=NULL){
    x = tmp;
  }




  
  int i0,i1,i2,i3;

  int nb_task_per_node = 32;
  int new_size;
  double *Z2; // free l. 438
  double **arg1;
  double tol_svd;
  double *Z3;

  int rest = 0 ; // rank%nb_task_per_node;
  
  /* printf("r: %i, rest = %i\n", rank,rest); */
  /* fflush(stdout); */

  if (rest == 0) {

    /* printf("r: %i, HERE\n", rank); */
    /* fflush(stdout); */
    
    if (rank == 0) {
      printf("######## Start building ALS ##############\n");
    }
    /* MPI_Barrier(comm); */
    fflush(stdout);
    int nb_defl = 2; // To give as argument of PCG_GLS_true later on
    int nb_blocks_loc;
    nb_blocks_loc = Nm1.nb_blocks_loc;
    
    /* printf("r: %i : nb_defl = %i, nb_blocks_loc = %i, so nb_vec_in_CS_loc = %i\n", rank,nb_defl,nb_blocks_loc,nb_defl*nb_blocks_loc); */
    /* fflush(stdout); */
    
    double *Z1; // free l. 200
    Z1 = (double *) calloc(n*nb_defl*nb_blocks_loc,sizeof(double));
    
    // Build the unorthogonalized coarse space of the blocks on a proc
    Build_ALS(A,Nm1,Z1,nb_defl,n,rank);
    
    if (rank == 0) {
      printf("######## ALS built ##############\n");
    }
    fflush(stdout);
    
    tol_svd = 1e-10; // To give as argument of PCG_GLS_true later on
    
    if (rank == 0) {
      printf("######## Start Ortho ##############\n");
    }
    fflush(stdout);
    
    arg1  = (double **) malloc(sizeof(double *));
    *arg1 = Z1;
  
    // Orthogonalize the coarse space Z on a proc
    new_size = Orthogonalize_Space_loc(arg1,n,nb_defl*nb_blocks_loc,tol_svd,rank);
  
    printf("r: %i : Nb of vec in CS after loc orth. : %i\n", rank, new_size);
    fflush(stdout);
  
    Z2 = *arg1;

    if (Z1 != NULL) {
      free(Z1);
    }
    else {
      printf("Z1 NULL\n");
    }

    if (rank == 0) {
      printf("######## End Ortho ##############\n");
    }
    fflush(stdout);
    
  }
  





  
  
  int nb_proc = size;

  if (rank == 0) {
    printf("###### From local map to map of undegenerated pixels #########\n");
    fflush(stdout);
  }
  
  int *gather_lcount = (int *) malloc(sizeof(int)*nb_proc);
  if (gather_lcount == NULL) {
    printf("gather_lcount NULL\n");
    fflush(stdout);
  }

  
  int sum_lcount = 0;
  
  MPI_Allgather(&n,1,MPI_INT,gather_lcount,1,MPI_INT,comm);

  for (i0 = 0; i0 < nb_proc; ++i0) {
    sum_lcount += gather_lcount[i0];
  }
  
  int *gather_lindices = (int *) malloc(sizeof(int)*sum_lcount); // free l. 444

  int *recvcount = (int *) malloc(sizeof(int)*nb_proc);
  int *location = (int *) malloc(sizeof(int)*nb_proc);

  location[0] = 0;
  // recvcount[0] = gather_lcount[0];

  /* int twice; */
  for (i0 = 1; i0 < nb_proc; ++i0) {
    // twice = gather_lcount[i0];
    // recvcount[i0] = twice;
    location[i0] = location[i0-1]+gather_lcount[i0-1];
  }
  

  MPI_Allgatherv((A->lindices)+A->trash_pix*A->nnz,n,MPI_INT,gather_lindices,gather_lcount,location,MPI_INT,comm);

  int dim_CS;

  dim_CS = ssort(gather_lindices,sum_lcount,3);

  if (rank == 0) {
    printf("local map : n=3*np=%i pix, map of undegenerated pix : dim_CS=%i pix\n", n,dim_CS);
    fflush(stdout);
  }

  int *invert_gather_lindices = (int *) malloc((gather_lindices[dim_CS-1]+1)*sizeof(int)); // free l. 445

  for (i0 = 0; i0 < dim_CS; ++i0) {
    invert_gather_lindices[gather_lindices[i0]] = i0;
  }
  
  int index_pix;

  // This array contains the bijection between the local size of the CS n and the contribution of local pixels to the Krylov space size dim_CS. Needed to extend and compress the CS between those sizes.
  int *bij = NULL;

  bij = (int *) malloc(n*sizeof(int)); // free l. ?
  
  for (i1 = 0; i1 < n; ++i1) {
    if (i1+(A->trash_pix)*(A->nnz) >= A->lcount) {
      printf("r: %i, FUCK1, element: %i, n: %i\n", rank,i1+(A->trash_pix)*(A->nnz), n);
      fflush(stdout);
    }
    
    if ( A->lindices[i1+(A->trash_pix)*(A->nnz)] >= gather_lindices[dim_CS-1]+1 ) {
      printf("r: %i, FUCK2, element: %i, borne: %i\n", rank,A->lindices[i1+(A->trash_pix)*(A->nnz)],gather_lindices[dim_CS-1]);
      fflush(stdout);
    }
      
    index_pix = invert_gather_lindices[A->lindices[i1+(A->trash_pix)*(A->nnz)]]; // test un -1 sur le retour de lindices
    bij[i1] = index_pix;
    
  }
  
  free(gather_lindices);
  free(invert_gather_lindices);


  if (rank == 0) {
    printf("###### END From local map to map of undegenerated pixels #########\n");
  }
  fflush(stdout);













  
  int *position, *count;
  position = (int *) calloc(nb_proc,sizeof(int));
  count = (int *) malloc(sizeof(int)*nb_proc);

  /* long *position, *count; */
  /* position = (long *) calloc(nb_proc,sizeof(long)); */
  /* count = (long *) malloc(sizeof(long)*nb_proc); */
  
  /* position[0] = 0; */
  
  MPI_Allgather(&new_size,1,MPI_INT,count,1,MPI_INT,comm);
  MPI_Barrier(comm);
  
  int tot_size_CS = 0;

  
  /* printf("r: %i, count = [", rank); */
  
  for (i0 = 0; i0 < nb_proc; ++i0) {
    
    tot_size_CS += count[i0];
    count[i0] *= dim_CS;
    
    /* printf("%i, ", count[i0]); */
    
  }

  /* printf("]\n"); */

  /* printf("r: %i, position = [", rank); */
  
  for (i0 = 1; i0 < nb_proc; ++i0) {

    position[i0] = position[i0-1] + count[i0-1];

    /* printf(" %i,", position[i0]); */
    
  }

  /* printf("]\n"); */
  /* fflush(stdout); */

  if (rank == 0) {
    
    printf("count[i0] = ");
    for (i0 = 0; i0 < nb_proc; ++i0) {
      printf(" %i,", count[i0]);
    }
    printf("\n");
    fflush(stdout);

    printf("posision[i0] = ");
    for (i0 = 0; i0 < nb_proc; ++i0) {
      printf(" %i,", position[i0]);
    }
    printf("\n");
    fflush(stdout);
    
  }
  
  double *Z4;
  double *Z5;
  
  if (rest == 0) {
    double *Z3;

    Z3 = (double *) calloc(dim_CS*new_size,sizeof(double));

    for (i0 = 0; i0 < new_size; ++i0) {
      for (i1 = 0; i1 < n; ++i1) {
  	if (i1 >= n) {
  	  printf("r: %i, FUCK i1 step out of bij : i1 = %i, bij limit = %i\n", rank,i1,n);
  	  fflush(stdout);
  	}

  	if (i0 >= new_size) {
  	  printf("r: %i, FUCK i0 step out of Z3 and Z2 : i0 = %i, Z limit = %i\n", rank,i0,new_size);
  	  fflush(stdout);
  	}

  	if (bij[i1] >= dim_CS) {
  	  printf("r: %i, FUCK bij[i1] step out of Z3 : bij[i1] = %i, Z limit = %i\n", rank,i1,dim_CS);
  	  fflush(stdout);
  	}
	
  	Z3[dim_CS*i0+bij[i1]] = Z2[n*i0+i1];
      }
    }
  
    Z4 = (double *) malloc(sizeof(double)*dim_CS*tot_size_CS);

    printf("r: %i, FUCK problem with the Allgather de Z3 : (count[nb_proc-1]=%i)+(position[nb_proc-1]=%i) = %i,  dim_CS*tot_size_CS = %i\n", rank,count[nb_proc-1],position[nb_proc-1],count[nb_proc-1]+position[nb_proc-1],dim_CS*tot_size_CS);
    fflush(stdout);
    
    if (count[nb_proc-1]+position[nb_proc-1] > dim_CS*tot_size_CS) {
      printf("r: %i, FUCK problem with the Allgather de Z3 : count[nb_proc-1]+position[nb_proc-1] = %i,  dim_CS*tot_size_CS = %i\n", rank,count[nb_proc-1]+position[nb_proc-1],dim_CS*tot_size_CS);
      fflush(stdout);
    }


    MPI_Barrier(comm);
    MPI_Allgatherv(Z3,dim_CS*new_size,MPI_DOUBLE,Z4,count,position,MPI_DOUBLE,comm);

    *arg1 = Z4;

    MPI_Barrier(comm);
    printf("r: %i, HERE\n", rank);
    fflush(stdout);
    
    new_size = Orthogonalize_Space_loc(arg1,dim_CS,tot_size_CS,tol_svd,rank);


    free(Z4);

    Z4 = *arg1;

    tot_size_CS = new_size;

    free(Z3);
    
    Z5 = (double *) malloc(sizeof(double)*n*tot_size_CS);

    for (i0 = 0; i0 < tot_size_CS; ++i0) {
      for (i1 = 0; i1 < n; ++i1) {
  	if (i1 >= n) {
  	  printf("r: %i, FUCK i1 step out of bij : i1 = %i, bij limit = %i\n", rank,i1,n);
  	  fflush(stdout);
  	}

  	if (i0 >= tot_size_CS) {
  	  printf("r: %i, FUCK i0 step out of Z4 and Z5 : i0 = %i, Z limit = %i\n", rank,i0,tot_size_CS);
  	  fflush(stdout);
  	}
	
  	Z5[n*i0+i1] = Z4[dim_CS*i0+bij[i1]];
      }
    }
    
    free(Z4);
    

  }
    
  if (rank == 0) {
    printf("The complete CS Z4 is dim_CS = %i rows by tot_size_CS = %i cols\n",dim_CS,tot_size_CS);
    fflush(stdout);
  }
  
  









  







  
  
  
  /* MPI_Bcast(Z5,nb_col_Q*dim_CS,MPI_DOUBLE,0,comm); */
  
  /* double *Z6 = (double *) malloc(n*nb_col_Q*sizeof(double));   */
  
  /* for (i0 = 0; i0 < nb_col_Q; ++i0) { */
  /*   for (i1 = 0; i1 < n; ++i1) { */
  /*     Z6[i0*n+i1] = Z5[i0*dim_CS+bij[i1]]; */
  /*   } */
  /* } */

  int nb_col_Q = tot_size_CS;
  int row_indice = 0;

  /* double *Z5; */
  /* Z5 = Z2; */
  
  double *pz = (double *) malloc(m*sizeof(double));
  double *pTnpz = (double *) malloc(n*sizeof(double));
  
  double *Z7 = (double *) calloc(n*nb_col_Q,sizeof(double));

  /* double norm; */
  
  for (i0 = 0; i0 < nb_col_Q; ++i0) {
    MatVecProd(A,&(Z5[i0*n]),pz,0);

    stbmmProd(Nm1,pz);

    TrMatVecProd(A,pz,pTnpz,0);

/* // Apply the product A_i.T * v for a timestream of one block */
/* int Apply_ATr_bloc(Mat *A, double *x, double *y, int size_y, int row_indice, int nb_rows) */
/* { */
/*   int i0,i1;           // Some loop index */
/*   int nnz; */

/*   nnz = A->nnz;   */
/*   for (i0 = 0; i0 < nb_rows; ++i0) { */
    
/*     int pix_index; */
/*     pix_index = A->indices[(row_indice+i0)*nnz]/nnz; */
    
/*     for(i1 = 0; i1 < nnz; ++i1){ */
      
/*       // if there are trashed pixels and that we watch a pixel, we need to shift indices by 1 due to presence of fictitious pixel at index 0 */
/*       if (A->trash_pix == 1) { */
/*   	if (pix_index =! 0) { */
/*   	  // printf("%i \n",A->indices[(row_indice+i0)*nnz+i1]-nnz < size_y); */
/*   	  y[A->indices[(row_indice+i0)*nnz+i1]-nnz] += (A->values[(row_indice+i0)*nnz+i1]) * x[i0];//  */
/*   	} */
/*       } */
/*       else{ */
/* 	// printf("%i \n",A->indices[(row_indice+i0)*nnz+i1] < size_y); */
/*   	y[A->indices[(row_indice+i0)*nnz+i1]] += (A->values[(row_indice+i0)*nnz+i1]) * x[i0]; //  */
/*       } */
/*     } */
/*   } */
  
/*   return 0; */
/* } */

    /* for (i1 = 0; i1 < Nm1.nb_blocks_loc; ++i1) { */

    /*   Apply_ATr_bloc(A,pz,pTnpz,n,row_indice,Nm1.tpltzblocks[i1].n); */

    /*   row_indice += Nm1.tpltzblocks[i1].n; */
      
      
    /* } */
    
    /* norm = 0; */

    for (i2 = 0; i2 < n; ++i2) {
      Z7[i0*n+i2] = pTnpz[i2];
      // Z7[i0*dim_CS+bij[i2]] = pTnpz[i2]; // divide by the number of proc that watch pixel i2 ? No, apparently
      // norm += pTnpz[i2];
    }

    /* printf("r: %i, norm = %f\n", rank,sqrt(norm)); */
    /* fflush(stdout); */
  }

  MPI_Barrier(comm);
  if (rank == 0) {
    printf("We are done applying the operator to the CS\n");
  }
  fflush(stdout);







  

  double *E_loc = (double *) malloc(sizeof(double)*nb_col_Q*nb_col_Q);
  double *E = (double *) malloc(sizeof(double)*nb_col_Q*nb_col_Q);

  

  cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,nb_col_Q,nb_col_Q,n,1,Z5,n,Z7,n,0,E_loc,nb_col_Q);

  MPI_Allreduce(E_loc, E, nb_col_Q*nb_col_Q, MPI_DOUBLE, MPI_SUM, comm);

  free(E_loc);


  
  // ###### Call LAPACK routines to compute Cholesky factorisation of E
  info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',nb_col_Q,E,nb_col_Q);
  
  if (info > 0) {
    printf("The leading minor of order %i is not positive-definite, and the factorization could not be completed.",info);
    fflush(stdout);
  }
  if (info < 0) {
    printf("Compute Cholesky factorisation of E : The parameter %i had an illegal value.\n",-info);
    fflush(stdout);
  }  

  /* MPI_Barrier(comm); */
  /* printf("r: %i, did we reach here ?\n", rank); */
  /* fflush(stdout); */
















  


  
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




   /* double *g_extended = (double *) calloc(dim_CS,sizeof(double)); */

   /* for (i0 = 0; i0 < n; ++i0) { */
   /*   g_extended[bij[i0]] = g[i0]; */
   /* }    */

   double *EZTg = (double *) malloc(nb_col_Q*sizeof(double));

   cblas_dgemv(CblasColMajor, CblasTrans, n, nb_col_Q, 1, Z5, n, g, 1, 0, EZTg, 1);

   /* double *EQg = (double *) malloc(nb_col_Q*sizeof(double)); */

   info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR,'L',nb_col_Q,1,E,nb_col_Q,EZTg,1);
   
   if (info < 0) {
     printf("Something went wrong applying E inverse to a vector : info = %i\n",info);
   }

   double *Qg = (double *) malloc(n*sizeof(double));

   cblas_dgemv(CblasColMajor, CblasNoTrans, n, nb_col_Q, 1, Z5, n, EZTg, 1, 0, Qg, 1);

   /* double *Qg_compress = (double *) malloc(n*sizeof(double)); */

   /* for (i0 = 0; i0 < n; ++i0) { */
   /*   Qg_compress[i0] = g_extended[bij[i0]]; */
   /* } */

   MatVecProd(A,Qg,pz,0);
   stbmmProd(Nm1,pz);
   TrMatVecProd(A,pz,pTnpz,0);

   /* double *pTnpz_extended = (double *) calloc(dim_CS,sizeof(double)); */

   /* for (i0 = 0; i0 < n; ++i0) { */
   /*   pTnpz_extended[bij[i0]] = pTnpz[i0]; */
   /* } */

   /* double *pTnpz_extended_reduce = (double *) calloc(dim_CS,sizeof(double)); */
   
   /* MPI_Allreduce(pTnpz_extended,pTnpz_extended_reduce,dim_CS,MPI_INT,MPI_SUM,comm); */

   /* for (i0 = 0; i0 < n; ++i0) { */
   /*   pTnpz[i0] = pTnpz_extended_reduce[bij[i0]]; */
   /* } */

   for (i0 = 0; i0 < n; ++i0) {
     g[i0] -= pTnpz[i0];
   }
   
   MatVecProd(&BJ, g, Cg, 0);
   // for(j=0; j<n; j++)                    //
   //   Cg[j]=c[j]*g[j]; 			//  Cg = M1*P.T*N*(b+n)-M1*(P.T N P)x
   // for(i=3360; i<3380; i++){
   //     printf("index = %d , Cg[%d]=%.18f\n", A->lindices[i], i, Cg[i]);
   // }

   /* double *M2g = (double *) malloc(n*sizeof(double)); */

   for (i0 = 0; i0 < n; ++i0) {
     Cg[i0] += Qg[i0];
   }

   

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

     


     

     /* double *g_extended = (double *) calloc(dim_CS,sizeof(double)); */

     /* for (i0 = 0; i0 < n; ++i0) { */
     /*   g_extended[bij[i0]] = g[i0]; */
     /* } */

     /* double *EZTg = (double *) malloc(nb_col_Q*sizeof(double)); */

     cblas_dgemv(CblasColMajor, CblasTrans, n, nb_col_Q, 1, Z5, n, g, 1, 0, EZTg, 1);

     /* double *EQg = (double *) malloc(nb_col_Q*sizeof(double)); */

     info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR,'L',nb_col_Q,1,E,nb_col_Q,EZTg,1);
     
     if (info < 0) {
       printf("Something went wrong applying E inverse to a vector : info = %i\n",info);
     }

     cblas_dgemv(CblasColMajor, CblasNoTrans, n, nb_col_Q, 1, Z5, n, EZTg, 1, 0, Qg, 1);

     /* double *Qg_compress = (double *) malloc(n*sizeof(double)); */

     /* for (i0 = 0; i0 < n; ++i0) { */
     /*   Qg_compress[i0] = g_extended[bij[i0]]; */
     /* } */

     MatVecProd(A,Qg,pz,0);
     stbmmProd(Nm1,pz);
     TrMatVecProd(A,pz,pTnpz,0);

     /* printf("r: %i, HERE\n", rank); */
     /* MPI_Barrier(comm); */
     /* fflush(stdout); */

     for (i0 = 0; i0 < n; ++i0) {
       g[i0] -= pTnpz[i0];
     }
   
     MatVecProd(&BJ, g, Cg, 0);
     // for(j=0; j<n; j++)                    //
     //   Cg[j]=c[j]*g[j]; 			//  Cg = M1*P.T*N*(b+n)-M1*(P.T N P)x
     // for(i=3360; i<3380; i++){
     //     printf("index = %d , Cg[%d]=%.18f\n", A->lindices[i], i, Cg[i]);
     // }

     /* double *M2g = (double *) malloc(n*sizeof(double)); */

     for (i0 = 0; i0 < n; ++i0) {
       Cg[i0] += Qg[i0];
     }




     
   
     // MatVecProd(&BJ, g, Cg, 0);
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
