
// MAPPRAISER preconditioner vdev
// Routines for computing the diagonal and block-diagonal Jacobi preconditioners for the PCG
// The routines also deal with degenerate pixels to ensure numerical stability of the system

/** @file   precond.c
    @author Frederic Dauvergne
    @date   November 2012
    @Last_update May 2019 by Hamza El Bouhargani*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include "midapack.h"
#include "mappraiser.h"
#include <fftw3.h>
#include <mkl.h>
#include <arpackdef.h>

extern int dgecon_(const char *norm, const int *n, double *a, const int *lda, const double *anorm, double *rcond, double *work, int *iwork, int *info, int len);
extern int dgetrf_(const int *m, const int *n, double *a, const int *lda, int *lpiv, int *info);
extern double dlange_(const char *norm, const int *m, const int *n, const double *a, const int *lda, double *work, const int norm_len);

extern int dgeqp3(const int *m, const int *n, double *a, const int *lda, const double *tau, const double *work, const int *lwork, const int *info); //M, N, A, LDA, JPVT, TAU, WORK, LWORK, INFO

extern void dsaupd_(int *ido, char *bmat, int *n, char *which,
            int *nev, double *tol, double *resid, int *ncv,
            double *v, int *ldv, int *iparam, int *ipntr,
            double *workd, double *workl, int *lworkl,
            int *info);

extern void dseupd_(int *rvec, char *All, int *select, double *d,
            double *v, int *ldv, double *sigma,
            char *bmat, int *n, char *which, int *nev,
            double *tol, double *resid, int *ncv, double *tv,
            int *tldv, int *iparam, int *ipntr, double *workd,
            double *workl, int *lworkl, int *ierr);

/* void dsaupd_(int* ido, char *bmat, int *n, char *which, int *nev, double *tol, double *resid, int *ncv, double *v, int *ldv, int *iparam, int *ipntr, double *workd, double *workl, int *lworkl, int *info ); */

int precondblockjacobilike(Mat *A, Tpltz Nm1, Mat *BJ, double *b, double *cond, int *lhits)
{
  int           i, j, k ;                       // some indexes
  int           m, m_cut, n, rank, size;
  int *indices_new, *tmp1;
  double *vpixBlock, *vpixBlock_loc, *hits_proc, *tmp2,*tmp3;
  // float *vpixBlock, *tmp2;
  double det, invdet;
  // int pointing_commflag = 6;
  int info, nb, lda;
  double anorm, rcond;

  int iw[3];
  double w[18];
  double x[9];
  nb = 3;
  lda = 3;

  m = A->m;
  m_cut = m;
  n = A->lcount;
  MPI_Comm_rank(A->comm, &rank);                 //
  MPI_Comm_size(A->comm, &size);

  indices_new = (int *) malloc((A->nnz)*n * sizeof(int));
  vpixBlock_loc = (double *) malloc(n * sizeof(double));
  vpixBlock = (double *) malloc(n*(A->nnz) * sizeof(double));
  hits_proc = (double *) malloc(n * sizeof(double));

  // //Init vpixBlock
  // for(i=0;i<n*(A->nnz);i++)
  //   vpixBlock[i] = 0.;
  //
  // //Compute local AtA blocks (local sum over same pix blocks)
  // for(i=0;i<A->m;i++){
  //   for(j=0;j<A->nnz;j++){
  //     for(k=0;k<A->nnz;k++){
  //       vpixBlock[(A->nnz)*A->indices[i*(A->nnz)+j]+k] += A->values[i*(A->nnz)+j]*A->values[i*(A->nnz)+k];
  //     }
  //   }
  // }
  //Compute local Atdiag(N^1)A
  getlocalW(A, Nm1, vpixBlock, lhits);
  // sum hits globally
  for(i=0;i<n;i+=3){
    hits_proc[i] = lhits[(int)i/3];
    hits_proc[i+1] = lhits[(int)i/3];
    hits_proc[i+2] = lhits[(int)i/3];
  }
  commScheme(A, hits_proc, 2);
  for(i=0;i<n;i+=3){
    lhits[(int)i/3] = (int)hits_proc[i];
  }
  free(hits_proc);



  //communicate with the other processes to have the global reduce
  //TODO : This should be done in a more efficient way
  for(i=0;i<n*(A->nnz);i+=(A->nnz)*(A->nnz)){
    for(j=0;j<(A->nnz);j++){
      vpixBlock_loc[(i/(A->nnz))+j] = vpixBlock[i+j];
    }
  }
  commScheme(A, vpixBlock_loc, 2);
  for(i=0;i<n*(A->nnz);i+=(A->nnz)*(A->nnz)){
    for(j=0;j<(A->nnz);j++){
      vpixBlock[i+j] = vpixBlock_loc[(i/(A->nnz))+j];
    }
  }

  for(i=3;i<n*(A->nnz);i+=(A->nnz)*(A->nnz)){
    for(j=0;j<(A->nnz);j++)
      vpixBlock_loc[(i-3)/(A->nnz)+j] = vpixBlock[i+j];
  }
  commScheme(A, vpixBlock_loc, 2);
  for(i=3;i<n*(A->nnz);i+=(A->nnz)*(A->nnz)){
    for(j=0;j<(A->nnz);j++)
      vpixBlock[i+j] = vpixBlock_loc[(i-3)/(A->nnz)+j];
  }

  for(i=6;i<n*(A->nnz);i+=(A->nnz)*(A->nnz)){
    for(j=0;j<(A->nnz);j++)
      vpixBlock_loc[(i-6)/(A->nnz)+j] = vpixBlock[i+j];
  }
  commScheme(A, vpixBlock_loc, 2);
  for(i=6;i<n*(A->nnz);i+=(A->nnz)*(A->nnz)){
    for(j=0;j<(A->nnz);j++)
      vpixBlock[i+j] = vpixBlock_loc[(i-6)/(A->nnz)+j];
  }

  //Compute the inverse of the global AtA blocks (beware this part is only valid for nnz = 3)
  int uncut_pixel_index = 0;
  for(i=0;i<n*(A->nnz);i+=(A->nnz)*(A->nnz)){
    // lhits[(int)i/((A->nnz)*(A->nnz))] = (int)vpixBlock[i];
    //init 3x3 block
    double block[3][3];
    for(j=0;j<3;j++){
      for(k=0;k<3;k++){
        block[j][k] = vpixBlock[i+(j*3)+k];
        x[k+3*j] = block[j][k];
      }
    }

    //Compute the reciprocal of the condition number of the block

    /* Computes the norm of x */
    anorm = dlange_("1", &nb, &nb, x, &lda, w, 1);

    /* Modifies x in place with a LU decomposition */
    dgetrf_(&nb, &nb, x, &lda, iw, &info);
    // if (info != 0) fprintf(stderr, "failure with error %d\n", info);

    /* Computes the reciprocal norm */
    dgecon_("1", &nb, x, &lda, &anorm, &rcond, w, iw, &info, 1);
    // if (info != 0) fprintf(stderr, "failure with error %d\n", info);

    cond[(int)i/9] = rcond;

    //Compute det
    //TODO: This should take into account the fact that the blocks are symmetric
    det = block[0][0] * (block[1][1] * block[2][2] - block[2][1] * block[1][2]) -
             block[0][1] * (block[1][0] * block[2][2] - block[1][2] * block[2][0]) +
             block[0][2] * (block[1][0] * block[2][1] - block[1][1] * block[2][0]);

    if(rcond > 1e-1){
    invdet = 1 / det;

    //Compute the inverse coeffs
    //TODO: This should take into account the fact that the blocks are symmetric
    vpixBlock[i] = (block[1][1] * block[2][2] - block[2][1] * block[1][2]) * invdet;
    vpixBlock[i+1] = (block[0][2] * block[2][1] - block[0][1] * block[2][2]) * invdet;
    vpixBlock[i+2] = (block[0][1] * block[1][2] - block[0][2] * block[1][1]) * invdet;
    vpixBlock[i+3] = (block[1][2] * block[2][0] - block[1][0] * block[2][2]) * invdet;
    vpixBlock[i+4] = (block[0][0] * block[2][2] - block[0][2] * block[2][0]) * invdet;
    vpixBlock[i+5] = (block[1][0] * block[0][2] - block[0][0] * block[1][2]) * invdet;
    vpixBlock[i+6] = (block[1][0] * block[2][1] - block[2][0] * block[1][1]) * invdet;
    vpixBlock[i+7] = (block[2][0] * block[0][1] - block[0][0] * block[2][1]) * invdet;
    vpixBlock[i+8] = (block[0][0] * block[1][1] - block[1][0] * block[0][1]) * invdet;
    }
    else{// Remove the degenerate pixels from the map-making

      // Remove the poorly conditioned pixel from the map, point the associated gap samples to trash pixel
      // Set flag of trash pixel in pointing matrix to 1
      A->trash_pix = 1;
      // Search for the corresponding gap samples
      // j = A->id0pix[(int)uncut_pixel_index/((A->nnz)*(A->nnz))]; // first index of time sample pointing to degenerate pixel
      // // Point the first gap sample to trash pixel
      // for(k=0;k<(A->nnz); k++){
      //   A->indices[j*(A->nnz)+k] = k-(A->nnz);
      //   A->values[j*(A->nnz)+k] = 0;
      // }
      // // Point all the subsequent gap samples to trash pixel
      // while(A->ll[j]!= -1){
      //   for(k=0;k<(A->nnz); k++){
      //     A->indices[A->ll[j]*(A->nnz)+k] = k-(A->nnz);
      //     A->values[A->ll[j]*(A->nnz)+k] = 0;
      //   }
      //   j = A->ll[j];
      // }
      j = A->id0pix[(int)uncut_pixel_index/((A->nnz)*(A->nnz))]; // last index of time sample pointing to degenerate pixel
      // Point the last gap sample to trash pixel
      for(k=0;k<(A->nnz); k++){
        A->indices[j*(A->nnz)+k] = k-(A->nnz);
        A->values[j*(A->nnz)+k] = 0;
      }
      // Set the time stream to zero
      b[j] = 0;
      // Point all the preceding gap samples to trash pixel and set them to zero in the TOD
      while(A->ll[j]!= -1){
        b[A->ll[j]] = 0;
        for(k=0;k<(A->nnz); k++){
          A->indices[A->ll[j]*(A->nnz)+k] = k-(A->nnz);
          A->values[A->ll[j]*(A->nnz)+k] = 0;
        }
        j = A->ll[j];
      }


      // Remove degenerate pixel from vpixBlock, lhits, and cond
      memmove(vpixBlock+i, vpixBlock+i+(A->nnz)*(A->nnz),(n*(A->nnz)-(A->nnz)*(A->nnz)-i)*sizeof(double));
      memmove(lhits+(int)i/((A->nnz)*(A->nnz)), lhits+(int)i/((A->nnz)*(A->nnz))+1, ((int)n/(A->nnz)-1-(int)i/((A->nnz)*(A->nnz)))*sizeof(int));
      memmove(cond+(int)i/((A->nnz)*(A->nnz)), cond+(int)i/((A->nnz)*(A->nnz))+1, ((int)n/(A->nnz)-1-(int)i/((A->nnz)*(A->nnz)))*sizeof(double));


      // Shrink effective size of vpixBlock
      n -= (A->nnz);
      i -= (A->nnz)*(A->nnz);
    }
    uncut_pixel_index += (A->nnz)*(A->nnz);
  }
  
  // free memory
  free(A->id0pix);
  free(A->ll);
  // Reallocate memory for preconditioner blocks and redefine pointing matrix in case of the presence of degenerate pixels
  if(A->trash_pix){
    // Reallocate memory of vpixBlock by shrinking its memory size to its effective size (no degenerate pixel)
    tmp2 = (double *) realloc(vpixBlock, n*(A->nnz)*sizeof(double));
    tmp1 = (int *) realloc(lhits, (int)n/(A->nnz)*sizeof(int));
    tmp3 = (double *) realloc(cond,(int)n/(A->nnz)*sizeof(double));
    if(tmp2 != NULL){
      vpixBlock = tmp2;
      lhits = tmp1;
      cond = tmp3;
    }
  }
  // map local indices to global indices in indices_cut
  for(i=0; i<m*A->nnz;i++){
    // switch to global indices
    if(A->indices[i]>=0) // only for no trash_pix
      A->indices[i] = A->lindices[A->indices[i]];
  }
  // free  memory of original pointing matrix and synchronize
  MatFree(A);
  // Define new pointing matrix (marginalized over degenerate pixels and free from gap samples)
  MatInit(A, m, A->nnz, A->indices, A->values, A->flag, MPI_COMM_WORLD);

  //Define Block-Jacobi preconditioner indices
  for(i=0;i<n;i++){
    for(j=0;j<(A->nnz);j++){
        indices_new[i*(A->nnz)+j] = A->lindices[(A->nnz)*(A->trash_pix)+(A->nnz)*((int)i/(A->nnz))+j];
    }
  }

  //Init Block-Jacobi preconditioner
  MatSetIndices(BJ, n, A->nnz, indices_new);
  MatSetValues(BJ, n, A->nnz, vpixBlock);
  BJ->trash_pix = 0;
  MatLocalShape(BJ,3);

  return 0;

}

int TrMatVecProd_loc(Mat *A, double *y, double * x)
{
  int i, j, k, e;
    
  if(A->trash_pix){
    for(i=0; i < A->lcount-A->nnz; i++)				//refresh vector
      x[i]=0.0;						//

      e=0;
      for(i=0; i< A->m*A->nnz; i+=A->nnz){
        if(A->indices[i]!=0){
          //local transform reduce
          for(j=0; j< A->nnz; j++){
            x[A->indices[i+j]-(A->nnz)] += A->values[i+j] * y[e];	//
          }
        }							//
        e++;
      }
  }
  else{
    for(i=0; i < A->lcount; i++)				//refresh vector
      x[i]=0.0;						//

      e=0;
      for(i=0; i< A->m*A->nnz; i+=A->nnz){//local transform reduce
        for(j=0; j< A->nnz; j++){
          x[A->indices[i+j]] += A->values[i+j] * y[e];	//
        }						//
        e++;
      }
  }
  
  return 0;
}

// Build (A_i.T * A_i) bloc of size nnz*nnz operator
int Build_ATA_bloc(Mat *A, Tpltz Nm1, double *ATA, int row_indice, int nb_rows, int np, int rank, int bloc)
{
  int nnz;                              // Number of coef per line of A
  int i0,i1,i2,i3,i4,i5,i6,i7,i8;       // some loop index
  int locpix;                           // index of the pixel watch at time i0 in the row_indice block on the proc
  int info;                             // LAPACK return variable
  
  nnz = A->nnz;
  
  for (i0 = 0; i0 < nb_rows; ++i0) {

    locpix = A->indices[(row_indice+i0)*nnz]/nnz-(A->trash_pix);

    if (locpix >= 0) {
      // contribution of the pixel locpix to the block of size nnz*nnz of A_i.T*A_i at time row_indice+i0
      for (i1 = 0; i1 < nnz; ++i1) {
    	for (i2 = 0; i2 < nnz; ++i2) {
    	  ATA[locpix*nnz*nnz+i1*nnz+i2] += (A->values[(row_indice+i0)*nnz+i1])*(A->values[(row_indice+i0)*nnz+i2]);
    	}
      }      
      
      /*
    	When nnz = 3, it's equivalent to those line of code :
    	ATA[locpix*nnz*nnz] += A->values[locpix*nnz]*A->values[locpix*nnz];
    	ATA[locpix*nnz*nnz+1] += A->values[locpix*nnz]*A->values[locpix*nnz+1];
    	ATA[locpix*nnz*nnz+2] += A->values[locpix*nnz]*A->values[locpix*nnz+2];
    	ATA[locpix*nnz*nnz+3] += A->values[locpix*nnz+1]*A->values[locpix*nnz];
    	ATA[locpix*nnz*nnz+4] += A->values[locpix*nnz+1]*A->values[locpix*nnz+1];
    	ATA[locpix*nnz*nnz+5] += A->values[locpix*nnz+1]*A->values[locpix*nnz+2];
    	ATA[locpix*nnz*nnz+6] += A->values[locpix*nnz+2]*A->values[locpix*nnz];
    	ATA[locpix*nnz*nnz+7] += A->values[locpix*nnz+2]*A->values[locpix*nnz+1];
    	ATA[locpix*nnz*nnz+8] += A->values[locpix*nnz+2]*A->values[locpix*nnz+2];
      */
    }

  }
  
  double *tmp_blck;
  tmp_blck = (double *) malloc(nnz*nnz*sizeof(double));

  for (i3 = 0; i3 < np; ++i3) {

    if (ATA[i3*nnz*nnz] > nnz-1) {
      
      for (i4=0; i4 < nnz*nnz; ++i4) {
  	tmp_blck[i4] = ATA[i3*nnz*nnz+i4];
      }      
      
      // ###### Compute the norm 1 of tmp_blk
      double A_norm1;
      A_norm1 = LAPACKE_dlange(LAPACK_ROW_MAJOR,'1',nnz,nnz,tmp_blck,nnz);
      
      // ###### Compute the condition number of the block to know if we'll have to apply it in Apply_ATA_bloc
      double rcond;
      // info = LAPACKE_dpocon(LAPACK_ROW_MAJOR,'L',nnz,tmp_blck,nnz,A_norm1,&rcond);
      info = LAPACKE_dgecon(LAPACK_ROW_MAJOR,'1',nnz,tmp_blck,nnz,A_norm1,&rcond);

      /* printf("HERE -----> %f\n", rcond); */
      
      // ###### Copy the Cholesky factor in ATA if condition number not bad : rcond = 1/cond
      if (rcond > 1e-1) { // <=> cond < 10
	
  	// ###### Call LAPACK routines to compute Cholesky factorisation of tmp_blck
  	info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',nnz,tmp_blck,nnz);
	
  	if (info > 0) {
  	  // printf("The leading minor of order %i is not positive-definite, and the factorization could not be completed.",info);
  	  ATA[i3*nnz*nnz] = -1;
  	}
  	if (info < 0) {
  	  printf("Compute Cholesky factorisation of tmp_blck : The parameter %i had an illegal value.\n",-info);
  	}
  	if (info == 0) {
  	  // printf("The factorization went good.");

	  int good_enough = 1; // good_enough stays 1 if there are no small scalar on the diagonal of the Cholesky factor. LAPACK seems to fail to catch some of thoses cases (info>0), therefore, it's not good_enough yet.
	  for(i7 = 0; i7 < nnz; ++i7){
	    if (tmp_blck[i7*nnz+i7] < 1e-2) {
	      good_enough = 0;
	    }
	  }
	  
	  if (good_enough) {
	    for (i7 = 0; i7 < nnz; ++i7) {	    
	      for (i8 = 0; i8 < i7+1; ++i8) {
		ATA[i3*nnz*nnz+i7*nnz+i8] = tmp_blck[i7*nnz+i8];
	      }
	    }
	  }
	  else{
	    ATA[i3*nnz*nnz] = -1;
	  }
	}
      }
      else {
  	ATA[i3*nnz*nnz] = -1;
      }
    }
    else {
      ATA[i3*nnz*nnz] = -1;
    }
      
  }

  free(tmp_blck);
  
  return 0;
}

// Apply the product A_i.T * v for a timestream of one block
int Apply_ATr_bloc(Mat *A, double *x, double *y, int size_y, int row_indice, int nb_rows)
{
  int i0,i1;           // Some loop index
  int nnz;

  nnz = A->nnz;  
  for (i0 = 0; i0 < nb_rows; ++i0) {
    
    int pix_index;
    pix_index = A->indices[(row_indice+i0)*nnz]/nnz;
    
    for(i1 = 0; i1 < nnz; ++i1){
      
      // if there are trashed pixels and that we watch a pixel, we need to shift indices by 1 due to presence of fictitious pixel at index 0
      if (A->trash_pix == 1) {
  	if (pix_index =! 0) {
  	  // printf("%i \n",A->indices[(row_indice+i0)*nnz+i1]-nnz < size_y);
  	  y[A->indices[(row_indice+i0)*nnz+i1]-nnz] += (A->values[(row_indice+i0)*nnz+i1]) * x[i0];// 
  	}
      }
      else{
	// printf("%i \n",A->indices[(row_indice+i0)*nnz+i1] < size_y);
  	y[A->indices[(row_indice+i0)*nnz+i1]] += (A->values[(row_indice+i0)*nnz+i1]) * x[i0]; // 
      }
    }
  }
  
  return 0;
}


// Apply the product (A_i.T * A_i)^{-1} * v for a timestream of one block
int Apply_ATA_bloc(Mat *A, double *ATA, double *y, double *z, int np)
{
  int i0,i1,i2,i3,i4;  // Loop index
  int nnz;             // Number of non-zeros per line of A
  double *tmp_blck;    // Temporary block of the block-operator A_i.T * A_i of size nnz*nnz
  double *tmp_vec;     // Temporary vector (A_i.T * A_i)^{-1} * v of size nnz
  int info;            // LAPACK return variable

  nnz = A->nnz;
  
  tmp_blck = (double *) calloc(sizeof(double),nnz*nnz);
  tmp_vec = (double *) calloc(sizeof(double),nnz);

  if (tmp_blck == NULL || tmp_vec ==NULL) {
    printf("STOP, allocation went wrong\n");
    fflush(stdout);
  }

  for (i0 = 0; i0 < np; ++i0) {

    for (i1 = 0; i1 < nnz; ++i1) {

      if (isnan(y[i0*nnz+i1]) == 1 || isfinite(y[i0*nnz+i1]) == 0) {
	printf("PROBLEM of nan = %i, or finite = %i\n",isnan(y[i0*nnz+i1]),isfinite(y[i0*nnz+i1]));
	fflush(stdout);
      }
      
      tmp_vec[i1] = y[i0*nnz+i1];
      for (i2 = 0; i2 < nnz; ++i2) {
	tmp_blck[i1*nnz+i2] = ATA[i0*nnz*nnz+i1*nnz+i2];
      }
    }
      
    double one;
    one = 1;

    if (tmp_blck[0] > nnz-1) {

      int nrhs;
      nrhs = 1;
          
      // ###### Apply (P_i.T * P_i)^{-1} * v using the cholesky factor
      info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR,'L',nnz,nrhs,tmp_blck,nnz,tmp_vec,1);

      if (info < 0) {
      	printf("Apply (P_i.T * P_i)^{-1} * v using the cholesky factor :\nThe parameter %i had an illegal value.\n.",-info);
      }
      if (info == 0) {

      	/* printf("The multiplication went good."); */

      	// ###### Copy the results in output of the function z
      	for (i3 = 0; i3 < nnz; ++i3) {
      	  z[i0*nnz+i3] = tmp_vec[i3];
      	}
      }
      
    }
    else {
      for (i4 = 0; i4 < nnz; ++i4) {
    	z[i0*nnz+i4] = 0; //y[i0*nnz+i4];
      }
    }
  }

  free(tmp_blck);
  free(tmp_vec);
  
  return 0;
}

// Get the fourier mode of order index_mode through fftw3 package
int get_Fourier_mode(fftw_complex *in, fftw_complex *out, int size, int index_mode)
{
  int i0;
  
  for (i0 = 0; i0 < size; ++i0) {

    in[i0][0] = (i0==index_mode);
    in[i0][1] = 0;

  }
  
  fftw_plan plan_forw;
  plan_forw = fftw_plan_dft_1d(size,in,out,FFTW_FORWARD,FFTW_ESTIMATE);

  fftw_execute(plan_forw); // Execute FFT to get Fourier mode n°i1

  return 0;
}

// Build the coarse space from eigenvector of the blocks
int Build_ALS(Mat *A, Tpltz Nm1, double *CS, int nb_defl, int n, int rank)
{
  int nb_blocks_loc;                  // Number of local bloc, ie on a the proc
  Block *tpltzblocks;                 // Pointer to the tpltzblocks struct
  int nti;                            // size of the time stream
  int row_indice;                     // Index of the first line in depointing matrix A for the bloc we look at
  int nnz;                            // Number of non-zeros per line in A
  int np;                             // number of pixels   
  double *ATA;                        // Array to store the (A_i.T * A_i) block operator
  /* fftw_complex *in;                   // Vector to store entry of the FFT                                                    */
  /* fftw_complex *out;                  // Vector to store output of the FFT */
  // double *diagNm1;                    // Vector to store the diagonal of Nm1
  double *x;                          // Vector to store eigen vector w.r.t to one bloc
  // double *xx;
  double *y;                          // Vector to store A_i.T * x
  double *z;                          // Vector to store (A_i.T * A_i)^{-1} * y                                             
  int i0,i1,i2,i3,i4;                 // loop indices
  a_int ID0 = 0;
  a_int N;
  a_int NEV = nb_defl;
  char *BMAT = "I";
  int invert = 0;
  char *WHICH;
  if (invert == 0) {
    WHICH = "SM";

    /* printf("WHICH = SM\n"); */
  }
  else {
    WHICH = "LM";
    
    /* printf("WHICH = LM\n"); */
  }
  /* fflush(stdout); */
  
  double TOL = 1e-6;
  double *RESID;
  a_int NCV;
  double *V;
  a_int LDV;
  a_int *IPARAM = (a_int *) calloc(11,sizeof(a_int));
  a_int ISHIFT = 1;
  a_int maxiter_arnoldi = 10;
  a_int NB = 1;
  a_int MODE = 1;
  IPARAM[0] = ISHIFT;
  IPARAM[2] = maxiter_arnoldi;
  IPARAM[3] = NB;
  IPARAM[6] = MODE;
  a_int *IPNTR = (a_int *) malloc(sizeof(a_int)*11);
  double *WORKD;
  double *WORKL;
  a_int lworkl;
  a_int INFO = 0;
  double norm_residual;
  double norm_residual_0;

  /* if (rank == 0) { */
  /*   printf("ARPACK computes %i eigenvectors\n", NEV); */
  /*   fflush(stdout); */
  /* } */

  /* (RVEC, HOWMANY, SELECT, D, Z, SIGMA, BMAT, N, WHICH, NEV, TOL, RESID, NCV, V, IPARAM, IPNTR, WORKD, WORKL, INFO) */

/* extern void dsaupd_(int *ido, char *bmat, int *n, char *which, */
/*             int *nev, double *tol, double *resid, int *ncv, */
/*             double *v, int *ldv, int *iparam, int *ipntr, */
/*             double *workd, double *workl, int *lworkl, */
/*             int *info); */

/* extern void dseupd_(int *rvec, char *All, int *select, double *d, */
/*             double *v, int *ldv, double *sigma, */
/*             char *bmat, int *n, char *which, int *nev, */
/*             double *tol, double *resid, int *ncv, double *tv, */
/*             int *tldv, int *iparam, int *ipntr, double *workd, */
/*             double *workl, int *lworkl, int *ierr); */

  a_int RVEC = 1;
  char *HOWMANY = "A";
  a_int *SELECT;
  double *D = (double *) calloc(NEV,sizeof(double));
  // double *D = (double *) calloc(1,sizeof(double));
  double *Z;
  a_int LDZ;
  double SIGMA;

  double *xx;
  int RCI_request;
  int *ipar;
  ipar = (int *) malloc(sizeof(int)*128);
  double *dpar;
  dpar = (double *) malloc(sizeof(double)*128);
  double *tmp;
  double *tmp2;
  double stopping_criterion = 0;
  double norm_b = 0;
  int CONVERGED = 0;
  double tol_cg = 1e-2;

  int vblock_size;
  int blocksize;
  fftw_complex *T_fft;
  fftw_complex *V_fft;
  double *V_rfft;
  fftw_plan plan_f;
  fftw_plan plan_b;
  int nfft;

  
  
  nb_blocks_loc = Nm1.nb_blocks_loc;
  tpltzblocks = Nm1.tpltzblocks;
  row_indice = 0;
  nnz = A->nnz;
  np = n/nnz;
  y = (double *) calloc(n,sizeof(double));
  z = (double *) malloc(n*sizeof(double));
  /* getlocDiagN(A,Nm1,diagNm1); */
  ATA = (double *)  calloc(np*nnz*nnz,sizeof(double));
    
  for (i0 = 0; i0 < nb_blocks_loc; ++i0) {
    
    nti = tpltzblocks[i0].n; // Size of the block

    /* printf("r: %i, block %i, T_block[0:9] = ", rank, i0); */
    /* for (i1 = 0; i1 < 10; ++i1) { */
    /*   printf("%f ,", tpltzblocks[i0].T_block[i1]); */
    /* } */
    /* printf("\n"); */
    /* fflush(stdout); */
    
    N = nti;
    LDV = nti;
    if (2*nb_defl+1 < N) {
      NCV = 2*nb_defl+1;
    }
    else {
      NCV = N;
    }
    if (NCV < 300) {
      NCV = 300;
    }

    /* printf("r: %i, nti : %i, ncv : %i\n", rank, nti,NCV); */
    /* fflush(stdout); */
    
    RESID = (double *) malloc(sizeof(double)*nti);
    V = (double *) malloc(sizeof(double)*LDV*NCV);
    WORKD = (double *) malloc(sizeof(double)*3*N);
    lworkl = NCV*(NCV+8);
    WORKL = (double *) malloc(sizeof(double)*lworkl);

    SELECT = (a_int *) malloc(sizeof(a_int)*NCV);
    Z = (double *) malloc(sizeof(double)*N*NEV);
    // Z = (double *) malloc(sizeof(double)*N);
    LDZ = N;
    
    x = (double *) malloc(nti*sizeof(double));
    xx = (double *) calloc(nti,sizeof(double));
    tmp = (double *) malloc(sizeof(double)*nti*4);
    tmp2 = (double *) malloc(sizeof(double)*nti);

    if (RESID == NULL || V == NULL || WORKD == NULL || WORKL == NULL || SELECT == NULL || Z == NULL || x == NULL) {
      printf("r: %i, allocation went south :\n", rank);
      printf("RESID == %i || V == %i || WORKD == %i || WORKL == %i || SELECT == %i || Z == %i || x == %i\n", (int) RESID, (int) V, (int) WORKD, (int) WORKL, (int) SELECT, (int) Z, (int) x);
      fflush(stdout);
      
    }

    // ###### Build ATA w.r.t. the local block i0
    Build_ATA_bloc(A,Nm1,ATA,row_indice,nti,np,rank,i0);

    
    /* in = (fftw_complex *) calloc(nti,sizeof(fftw_complex)); */
    /* out = (fftw_complex *) calloc(nti,sizeof(fftw_complex)); */

    double *T_block = (double *) malloc(sizeof(double)*tpltzblocks[i0].lambda);
    double d = tpltzblocks[i0].T_block[0];

    /* printf("rank : %i, d/2 : %f\n", rank,d/2); */
    /* fflush(stdout); */

    
    for (i1 = 0; i1 < tpltzblocks[i0].lambda; ++i1) {
      T_block[i1] = tpltzblocks[i0].T_block[i1]; // /d;
      // T_block[i1] = 0;
    }

    /* printf("r: %i, HERE !!!!!\n", rank); */
    /* fflush(stdout); */

    /* printf("r: %i, start test : blocksize %i, nfft %i\n", rank,blocksize,nfft); */
    /* fflush(stdout); */

    /* Tpltz my_block; */
    /* my_block.nrow = N; */
    /* my_block.m_cw = 1; */
    /* my_block.m_rw = 1; */

    /* Block *T; */
    /* T = (Block *) malloc(sizeof(Block)*1); */

    /* T->idv = 1; // tpltzblocks[i0].idv; */
    /* T->T_block = T_block; // tpltzblocks[i0].T_block; */
    /* T->lambda = 1; // tpltzblocks[i0].lambda; */
    /* T->n = tpltzblocks[i0].n; */
    
    /* my_block.tpltzblocks = T; */
    /* my_block.nb_blocks_loc = 1; */
    /* my_block.nb_blocks_tot = 1; */
    /* my_block.local_V_size = Nm1.local_V_size; */
    /* my_block.flag_stgy = Nm1.flag_stgy; */
    /* my_block.comm = Nm1.comm; */

    /* double *test; */
    /* test = (double *) calloc(N,sizeof(double)); */
    /* // test[0] = 1; */

    /* for (i1 = 0; i1 < N; ++i1) { */
    /*   test[i1] = 1; */
    /* } */
    
    /* stbmmProd(my_block,test); */
    
    /* stmm_core(&test,N,1,T_block,T_fft,blocksize,tpltzblocks[i0].lambda,V_fft,V_rfft,nfft,plan_f,plan_b,1,Nm1.flag_stgy.flag_nofft); */

    /* stmm_main(&test,N,1,0,N,T_block,T_fft,tpltzblocks[i0].lambda,V_fft,V_rfft,plan_f,plan_b,blocksize,nfft,Nm1.flag_stgy); */
    
    /* MPI_Barrier(A->comm); */

    /* printf("r: %i, my_block*test[0:5] = [%f,%f,%f,%f,%f]\n", rank,test[0],test[1],test[2],test[3],test[4]); */
    /* fflush(stdout); */

    dsaupd_(&ID0, BMAT, &N, WHICH, &NEV, &TOL, RESID, &NCV, V, &LDV, IPARAM, IPNTR, WORKD, WORKL, &lworkl, &INFO);
    
    /* x = &(WORKD[IPNTR[0]-1]); */


    tpltz_init(N, tpltzblocks[i0].lambda, &nfft, &blocksize, &T_fft, T_block, &V_fft, &V_rfft, &plan_f, &plan_b, Nm1.flag_stgy);
    
    while (ID0 == 1 || ID0 == -1) {

      /* printf("ID0 = %i\n", ID0); */
      /* fflush(stdout); */
 
      for (i1 = 0; i1 < nti; ++i1) {
	x[i1] = WORKD[IPNTR[0]-1+i1];
      }

      /* printf("r: %i, ID0 = %i\n", rank,ID0); */
      /* fflush(stdout); */

      if (invert == 0) { // We compute the smallest eigenvectors of the blocks

	/* printf("r: %i, We compute the smallest eigenvectors of the blocks\n", rank); */
	/* fflush(stdout); */
	
	/* stmm_core(&x,N,1,T_block,T_fft,blocksize,tpltzblocks[i0].lambda,V_fft,V_rfft,nfft,plan_f,plan_b,1,Nm1.flag_stgy.flag_nofft); */
	
	stmm_main(&x,N,1,0,N,T_block,T_fft,tpltzblocks[i0].lambda,V_fft,V_rfft,plan_f,plan_b,blocksize,nfft,Nm1.flag_stgy);
	
	/* stbmmProd(my_block,x); */
	
	for (i1 = 0; i1 < N; ++i1) {
	  if (IPNTR[1]-1+i1 >= 3*N) {
	    printf("r: %i, PB ARRAY SIZE IPNTR[1]-1+i1=%i >= 3*N=%i\n", rank,IPNTR[1]-1+i1,3*N);
	    fflush(stdout);
	  }
	
	  WORKD[IPNTR[1]-1+i1] = x[i1];
	
	  if (ID0 == 1) {
	    WORKD[IPNTR[2]-1+i1] = WORKD[IPNTR[0]-1+i1];
	  }
	}
      }
      else{ // We compute the largest eigenvectors of the inverse of the blocks

	/* printf("r: %i, We compute the largest eigenvectors of the inverse of the blocks\n", rank); */
	/* fflush(stdout); */
	
	for (i1 = 0; i1 < N; ++i1) {
	  norm_b += x[i1]*x[i1];
	  xx[i1] = x[i1];
	}
	norm_b = sqrt(norm_b);

	/* printf("r: %i, norm rhs = %f\n", rank,norm_b); */
	/* fflush(stdout); */

	dcg_init(&N,xx,x,&RCI_request,ipar,dpar,tmp);


	ipar[0] = N;                                         // Specifies the size of the problem
	ipar[3] = 0;                                         // contains the current number of iteration
	ipar[4] = 5000;                                      // Set the maximum number of iteration in CG
	ipar[5] = 1;                                         // print all the error messages                not 0 means do it
	ipar[6] = 1;                                         // print all the warning messages              not 0 means do it
	ipar[7] = 1;                                         // Max iter. test,                             not 0 means do it
	ipar[8] = 1;                                         // Residual test,                              not 0 means do it
	ipar[9] = 0;                                         // User specific criterion stop test,          not 0 means do it
	
	dpar[0] = tol_cg;                                    // Relative tolerance
	dpar[1] = 0;                                         // abs. tolerance
	dpar[2] = norm_b;                                    // specifies the norm-2 of the initial residual
	
	dcg_check(&N,xx,x,&RCI_request,ipar,dpar,tmp);
	
	if (RCI_request == 0) {
	  
	  RCI_request = 1;

	  while (RCI_request > 0 && CONVERGED == 0) {
	  
	    dcg(&N,xx,x,&RCI_request,ipar,dpar,tmp);

	    if (RCI_request == 1) {

	      for (i1 = 0; i1 < N; ++i1) {
	      tmp2[i1] = tmp[i1];
	      }
	    
	      stmm_main(&tmp2,N,1,0,N,T_block,T_fft,tpltzblocks[i0].lambda,V_fft,V_rfft,plan_f,plan_b,blocksize,nfft,Nm1.flag_stgy);

	      for (i1 = 0; i1 < N; ++i1) {
		tmp[N+i1] = tmp2[i1];
	      }


	    }
	    else if (RCI_request == 2) {

	      for (i1 = 0; i1 < N ; ++i1) {
		tmp2[i1] = xx[i1];
	      }
	    
	      stmm_main(&tmp2,N,1,0,N,T_block,T_fft,tpltzblocks[i0].lambda,V_fft,V_rfft,plan_f,plan_b,blocksize,nfft,Nm1.flag_stgy);

	      stopping_criterion = 0;

	      for (i1 = 0; i1 < N; ++i1) {
		stopping_criterion += (tmp2[i1]-x[i1])*(tmp2[i1]-x[i1]);
	      }
	    
	      stopping_criterion = sqrt(stopping_criterion)/norm_b;

	      if (stopping_criterion < tol_cg) {
		CONVERGED = 1;
		
		printf("r: %i, CG has converged : rel_res = %f < tol = %f, in %i iteration\n", rank,stopping_criterion,dpar[0],ipar[3]);
	      
	      }
	      else {
		printf("r: %i, not there yet : %f > %f, iteration %i out of %i\n", rank,stopping_criterion,dpar[0],ipar[3],ipar[4]);
		fflush(stdout);
	      }
	      
	      
	    }
	    else if (RCI_request < 0){
	      printf("r: %i, Problem with CG, RCI_resquest = %i, res = %f < (tol = %f) * (res_ini = %f) = %f in %i iteration out of %i \n", rank,RCI_request,dpar[4],dpar[0],dpar[2],dpar[0]*dpar[2], ipar[3],ipar[4]);
	      fflush(stdout);
	    }
	  }

	  if (RCI_request == 0) {
	    // printf("r: %i, CG has converged : rel_res = %f < tol = %f, in %i iteration out of %i\n", rank,dpar[4],dpar[0]/dpar[2],dpar[0],ipar[3],ipar[4]);
	  }
	  else {
	    printf("r: %i, CG has NOT converged : rel_res = %f < tol = %f, in %i iteration out of %i, rci_request = %i\n", rank,dpar[4]/dpar[2],dpar[0],ipar[3],ipar[4],RCI_request);
	  }
	  fflush(stdout);
	  
	  
	  
	}
	else {
	  printf("r: %i, PROBLEM with dcg_init/dcg_check RCI_request = %i\n", rank,RCI_request);
	  fflush(stdout);
	}
	
	
	
	/* stmm_main(&x,N,1,0,N,T_block,T_fft,tpltzblocks[i0].lambda,V_fft,V_rfft,plan_f,plan_b,blocksize,nfft,Nm1.flag_stgy); */
      
	/* stbmmProd(my_block,x); */
      
	for (i1 = 0; i1 < N; ++i1) {
	  if (IPNTR[1]-1+i1 >= 3*N) {
	    printf("r: %i, PB ARRAY SIZE IPNTR[1]-1+i1=%i >= 3*N=%i\n", rank,IPNTR[1]-1+i1,3*N);
	    fflush(stdout);
	  }
	
	  WORKD[IPNTR[1]-1+i1] = xx[i1];
	
	  /* if (ID0 == 1) { */
	  /*   WORKD[IPNTR[2]-1+i1] = WORKD[IPNTR[0]-1+i1]; */
	  /* } */
	}
 
      }
 
      dsaupd_(&ID0, BMAT, &N, WHICH, &NEV, &TOL, RESID, &NCV, V, &LDV, IPARAM, IPNTR, WORKD, WORKL, &lworkl, &INFO);
      
    }    
    
    tpltz_cleanup(&T_fft,&V_fft,&V_rfft,&plan_f,&plan_b);

    /* if (INFO == 0 || (INFO == 1 && IPARAM[2] != maxiter_arnoldi)) { // everything went good with ARPACK */
      
    /*   for (i1 = 0; i1 < IPARAM[4]; ++i1) { // IPARAM[4] = nb of eigenvalue/eigenvector found by ARPACK */

    /* 	for (i2 = 0; i2 < NCV; ++i2) { */
    /* 	  if (i2 == i1) { */
    /* 	    SELECT[i2] = 1; */
    /* 	  } */
    /* 	  else { */
    /* 	    SELECT[i2] = 0; */
    /* 	  } */
    /* 	} */

    /* 	dseupd_(&RVEC, HOWMANY, SELECT, D, Z, &LDZ, &SIGMA, BMAT, &N, WHICH, &NEV, &TOL, RESID, &NCV, V, &LDV, IPARAM, IPNTR, WORKD, WORKL, &lworkl, &INFO); */

    /* 	if (INFO < 0) { */
    /* 	  printf("PROBLEM dsEupd on r: %i, INFO = %i, ITER = %i, EV found = %i\n", rank,INFO,IPARAM[2],IPARAM[4]); */
    /* 	  fflush(stdout); */
    /* 	} */
    /* 	else { */

    /* 	  for (i2 = 0; i2 < n; ++i2) { */
    /* 	    y[i2] = 0; */
    /* 	    z[i2] = 0; */
    /* 	  } */
	
    /* 	  // ###### Call of the function to do the local pointing : pointing of 1 block       */
    /* 	  Apply_ATr_bloc(A,Z,y,n,row_indice,nti); // Compute P_i^\top * eigenvector i1 */
      
    /* 	  // ###### Do the (A_i^T*A_i)^\dagger local product */
    /* 	  Apply_ATA_bloc(A,ATA,y,z,np); */
      
    /* 	  // ###### Store the resulting vector in coarse space array CS */
    /* 	  for (i3 = 0; i3 < n; ++i3) { */
	  
    /* 	    if (i0*nb_defl*n+i1*n+i3 >= n*nb_defl*nb_blocks_loc) { */
    /* 	      printf("r: %i, PB SIZE ARRAY i0*nb_defl*n+i1*n+i3 = %i >= n*nb_defl*nb_blocks_loc = %i\n", rank,i0*nb_defl*n+i1*n+i3,n*nb_defl*nb_blocks_loc); */
    /* 	      fflush(stdout); */
    /* 	    } */
	  
    /* 	    CS[i0*nb_defl*n+i1*n+i3] = z[i3]; */
    /* 	  } */
      
    /* 	} */
    /*   }	 */
    /* } */
    /* else { */
    /*   printf("PROBLEM dsAupd on r: %i, INFO = %i, ITER = %i, EV found = %i\n", rank,INFO,IPARAM[2],IPARAM[4]); */
    /*   fflush(stdout); */
    /* } */

    if (INFO == 0 || (INFO == 1 && IPARAM[4] != maxiter_arnoldi)) {
      
      dseupd_(&RVEC, HOWMANY, SELECT, D, Z, &LDZ, &SIGMA, BMAT, &N, WHICH, &NEV, &TOL, RESID, &NCV, V, &LDV, IPARAM, IPNTR, WORKD, WORKL, &lworkl, &INFO);

      if (INFO == 0) {
	
	/* printf("r: %i, block : %i, the eigenvalues are :\n", rank,i0); */
	/* for (i1 = 0; i1 < NEV; ++i1) { */
	/*   printf("%f ", D[i1]); */
	/* } */
	/* printf("\n"); */
      }
      else {
	printf("PROBLEM dseupd on r: %i, INFO = %i\n", rank,INFO);
      }
      fflush(stdout);
      
    }
    else {
      printf("PROBLEM dsaupd on r: %i, INFO = %i, ITER = %i, EV found = %i\n", rank,INFO,IPARAM[2],IPARAM[4]);
      fflush(stdout);
    }
    

    
    for (i1=0; i1<IPARAM[4]; ++i1) {
      
      /* // ###### Get the fourier mode of order i0 and take it as an eigenvector of the block */
      /* get_Fourier_mode(in,out,nti,i0); */

      /* for (i3 = 0; i3 < nti; ++i3) { */
      /* 	x[i3] = out[i3][0]; // Store the real part of the results of FFT in an array. */
      /* } */

      if (i1*nti >= NEV*N) {
	printf("r: %i, PB SIZE ARRAY i1*nti = %i >= NEV*N = %i\n", rank,i1*nti,NEV*N);
	fflush(stdout);
      }

      for (i2 = 0; i2 < n; ++i2) {
	y[i2] = 0;
	z[i2] = 0;
      }

      
      // ###### Call of the function to do the local pointing : pointing of 1 block      
      Apply_ATr_bloc(A,&(Z[i1*nti]),y,n,row_indice,nti); // Compute P_i^\top * eigenvector i1

      // ###### Do the (A_i^T*A_i)^\dagger local product
      Apply_ATA_bloc(A,ATA,y,z,np);
      
      // ###### Store the resulting vector in coarse space array CS
      for (i4 = 0; i4 < n; ++i4) {
	if (i0*nb_defl*n+i1*n+i4 >= n*nb_defl*nb_blocks_loc) {
	  printf("r: %i, PB SIZE ARRAY i0*nb_defl*n+i1*n+i4 = %i >= n*nb_defl*nb_blocks_loc = %i\n", rank,i0*nb_defl*n+i1*n+i4,n*nb_defl*nb_blocks_loc);
	  fflush(stdout);
	}
    	CS[i0*nb_defl*n+i1*n+i4] = z[i4];
      }
      
    }

    
    free(RESID);
    free(V);
    free(WORKD);
    free(WORKL);
    free(SELECT);
    free(Z);
    free(T_block);
    free(x);
    free(xx);
    free(tmp);
    free(tmp2);
        
    row_indice += nti;

    ID0 = 0;
    
  }
    
  free(z);
  free(y);
  free(D);
  free(IPARAM);
  free(IPNTR);
  free(ATA);
  free(ipar);
  free(dpar);

  return 0;
}

// Build a orthonormal basis of a coarse space Z
int Orthogonalize_Space_loc(double **Z, int nb_rows, int nb_cols, double tol_svd, int rank)
{
  int i0,i1,i2,i3,i4;                              // Loop index
  double *tau;                                     // Coef. of the Householder reflection in the Q-factor
  int info;                                        // LAPACK return variable
  int size_CS;
  
  /* double *M; */
  /* int line = 4; */
  /* int col = 3; */
  /* M = (double *) calloc(line*col,sizeof(double)); */
  /* M[0] = 1; */
  /* M[1] = 0; */
  /* M[2] = 1; */
  /* M[3] = 0; */
  /* M[4] = 0; */
  /* M[5] = 1; */
  /* M[6] = 0; */
  /* M[7] = 1; */
  /* M[8] = 2; */
  /* M[9] = 0; */
  /* M[10] = 2; */
  /* M[11] = 0; */

  /* if (rank == 0) { */
  /*   printf("Adress of Z :\n"); */
  /* } */
  
  /* printf("r: %i, %p\n", rank,*Z); */
  /* fflush(stdout); */
  
  tau = (double *) malloc(sizeof(double)*nb_cols);

  /* tau = (double *) malloc(sizeof(double)*col); */
  
  /* // Compute non-pivoting QR factorization of coarse space Z */
  /* info = LAPACKE_dgeqrfp(LAPACK_COL_MAJOR,line,col,M,line,tau); */

  // Compute non-pivoting QR factorization of coarse space Z
  info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR,nb_rows,nb_cols,*Z,nb_rows,tau);

  if (info < 0) {
    printf("r: %i - Compute non-pivoting QR factorization of coarse space Z : The parameter %i had an illegal value.\n",rank,-info);
  }
  
  /* for (i0 = 0; i0 < 8; ++i0) { */
  /*   printf("%f \n", M[i0]); */
  /* } */

  /* printf("r: %i, R from M :\n", rank); */

  /* for (i0 = 0; i0 < col; ++i0) { */
  /*   for (i1 = 0; i1 < col; ++i1) { */
  /*     printf("%f ", M[i0+i1*line]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  
  // ###### Extract R from data in COL_MAJOR
  double *R;
  R = (double *) calloc(nb_cols*nb_cols,sizeof(double));

  /* // ###### Extract R from data in COL_MAJOR */
  /* double *R; */
  /* R = (double *) calloc(col*col,sizeof(double)); */

  double test = 0;

  /* printf("r: %i\n", rank); */

  /* printf("r: %i ----> Here !!!\n",rank); */
  /* fflush(stdout); */

    
  for (i0 = 0; i0 < nb_cols; ++i0) {
    // test += (*Z)[i0+i0*nb_rows];
    // printf("%f ", (*Z)[i0+i0*nb_rows]);
    for (i1 = 0; i1 < i0+1; ++i1) {
      if ((i1+i0*nb_cols < nb_cols*nb_cols ) && (i1+i0*nb_rows < nb_rows*nb_cols)) {
	// printf("r: %i, NO PROBLEM\n", rank);
	// fflush(stdout);
	R[i1+i0*nb_cols] = 1; // (*Z)[i1+i0*nb_rows];
      }
      else {
	printf("r: %i, PROBLEM\n", rank);
	fflush(stdout);
      }
    }
  }
  
  /* printf("r: %i, %f\n", rank, test/nb_cols); */
  /* fflush(stdout); */


  /* printf("\n"); */
  
  /* for (i0 = 0; i0 < col; ++i0) { */
  /*   // test += (*Z)[i0+i0*nb_cols]; */
  /*   for (i1 = 0; i1 < i0+1; ++i1) { */
  /*     R[i1+i0*col] = M[i1+i0*line]; */
  /*   } */
  /* } */

  /* printf("r: %i\n", rank); */

  /* printf("r: %i, R from R :\n", rank); */
  
  /* for (i0 = 0; i0 < col; ++i0) { */
  /*   for (i1 = 0; i1 < col; ++i1) { */
  /*     printf("%f ", R[i0+i1*col]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  /* if (rank == 1) { */
  /*   printf("r: %i\n",rank); */
    
  /*   for (i0 = 0; i0 < nb_cols; ++i0) { */
  /*     printf("%f ",R[i0*nb_cols+i0]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  
  /* return 0; */

  /* printf("\n"); */
  
  /* printf("HERE BEFORE------> r: %i, %f\n", rank,test); */
  
  // ###### Compute a RR-QR on the R-factor
  int *jpvt;
  jpvt = (int *) calloc(nb_cols,sizeof(int));
  double *tau_tmp;
  tau_tmp = (double *) malloc(sizeof(double)*nb_cols);
  
  info = LAPACKE_dgeqp3(LAPACK_COL_MAJOR,nb_cols,nb_cols,R,nb_cols,jpvt,tau_tmp);

  /* printf("r: %i\n", rank); */
  /* for (i2 = 0; i2 < nb_cols; ++i2) { */
  /*   printf("%i ",jpvt[i2]); */
  /* } */
  /* printf("\n"); */
  /* fflush(stdout); */


  /* test = 0; */
  
  /* for (i0 = 0; i0 < nb_cols; ++i0) { */
  /*   test += R[i0+i0*nb_cols]; */
  /* } */

  /* printf("HERE AFTER ------> r: %i, %f\n", rank,test); */

  if (info < 0) {
    printf("r: %i - Compute a RR-QR on the R-factor : The parameter %i had an illegal value.\n",rank,-info);
  }
  
  free(tau_tmp);
  
  // ###### Get the Q-factor from Z
  info = LAPACKE_dorgqr(LAPACK_COL_MAJOR,nb_rows,nb_cols,nb_cols,*Z,nb_rows,tau);

  if (info < 0) {
    printf("r: %i - Get the Q-factor from Z : The parameter %i had an illegal value.\n",rank,-info);
  }

  free(tau);
  
  // ###### Select the colums of the Q-factor related to high enough singular values
  size_CS = 0;

  /* if (rank == 1) { */
  /*   printf("r: %i\n",rank); */
    
  /*   for (i0 = 0; i0 < nb_cols; ++i0) { */
  /*     printf("%f ",R[i0*nb_cols+i0]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  
  while ( (abs(R[size_CS*nb_cols+size_CS]) > tol_svd) && (size_CS<nb_cols) ) {
    // printf("%i ", (R[size_CS*nb_cols+size_CS]>tol_svd)* (size_CS<nb_cols));
    size_CS += 1;
  }

  /* printf("r: %i, %i\n", rank, size_CS); */
  /* fflush(stdout); */
  
  // printf("\n");
  // printf("%i ", (R[size_CS*nb_cols+size_CS]>tol_svd)* (size_CS<nb_cols));
  
  free(R);

  double *CS;
  
  CS = (double *) malloc(sizeof(double)*nb_rows*(size_CS));

  /* if (rank == 0) { */
  /*   printf("Adress of Z :\n"); */
  /* } */

  /* printf("r: %i, %p\n", rank,CS); */
  /* fflush(stdout); */
  
  /* printf("r: %i, nb_cols: %i\n",rank,nb_cols); */
  /* fflush(stdout); */
  
  for (i2 = 0; i2 < size_CS; ++i2) {
    // printf("r: %i, jpvt : %i\n", rank,jpvt[i2]-1);
    // fflush(stdout);
    for (i3 = 0; i3 < nb_rows; ++i3) {
      if ((i2*nb_rows+i3 < nb_rows*size_CS) && ((jpvt[i2]-1)*nb_rows+i3 < nb_rows*nb_cols) && (i2 < nb_cols)) {
	// printf("r: %i, NO PROBLEM\n", rank);
	// fflush(stdout);
	CS[i2*nb_rows+i3] = (*Z)[(jpvt[i2]-1)*nb_rows+i3];
      }
      else {
	printf("r: %i, PROBLEM\n", rank);
	fflush(stdout);
      }
    }
  }

  free(jpvt);
  /* free(*Z); */

  /* &*Z = &CS; */

  /* test = 0; */
  /* for (i2 = 0; i2 < size_CS; ++i2) { */
  /*   for (i3 = 0; i3 < nb_rows; ++i3) { */
  /*     test += CS[i2*nb_rows+i3]; */
  /*   } */
  /* } */

  /* double mean = test/(size_CS*nb_rows); */
  /* double variance = 0; */
  /* for (i2 = 0; i2 < size_CS; ++i2) { */
  /*   for (i3 = 0; i3 < nb_rows; ++i3) { */
  /*     variance += (mean-CS[i2*nb_rows+i3])*(mean-CS[i2*nb_rows+i3]); */
  /*   } */
  /* } */
  
  /* printf("r: %i, mean %f, var %f\n", rank,mean,sqrt(variance)); */


  
  /* for (i2 = 0; i2 < size_CS; ++i2) { */
  /*   for (i3 = 0; i3 < nb_rows; ++i3) { */
  /*     (*Z)[i2*nb_rows+i3] = 1; */
  /*     CS[i2*nb_rows+i3] = 1; */
  /*   } */
  /* } */
  
  /* for (i4 = 0; i4 < nb_rows*(size_CS); ++i4) { */
  /*   (*Z)[i4] = CS[i4]; */
  /* } */

  /* printf("r: %i CHECKING\n",rank); */
  /* int i; */
  /* for (i = 0; i < size_CS*nb_rows; ++i) { */
  /*   if (isnan((*Z)[i]) || (1-isfinite((*Z)[i])) ) { */
  /*     printf("FUCK !!?!?\n"); */
  /*   } */
  /* } */
  /* printf("SOMETHING\n"); */
  
  /* free(*Z); */
  
  /* *Z = (double *) realloc(*Z,nb_rows*(size_CS)); */

  /* free(CS); */

  *Z = CS; // *arg1 = CS;
  
  /* printf("The size of the coarse space on the proc %i is %i\n",rank,size_CS); */

  return size_CS;
  
}

// Get the total size of the coarse over all the processes
int Communicate_sizes(int new_size, MPI_Comm comm){

  int nb_proc;                         // Number of proc
  int rank;                            // Id of the proc
  int *size_on_proc;                   // Array of the size of a coarse space on a proc before reduce
  int *size_on_proc_reduce;            // Array of the size of a coarse space on a proc after reduce
  int tot_size_CS = 0;                 // Total size of the coarse space
  int i0;                              // Some loop index_mode
  
  MPI_Comm_size(comm,&nb_proc);
  MPI_Comm_rank(comm,&rank);

  // Array of zeros containing the size of the CS of proc rank at position rank
  size_on_proc = (int *) calloc(nb_proc,sizeof(int));
  size_on_proc[rank] = new_size;

  // size_on_proc_reduce is the sum of the size_on_proc array element wise
  size_on_proc_reduce = (int *) calloc(nb_proc,sizeof(int));
  MPI_Allreduce(size_on_proc,size_on_proc_reduce,nb_proc,MPI_INT,MPI_SUM,comm);

  free(size_on_proc);
  
  // printf("r: %i\n",rank);

  // Compute the total size of the CS
  for (i0 = 0; i0 < nb_proc; ++i0) {
    // printf("%i ", size_on_proc_reduce[i0]);
    tot_size_CS += size_on_proc_reduce[i0];
  }

  return tot_size_CS;

}


// Communicate the coarse space array between the procs to build the full coarse space
double * Communicate_CS(double *Z, int new_size, int *tot_size_CS, MPI_Comm comm, int nb_rows){
  
  int nb_proc;                         // Number of proc
  int rank;                            // Id of the proc
  int *size_on_proc;                   // Array of the size of a coarse space on a proc before reduce
  int *size_on_proc_reduce;            // Array of the size of a coarse space on a proc after reduce
  // int tot_size_CS = 0;                 // Total size of the coarse space
  double *CS;                          // Array of the total coarse space
  int *location;                       // Array of where to write in CS for the procs
  int *recvcount;                      // Integer array (of length nb_proc) containing the number of elements that are received from each process
  int i0;                              // Some loop index

  MPI_Comm_size(comm,&nb_proc);
  MPI_Comm_rank(comm,&rank);

  // Array of zeros containing the size of the CS of proc rank at position rank
  size_on_proc = (int *) calloc(nb_proc,sizeof(int));
  size_on_proc[rank] = new_size;

  // size_on_proc_reduce is the sum of the size_on_proc array element wise
  size_on_proc_reduce = (int *) calloc(nb_proc,sizeof(int));
  MPI_Allreduce(size_on_proc,size_on_proc_reduce,nb_proc,MPI_INT,MPI_SUM,comm);

  free(size_on_proc);

  /* if(rank == 15) */
  /* printf("r: %i\n",rank); */

  *tot_size_CS = 0;
  
  // Compute the total size of the CS
  for (i0 = 0; i0 < nb_proc; ++i0) {
    // if(rank == 15)
    // printf("%i ", size_on_proc_reduce[i0]*nb_rows);
    *tot_size_CS += size_on_proc_reduce[i0];
  }

  /* if(rank == 15) */
  /* printf("\n"); */

  /* fflush(stdout); */
  
  /* MPI_Barrier(comm); */

  /* printf("r: %i, %i\n",rank,*tot_size_CS); */

  CS = (double *) malloc(sizeof(double)*nb_rows*(*tot_size_CS));

  location = (int *) calloc(nb_proc,sizeof(int));
  recvcount = (int *) calloc(nb_proc,sizeof(int));

  int tmp = 0;
  for (i0 = 1; i0 < nb_proc; ++i0) {
    location[i0] = size_on_proc_reduce[i0-1]*nb_rows + location[i0-1];
    // tmp = location[i0];
  }

  // if(rank == 15)
    // printf("r: %i\n",rank);
  
  for (i0 = 0; i0 < nb_proc; ++i0) {
    recvcount[i0] = size_on_proc_reduce[i0]*nb_rows;
    // if(rank == 15)
      // printf("%i ",location[i0]);
    // printf("%i ",recvcount[i0]/nb_rows);
  }

  /* if(rank == 15) */
  /* printf("\n"); */

  /* if (rank == 0) { */
  /*   printf("-----> HERE : r: %i\n",rank); */
  /* } */
  
  /* fflush(stdout); */

  /* MPI_Barrier(comm); */
  
  MPI_Allgatherv(Z,new_size*nb_rows,MPI_DOUBLE,CS,recvcount,location,MPI_DOUBLE,comm);
  
  /* int i; */
  /* if (rank == 0) { */
  /*   for (i = 0; i < (*tot_size_CS)*nb_rows; ++i) { */
  /*     printf("%f ",CS[i]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  free(size_on_proc_reduce);
  free(recvcount);
  free(location);

  /* printf("r: %i, %p\n",rank,CS); */

  /* Z = CS; */
  
  return CS;
}

/* // Compute the Cholesky factor of the Coarse matrix E (ref. Tang, Nabben, Vuik, Erlan notation) */
/* int Factorize_CS(Mat *A, Tpltz Nm1, double *Z, int nb_rows, int nb_cols, double *E){ */

/*   double *z;                          // Array to store vector a column of Z */
/*   double *Az;                         // Array to store vector the result of depointing A*z */
/*   double *NAz;                        // Array to store vector N*A*z */
/*   double *ANAz;                       // Array to store vector A.T*N*A*z */
/*   double *AZ;                         // Array to store matrix A.T.N.A*Z */

/*   for (i0 = 0; i0 < nb_cols; ++i0) { */
    
/*   } */


/* } */

/* // Apply the ADEF1 2lvl preconditionner to a vector (ref. Tang, Nabben, Vuik, Erlan notation) */
/* int Apply_ADEF1(Mat BJ, Mat *A, Tpltz Nm1, double *CS, double *E, double *in, double *out){ */




/* } */














































int precondjacobilike_avg(Mat A, Tpltz Nm1, double *c)
{

  int           i, j, k ;                       // some indexes
  int           m, n, rank, size;
  double        localreduce;                    //reduce buffer
  double        st, t;                          //timers

  m=A.m;                                        //number of local time samples
  n=A.lcount;                                   //number of local pixels
  MPI_Comm_rank(A.comm, &rank);                 //
  MPI_Comm_size(A.comm, &size);                 //

  double diagNm1;


//Compute diag( AtA )
  DiagAtA(&A, c, 2);

//multiply by the diagonal Toeplitz
  diagNm1 = Nm1.tpltzblocks[0].T_block[0];

  printf("diagNm1 = %f \n", diagNm1 );
  for(j=0; j<n; j++)
    c[j] = diagNm1 * c[j];

// compute c inverse vector
  for(j=0; j<n; j++)
    c[j] = 1./c[j] ;


  return 0;
}




int precondjacobilike(Mat A, Tpltz Nm1, int *lhits, double *cond, double *vpixDiag)
{

  int           i, j, k ;                       // some indexes
  int           m, n, rank, size;
  double        localreduce;                    //reduce buffer
  double        st, t;                          //timers

  MPI_Comm_rank(A.comm, &rank);                 //
  MPI_Comm_size(A.comm, &size);                 //

  m=A.m;                                        //number of local time samples
  n=A.lcount;                                   //number of local pixels

//Compute local diag( At diag(N^1) A )
  getlocDiagN(&A, Nm1, vpixDiag);

/*
  for(i=0; i<10; i++)
    printf("rank=%d, vpixDiag[%d]=%lf \n", rank, i, vpixDiag[i]) ;

    printf("rank=%d, vpixDiag[n-1]=%lf \n", rank, vpixDiag[n-1]);
*/

//communicate with the other processes to have the global reduce
  commScheme(&A, vpixDiag, 2);
  // for(i=0;i<50;i++){
  //   printf("global AtA block: vpixDiag[%d]=%f\n",i,vpixDiag[i]);
  // }
// compute the inverse vector
  for(i=0; i<n; i++){
    if(i%3 == 0){
      lhits[(int)i/3] = (int)vpixDiag[i];
      cond[(int)i/3] = vpixDiag[i+1] + vpixDiag[i+2];
    }
    vpixDiag[i] = 1./vpixDiag[i] ;
  }
  return 0;
}

//do the local Atdiag(Nm1)A with as output a block-diagonal matrix (stored as a vector) in the pixel domain
int getlocalW(Mat *A, Tpltz Nm1, double *vpixBlock, int *lhits)
{
  int           i, j, k, l ;                       // some indexes
  int           m;
  
  m=Nm1.local_V_size;  //number of local time samples
  int nnz=(A->nnz);
  
  //Define the indices for each process
  int idv0, idvn;  //indice of the first and the last block of V for each processes
  int *nnew;
  nnew = (int*) calloc(Nm1.nb_blocks_loc, sizeof(int));
  int64_t idpnew;
  int local_V_size_new;
  //get idv0 and idvn
  get_overlapping_blocks_params( Nm1.nb_blocks_loc, Nm1.tpltzblocks, Nm1.local_V_size, Nm1.nrow, Nm1.idp, &idpnew, &local_V_size_new, nnew, &idv0, &idvn);
  // double *vpixDiag;
  // vpixDiag = (double *) malloc(A->lcount *sizeof(double));

  int istart, il, istartn;
  for(i=0; i < nnz * A->lcount; i++)
    vpixBlock[i]=0.0;//0.0;

  int vShft=idpnew-Nm1.idp; //=Nm1.tpltzblocks[idv0].idv-Nm1.idp in principle
  /*
    printf("Nm1.idp=%d, idpnew=%d, vShft=%d\n", Nm1.idp, idpnew, vShft);
    printf("idv0=%d, idvn=%d\n", idv0, idvn);
    printf("Nm1.nb_blocks_loc=%d, Nm1.local_V_size=%d\n", Nm1.nb_blocks_loc, Nm1.local_V_size);

    for(i=0; i < Nm1.nb_blocks_loc; i++)
    printf("Nm1.tpltzblocks[%d].idv=%d\n", i, Nm1.tpltzblocks[i].idv);
  */
  
  //go until the first piecewise stationary period
  for(i=0;i<vShft;i++){
    lhits[(int)(A->indices[i*nnz]/nnz)] += 1;
    for(j=0;j<nnz;j++){
      for(k=0;k<nnz;k++){
	vpixBlock[nnz*A->indices[i*nnz+j]+k] += A->values[i*nnz+j]*A->values[i*nnz+k];
      }
    }
  }
  
  //temporary buffer for one diag value of Nm1
  double diagNm1;
  //loop on the blocks
  for(k=idv0; k<(idv0+Nm1.nb_blocks_loc); k++) {
  if (nnew[idv0]>0) {  //if nnew==0, this is a wrong defined block

    if (k+1<idv0+Nm1.nb_blocks_loc)   //if there is a next block, compute his next first indice
      istartn= Nm1.tpltzblocks[k+1].idv-Nm1.idp ;
    else
      istartn= Nm1.local_V_size;
      // istartn = 0;


    istart = max( 0, Nm1.tpltzblocks[k].idv-Nm1.idp);
    il = Nm1.tpltzblocks[k].n; // added this line to default code


    //if block cut from the left:
    if (k==idv0)
      il     = min( Nm1.tpltzblocks[k].n, Nm1.tpltzblocks[k].idv + Nm1.tpltzblocks[k].n - Nm1.idp );
    //if block cut from the right:
    if (k==idv0+Nm1.nb_blocks_loc-1)
      il     = min(il , (Nm1.idp + Nm1.local_V_size) - Nm1.tpltzblocks[k].idv );
    //if block alone in the middle, and cut from both sides
    if (Nm1.nb_blocks_loc==1)
      il     = min(il , Nm1.local_V_size);

    //get the diagonal value of the Toeplitz
    diagNm1 = Nm1.tpltzblocks[k].T_block[0];
/*
    printf("istart=%d, il=%d, istartn=%d\n", istart, il, istartn);
    printf("Nm1.tpltzblocks[k=%d].idv=%d, Nm1.tpltzblocks[k=%d].n=%d, Nm1.idp=%d\n", k, Nm1.tpltzblocks[k].idv, k, Nm1.tpltzblocks[k].n, Nm1.idp);
*/
//a piecewise stationary period
    for(i = istart; i<istart+il; i++){
      lhits[(int)(A->indices[i*nnz]/nnz)] += 1;
      for(j=0;j<nnz;j++){
        for(l=0;l<nnz;l++){
          vpixBlock[nnz*A->indices[i*nnz+j]+l] += A->values[i*nnz+j]*A->values[i*nnz+l]*diagNm1;
        }
      }
    }

//continue until the next period if exist or to the last line of V
    for(i = istart+il; i<istartn; i++){
      lhits[(int)(A->indices[i*nnz]/nnz)] += 1;
      for(j=0;j<nnz;j++){
        for(l=0;l<nnz;l++){
          vpixBlock[nnz*A->indices[i*nnz+j]+l] += A->values[i*nnz+j]*A->values[i*nnz+l];
        }
      }
    }

  }}//end of the loop over the blocks

  return 0;
}

//do the local diag( At diag(Nm1) A ) with as output a vector in the pixel domain
int getlocDiagN(Mat *A, Tpltz Nm1, double *vpixDiag)
{
  int           i, j, k ;                       // some indexes
  int           m;

  m=Nm1.local_V_size;  //number of local time samples
//  int nnz=(A->nnz);

  //Define the indices for each process
  int idv0, idvn;  //indice of the first and the last block of V for each processes
  int *nnew;
  nnew = (int*) calloc(Nm1.nb_blocks_loc, sizeof(int));
  int64_t idpnew;
  int local_V_size_new;
//get idv0 and idvn
  get_overlapping_blocks_params( Nm1.nb_blocks_loc, Nm1.tpltzblocks, Nm1.local_V_size, Nm1.nrow, Nm1.idp, &idpnew, &local_V_size_new, nnew, &idv0, &idvn);
 // double *vpixDiag;
 // vpixDiag = (double *) malloc(A->lcount *sizeof(double));

  int istart, il, istartn;
  for(i=0; i < A->lcount; i++)
    vpixDiag[i]=0.0;//0.0;

  int vShft=idpnew-Nm1.idp; //=Nm1.tpltzblocks[idv0].idv-Nm1.idp in principle
/*
  printf("Nm1.idp=%d, idpnew=%d, vShft=%d\n", Nm1.idp, idpnew, vShft);
  printf("idv0=%d, idvn=%d\n", idv0, idvn);
  printf("Nm1.nb_blocks_loc=%d, Nm1.local_V_size=%d\n", Nm1.nb_blocks_loc, Nm1.local_V_size);

  for(i=0; i < Nm1.nb_blocks_loc; i++)
    printf("Nm1.tpltzblocks[%d].idv=%d\n", i, Nm1.tpltzblocks[i].idv);
*/

//go until the first piecewise stationary period
    for (i= 0; i<vShft; i++) {
      for (j=0; j<(A->nnz); j++)
        vpixDiag[A->indices[i*(A->nnz)+j]]+=(A->values[i*(A->nnz)+j]*A->values[i*(A->nnz)+j]);
    }

//temporary buffer for one diag value of Nm1
  int diagNm1;
//loop on the blocks
  for(k=idv0; k<(idv0+Nm1.nb_blocks_loc); k++) {
  if (nnew[idv0]>0) {  //if nnew==0, this is a wrong defined block

    if (k+1<idv0+Nm1.nb_blocks_loc)   //if there is a next block, compute his next first indice
      istartn= Nm1.tpltzblocks[k+1].idv-Nm1.idp ;
    else
      istartn = 0;

    istart = max( 0, Nm1.tpltzblocks[k].idv-Nm1.idp);

    //if block cut from the left:
    if (k==idv0)
      il     = min( Nm1.tpltzblocks[k].n, Nm1.tpltzblocks[k].idv + Nm1.tpltzblocks[k].n - Nm1.idp );
    //if block cut from the right:
    if (k==idv0+Nm1.nb_blocks_loc-1)
      il     = min(il , (Nm1.idp + Nm1.local_V_size) - Nm1.tpltzblocks[k].idv );
    //if block alone in the middle, and cut from both sides
    if (Nm1.nb_blocks_loc==1)
      il     = min(il , Nm1.local_V_size);

    //get the diagonal value of the Toeplitz
    diagNm1 = Nm1.tpltzblocks[k].T_block[0];

/*
    printf("istart=%d, il=%d, istartn=%d\n", istart, il, istartn);
    printf("Nm1.tpltzblocks[k].idv=%d, Nm1.tpltzblocks[k].n=%d, Nm1.idp=%d\n", Nm1.tpltzblocks[k].idv, Nm1.tpltzblocks[k].n, Nm1.idp);
*/
//a piecewise stationary period
    for (i= istart; i<istart+il; i++) {
      for (j=0; j<(A->nnz); j++)
        vpixDiag[A->indices[i*(A->nnz)+j]]+=(A->values[i*(A->nnz)+j]*A->values[i*(A->nnz)+j])*diagNm1;
    }

//continue until the next period if exist
    for (i= istart+il; i<istartn; i++) {
      for (j=0; j<(A->nnz); j++)
        vpixDiag[A->indices[i*(A->nnz)+j]]+=(A->values[i*(A->nnz)+j]*A->values[i*(A->nnz)+j]);
    }


  }}//end of the loop over the blocks


  return 0;
}


//communication scheme in the pixel domain for the vector vpixDiag
//extract from a Madmap routine
int commScheme(Mat *A, double *vpixDiag, int pflag){
  int i, j, k;
  int nSmax, nRmax, nStot, nRtot;
  double *lvalues, *com_val, *out_val;

#if W_MPI
  lvalues = (double *) malloc((A->lcount-(A->nnz)*(A->trash_pix)) *sizeof(double));    /*<allocate and set to 0.0 local values*/
  memcpy(lvalues, vpixDiag, (A->lcount-(A->nnz)*(A->trash_pix)) *sizeof(double)); /*<copy local values into result values*/

  nRmax=0;
  nSmax=0;

  if(A->flag  == BUTTERFLY){                                  /*<branch butterfly*/
    //memcpy(out_values, lvalues, (A->lcount) *sizeof(double)); /*<copy local values into result values*/
    for(k=0; k< A->steps; k++)                                  /*compute max communication buffer size*/
      if(A->nR[k] > nRmax)
        nRmax = A->nR[k];
    for(k=0; k< A->steps; k++)
      if(A->nS[k] > nSmax)
        nSmax = A->nS[k];

    com_val=(double *) malloc( A->com_count *sizeof(double));
    for(i=0; i < A->com_count; i++){
      com_val[i]=0.0;
    }
//already done    memcpy(vpixDiag, lvalues, (A->lcount) *sizeof(double)); /*<copy local values into result values*/
    m2m(lvalues, A->lindices+(A->nnz)*(A->trash_pix), A->lcount-(A->nnz)*(A->trash_pix), com_val, A->com_indices, A->com_count);
    butterfly_reduce(A->R, A->nR, nRmax, A->S, A->nS, nSmax, com_val, A->steps, A->comm);
    m2m(com_val, A->com_indices, A->com_count, vpixDiag, A->lindices+(A->nnz)*(A->trash_pix), A->lcount-(A->nnz)*(A->trash_pix));
    free(com_val);
  }
  else if(A->flag == BUTTERFLY_BLOCKING_1){
    for(k=0; k< A->steps; k++)                                  //compute max communication buffer size
      if(A->nR[k] > nRmax)
        nRmax = A->nR[k];
    for(k=0; k< A->steps; k++)
      if(A->nS[k] > nSmax)
        nSmax = A->nS[k];
    com_val=(double *) malloc( A->com_count *sizeof(double));
    for(i=0; i < A->com_count; i++)
      com_val[i]=0.0;
    m2m(lvalues, A->lindices+(A->nnz)*(A->trash_pix), A->lcount-(A->nnz)*(A->trash_pix), com_val, A->com_indices, A->com_count);
    butterfly_blocking_1instr_reduce(A->R, A->nR, nRmax, A->S, A->nS, nSmax, com_val, A->steps, A->comm);
    m2m(com_val, A->com_indices, A->com_count, vpixDiag, A->lindices+(A->nnz)*(A->trash_pix), A->lcount-(A->nnz)*(A->trash_pix));
    free(com_val);
  }
  else if(A->flag == BUTTERFLY_BLOCKING_2){
    for(k=0; k< A->steps; k++)                                  //compute max communication buffer size
      if(A->nR[k] > nRmax)
        nRmax = A->nR[k];
    for(k=0; k< A->steps; k++)
      if(A->nS[k] > nSmax)
        nSmax = A->nS[k];
    com_val=(double *) malloc( A->com_count *sizeof(double));
    for(i=0; i < A->com_count; i++)
      com_val[i]=0.0;
    m2m(lvalues, A->lindices+(A->nnz)*(A->trash_pix), A->lcount-(A->nnz)*(A->trash_pix), com_val, A->com_indices, A->com_count);
    butterfly_blocking_1instr_reduce(A->R, A->nR, nRmax, A->S, A->nS, nSmax, com_val, A->steps, A->comm);
    m2m(com_val, A->com_indices, A->com_count, vpixDiag, A->lindices+(A->nnz)*(A->trash_pix), A->lcount-(A->nnz)*(A->trash_pix));
    free(com_val);
  }
  else if(A->flag == NOEMPTYSTEPRING){
    for(k=1; k< A->steps; k++)				//compute max communication buffer size
      if(A->nR[k] > nRmax)
        nRmax = A->nR[k];
    nSmax = nRmax;
    ring_noempty_step_reduce(A->R, A->nR, nRmax, A->S, A->nS, nSmax, lvalues, vpixDiag, A->steps, A->comm);
  }
  else if(A->flag == RING){
//already done    memcpy(vpixDiag, lvalues, (A->lcount) *sizeof(double)); /*<copy local values into result values*/
    for(k=1; k< A->steps; k++)                                  /*compute max communication buffer size*/
      if(A->nR[k] > nRmax)
        nRmax = A->nR[k];

    nSmax = nRmax;
    ring_reduce(A->R, A->nR, nRmax, A->S, A->nS, nSmax, lvalues, vpixDiag, A->steps, A->comm);
  }
  else if(A->flag == NONBLOCKING){
//already done    memcpy(vpixDiag, lvalues, (A->lcount) *sizeof(double)); /*<copy local values into result values*/
    ring_nonblocking_reduce(A->R, A->nR, A->S, A->nS, lvalues, vpixDiag, A->steps, A->comm);
  }
  else if(A->flag == NOEMPTY){
//already done    memcpy(vpixDiag, lvalues, (A->lcount) *sizeof(double)); /*<copy local values into result values*/
    int ne=0;
    for(k=1; k< A->steps; k++)
      if(A->nR[k]!=0)
        ne++;

    ring_noempty_reduce(A->R, A->nR, ne, A->S, A->nS, ne, lvalues, vpixDiag, A->steps, A->comm);
  }
  else if(A->flag == ALLREDUCE){
    com_val=(double *) malloc( A->com_count *sizeof(double));
    out_val=(double *) malloc( A->com_count *sizeof(double));
    for(i=0; i < A->com_count; i++){
      com_val[i]=0.0;
      out_val[i]=0.0;
    }
    s2m(com_val, lvalues, A->com_indices, A->lcount-(A->nnz)*(A->trash_pix));
    /*for(i=0; i < A->com_count; i++){
       printf("%lf ", com_val[i]);
    } */
    MPI_Allreduce(com_val, out_val, A->com_count, MPI_DOUBLE, MPI_SUM, A->comm);	//maximum index
    /*for(i=0; i < A->com_count; i++){
       printf("%lf ", out_val[i]);
    } */
    m2s(out_val, vpixDiag, A->com_indices, A->lcount-(A->nnz)*(A->trash_pix));                                 //sum receive buffer into values
    free(com_val);
    free(out_val);
  }
  else if(A->flag == ALLTOALLV){
    nRtot=nStot=0;
    for(k=0; k< A->steps; k++){				//compute buffer sizes
       nRtot += A->nR[k];                // to receive
       nStot += A->nS[k];                // to send
     }
    alltoallv_reduce(A->R, A->nR, nRtot, A->S, A->nS, nStot, lvalues, vpixDiag, A->steps, A->comm);
  }
  else{
    printf("\n\n####### WARNING ! : Unvalid communication scheme #######\n\n");
    return 1;
  }
#endif
  free(lvalues);
  return 0;
}







/** @brief Compute Diag(A' diag(Nm1) A).
    @param out_values local output array of doubles*/

int DiagAtA(Mat *A, double *diag, int pflag){
  int i, j, k;
  int nSmax, nRmax;
  double *lvalues;

  lvalues = (double *) malloc(A->lcount *sizeof(double));    /*<allocate and set to 0.0 local va
lues*/
  for(i=0; i < A->lcount; i++)
    lvalues[i]=0.0;

//Naive computation with a full defined diag(Nm1):
  for(i=0; i< A->m; i++)
    for (j=0; j< A->nnz; j++)                                   /*<dot products */
      lvalues[A->indices[i*(A->nnz)+j]]+=(A->values[i*(A->nnz)+j]*A->values[i*(A->nnz)+j]) ;//*vdiagNm1[i];



#if W_MPI
  nRmax=0;  nSmax=0;

  if(A->flag  == BUTTERFLY){                                  /*<branch butterfly*/
    //memcpy(out_values, lvalues, (A->lcount) *sizeof(double)); /*<copy local values into result values*/
    for(k=0; k< A->steps; k++)                                  /*compute max communication buffer size*/
      if(A->nR[k] > nRmax)
        nRmax = A->nR[k];
    for(k=0; k< A->steps; k++)
      if(A->nS[k] > nSmax)
        nSmax = A->nS[k];

    double *com_val;
    com_val=(double *) malloc( A->com_count *sizeof(double));
    for(i=0; i < A->com_count; i++){
      com_val[i]=0.0;
    }
    memcpy(diag, lvalues, (A->lcount) *sizeof(double)); /*<copy local values into result values*/
    m2m(lvalues, A->lindices, A->lcount, com_val, A->com_indices, A->com_count);
    butterfly_reduce(A->R, A->nR, nRmax, A->S, A->nS, nSmax, com_val, A->steps, A->comm);
    m2m(com_val, A->com_indices, A->com_count, diag, A->lindices, A->lcount);
    free(com_val);
  }
  else if(A->flag == RING){
    memcpy(diag, lvalues, (A->lcount) *sizeof(double)); /*<copy local values into result values*/
    for(k=1; k< A->steps; k++)                                  /*compute max communication buffer size*/
      if(A->nR[k] > nRmax)
        nRmax = A->nR[k];

    nSmax = nRmax;
    ring_reduce(A->R, A->nR, nRmax, A->S, A->nS, nSmax, lvalues, diag, A->steps, A->comm);
  }
  else if(A->flag == NONBLOCKING){
    memcpy(diag, lvalues, (A->lcount) *sizeof(double)); /*<copy local values into result values*/
    ring_nonblocking_reduce(A->R, A->nR, A->S, A->nS, lvalues, diag, A->steps, A->comm);
  }
  else if(A->flag == NOEMPTY){
    memcpy(diag, lvalues, (A->lcount) *sizeof(double)); /*<copy local values into result values*/
    int ne=0;
    for(k=1; k< A->steps; k++)
      if(A->nR[k]!=0)
        ne++;

    ring_noempty_reduce(A->R, A->nR, ne, A->S, A->nS, ne, lvalues, diag, A->steps, A->comm);
  }
  else{
    return 1;
  }
#endif
  free(lvalues);
  return 0;
}




int get_pixshare_pond(Mat *A, double *pixpond )
{

  int           i, j, k ;                       // some indexes
  int           m, n, rank, size;
  double        localreduce;                    //reduce buffer
  double        st, t;                          //timers

//  double        *eyesdble;

  MPI_Comm_rank(A->comm, &rank);                 //
  MPI_Comm_size(A->comm, &size);                 //

  m=A->m;                                        //number of local time samples
  n=A->lcount-(A->nnz)*(A->trash_pix);                                   //number of local pixels

//  eyesdble = (double *) malloc(n*sizeof(double));

//create a eyes local vector
  for(i=0; i<n; i++)
    pixpond[i] = 1.;

//communicate with the others processes to have the global reduce
  commScheme(A, pixpond, 2);

// compute the inverse vector
  for(i=0; i<n; i++)
    pixpond[i] = 1./pixpond[i] ;


  return 0;
}
