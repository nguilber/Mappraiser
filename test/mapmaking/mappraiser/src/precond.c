
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

extern int dgecon_(const char *norm, const int *n, double *a, const int *lda, const double *anorm, double *rcond, double *work, int *iwork, int *info, int len);
extern int dgetrf_(const int *m, const int *n, double *a, const int *lda, int *lpiv, int *info);
extern double dlange_(const char *norm, const int *m, const int *n, const double *a, const int *lda, double *work, const int norm_len);

extern int dgeqp3(const int *m, const int *n, double *a, const int *lda, const double *tau, const double *work, const int *lwork, const int *info); //M, N, A, LDA, JPVT, TAU, WORK, LWORK, INFO

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
int Build_ATA_bloc(Mat *A, Tpltz Nm1, double *ATA, int row_indice, int nb_rows, int np)
{
  int nnz;                              // Number of coef per line of A
  int i0,i1,i2,i3,i4,i5,i6,i7,i8;       // some loop index
  int locpix;                           // index of the pixel watch at time i0 in the row_indice block on the proc
  int info;                             // LAPACK return variable
  
  nnz = A->nnz;
  
  for (i0 = 0; i0 < nb_rows; ++i0) {
    locpix = A->indices[row_indice*nnz+i0*nnz];
        
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
  
  double *tmp_blck;
  tmp_blck = (double *) malloc(nnz*nnz*sizeof(double));

  for (i3 = 0; i3 < np; ++i3) {

    if (ATA[i3*nnz*nnz] > 3) {
      
      for (i4=0; i4 < nnz*nnz; ++i4) {
	tmp_blck[i4] = ATA[i3*nnz*nnz+i4];	
      }
      
      // ###### Compute the norm 1 of tmp_blk
      double A_norm1;
      A_norm1 = LAPACKE_dlange(LAPACK_ROW_MAJOR,'1',nnz,nnz,tmp_blck,nnz);
      
      // ###### Compute the condition number of the block to know if we'll have to apply it in Apply_ATA_bloc : need to have the cholesky factorization done first, that's why it's not before LAPACKE_dpotrf
      double rcond;
      // info = LAPACKE_dpocon(LAPACK_ROW_MAJOR,'L',nnz,tmp_blck,nnz,A_norm1,&rcond);
      info = LAPACKE_dgecon(LAPACK_ROW_MAJOR,'1',nnz,tmp_blck,nnz,A_norm1,&rcond);
      
      // ###### Copy the Cholesky factor in ATA if condition number not bad : rcond = 1/cond > 1/1000 <=> cond < 1000
      if (rcond > 1e-3) { // <=> cond < 1000
	
	// ###### Call LAPACK routines to compute Cholesky factorisation of tmp_blck
	info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',nnz,tmp_blck,nnz);
	
	if (info > 0) {
	  // printf("The leading minor of order %i is not positive-definite, and the factorization could not be completed.",info);
	  ATA[i3*nnz*nnz] = -1;
	}
	if (info < 0) {
	  printf("The parameter %i had an illegal value.",-info);
	}
	if (info == 0) {
	  // printf("The factorization went good.");
	  for (i7 = 0; i7 < nnz; ++i7) {
	    for (i8 = 0; i8 < i7+1; ++i8) {
	      ATA[i3*nnz*nnz+i7*nnz+i8] = tmp_blck[i7*nnz+i8];
	    }
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

  return 0;
}

// Apply the product A_i.T * v for a timestream of one block
int Apply_ATr_bloc(Mat *A, double *x, double *y, int row_indice, int nb_rows)
{
  int i0,i1;           // Some loop index
  
  for (i0 = row_indice; i0 < row_indice+nb_rows; ++i0) {
    for(i1 = 0; i1 < A->nnz; ++i1){				//
      y[A->indices[i0*A->nnz+i1]] += (A->values[i0*A->nnz+i1]) * x[i0];
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
  tmp_blck = (double *) malloc(sizeof(double)*nnz*nnz);
  tmp_vec = (double *) malloc(sizeof(double)*nnz);

  for (i0 = 0; i0 < np; ++i0) {

    for (i1 = 0; i1 < nnz; ++i1) {
      tmp_vec[i1] = y[i0*nnz+i1];
      for (i2 = 0; i2 < i1+1; ++i2) {
	tmp_blck[i1*nnz+i2] = ATA[i0+i1*nnz+i2];
      }
    }

    if (tmp_blck[0] =! -1) { 
      // ###### Apply (P_i.T * P_i)^{-1} * v using the cholesky factor
      info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR,'L',nnz,1,tmp_blck,nnz,tmp_vec,nnz);
      
      if (info =! 0) {
	printf("The parameter %i had an illegal value",info);
      }

      // ###### Copy the results in output of the function z
      for (i3 = 0; i3 < nnz; ++i3) {
	z[i0*nnz+i3] = tmp_vec[i3];
      }
    }
    else {
      for (i4 = 0; i4 < nnz; ++i4) {
	z[i0*nnz+i4] = 0.0;
      }
    }
  }
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
int Build_ALS(Mat *A, Tpltz Nm1, double *Z, int nb_defl, int np)
{
  int nb_blocks_loc;                  // Number of local bloc, ie on a the proc
  Block *tpltzblocks;                 // Pointer to the tpltzblocks struct
  int nti;                            // size of the time stream
  int row_indice;                     // Index of the first line in depointing matrix A for the bloc we look at
  int nnz;                            // Number of non-zeros per line in A
  double *CS;                         // Array for the coarse space before reduction of the number of vector through QR     
  double *ATA;                        // Array to store the (A_i.T * A_i) block operator
  fftw_complex *in;                   // Vector to store entry of the FFT                                                   
  fftw_complex *out;                  // Vector to store output of the FFT                                                  
  double *x;                          // Vector to store eigen vector w.r.t to one bloc                                     
  double *y;                          // Vector to store A_i.T * x                                                          
  double *z;                          // Vector to store (A_i.T * A_i)^{-1} * y                                             
  int i0,i1,i2,i3,i4;                 // loop indices
  
  nb_blocks_loc = Nm1.nb_blocks_loc;
  tpltzblocks = Nm1.tpltzblocks;
  row_indice = 0;
  nnz = A->nnz;
  CS = (double *) malloc(sizeof(double)*np*nb_blocks_loc*nb_defl);
  in = (fftw_complex *) malloc(nti*sizeof(fftw_complex));
  out = (fftw_complex *) malloc(nti*sizeof(fftw_complex));  
  x = (double *) malloc(nti*sizeof(double));
  y = (double *) calloc(np,sizeof(double));
  z = (double *) malloc(np*sizeof(double));
  ATA = (double *)  malloc(np*nnz*nnz*sizeof(double));
    
  for (i0 = 0; i0 < nb_blocks_loc; ++i0) {
    nti = tpltzblocks[i0].n; // Size of the block
    
    ATA = (double *)  calloc(np*nnz*nnz,sizeof(double));
    
    // ###### Build ATA w.r.t. the local block i0
    Build_ATA_bloc(A,Nm1,ATA,row_indice,nti,np);
    
    for (i1=0; i1<nb_defl; ++i1) {
      // ###### Get the fourier mode of order i0 and take it as an eigenvector of the block
      get_Fourier_mode(in,out,nti,i0);
      
      // ###### Get the eigenvector through ARPACK
      // get_ARPACK(...);
      
      for (i3 = 0; i3 < nti; ++i3) {
    	x[i3] = out[i3][0]; // Store the real part of the results of FFT in an array.
      }
      
      // ###### Call of the function to do the local pointing : pointing of 1 block
      Apply_ATr_bloc(A,x,y,row_indice,nti); // Compute P_i^\top * Fourier mode n°i1

      
      // ###### Do the (A_i^T*A_i)^\dagger local product
      Apply_ATA_bloc(A,ATA,y,z,np);
      
      // ###### Store the resulting vector in coarse space array CS
      for (i4 = 0; i4 < np; ++i4) {
    	CS[i0*nb_defl*np+i1*np+i4] = z[i4];
      }

    }
    row_indice += nti;
  }
  return 0;
}

// Build a orthonormal basis of a coarse space Z
int Orthogonalize_Space_loc(double *Z, int nb_rows, int nb_cols, double tol_svd, int *size_CS)
{
  int i0,i1,i2,i3,i4;                              // Loop index
  double *tau;                                     // Coef. of the Householder reflection in the Q-factor
  int info;                                        // LAPACK return variable
    
  tau = (double *) malloc(sizeof(double)*nb_cols);

  // Compute non-pivoting QR factorization of coarse space Z
  info = LAPACKE_dgeqrfp(LAPACK_COL_MAJOR,nb_rows,nb_cols,Z,nb_rows,tau);

  if (info =! 0) {
    printf("The parameter %i had an illegal value",info);
  }
  
  // ###### Extract R from data in COL_MAJOR
  double *R;
  R = (double *) calloc(nb_cols*nb_cols,sizeof(double));
  
  for (i0 = 0; i0 < nb_cols; ++i0) {
    for (i1 = 0; i1 < i0+1; ++i1) {
      R[i0+i1*nb_cols] = Z[i0+i1*nb_cols];
    }
  }
  
  // ###### Compute a RR-QR on the R-factor
  int *jpvt;
  jpvt = (int *) malloc(sizeof(int)*nb_cols);
  double *tau_tmp;
  tau_tmp = (double *) malloc(sizeof(double)*nb_cols);
  
  info = LAPACKE_dgeqp3(LAPACK_COL_MAJOR,nb_cols,nb_cols,R,nb_cols,jpvt,tau_tmp);

  if (info =! 0) {
    printf("The parameter %i had an illegal value",info);
  }
  
  /* free(tau_tmp); */
  
  // ###### Get the Q-factor from Z
  info = LAPACKE_dorgqr(LAPACK_COL_MAJOR,nb_rows,nb_cols,nb_cols,Z,nb_rows,tau);

  if (info =! 0) {
    printf("The parameter %i had an illegal value",info);
  }
  
  // ###### Select the colums of the Q-factor related to high enough singular values
  size_CS[0] = 0;
  
  while (R[size_CS[0]*nb_cols+size_CS[0]] > tol_svd) {    
    size_CS[0] += 1;
  }
  
  /* free(R); */

  double *CS;
  
  CS = (double *) malloc(sizeof(double)*nb_rows*size_CS[0]);
  
  for (i2 = 0; i2 < size_CS[0]; ++i2) {
    for (i3 = 0; i3 < nb_rows; ++i3) {
      CS[i2*nb_rows+i3] = Z[jpvt[i2]*nb_rows+i3];
    }
  }

  for (i4 = 0; i4 < nb_rows*size_CS[0]; ++i4) {
    Z[i4] = CS[i4];
  }

  Z = (double *) realloc(Z,nb_rows*size_CS[0]);

  return 0;
}

















































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
