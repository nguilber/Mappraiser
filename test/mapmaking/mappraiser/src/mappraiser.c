// MAPPRAISER vdev
// Massively parallel iterative map-making code using the Midapack library v1.2b, Nov 2012
// The routine processes pointing, signal and noise data arrays and produces maps in fits files

/** @file   mappraiser.c
    @author Hamza El Bouhargani
    @date   May 2019
    @Last_update June 2020 by Aygul Jamal  */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <mkl.h>
#include "fitsio.h"
#include "midapack.h"
#include "mappraiser.h"


int x2map_pol( double *mapI, double *mapQ, double *mapU, double *Cond, int * hits, int npix, double *x, int *lstid, double *cond, int *lhits, int xsize);

void MLmap(MPI_Comm comm, char *outpath, char *ref, int solver, int precond, int Z_2lvl, int pointing_commflag, double tol, int maxiter, int enlFac, int ortho_alg, int bs_red, int nside, void *data_size_proc, int nb_blocks_loc, void *local_blocks_sizes, int Nnz, void *pix, void *pixweights, void *signal, double *noise, int lambda, double *invtt, void *Neighbours, void *InterpWeights)
{
  int64_t	M;       //Global number of rows
  int		m, Nb_t_Intervals;  //local number of rows of the pointing matrix A, nbr of stationary intervals
  int64_t	gif;			//global indice for the first local line
  int		i, j, k;
  Mat	A, B;			        //pointing matrix structure
  int 		*id0pix, *ll, *id0pix_B, *ll_B;
  double	*x;	//pixel domain vectors
  void	*copy_signal, *pix_copy, *pixweights_copy;	//pixel domain vectors
  double	*x_B;	//pixel domain vectors
  double	st, t;		 	//timer, start time
  int 		rank, size;
  MPI_Status status;
  FILE *fp;
  int npix = 12*pow(nside,2);
  double normb = 0;
  //mkl_set_num_threads(1); // Circumvent an MKL bug

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  if(rank==0){
    printf("\n############# MAPPRAISER : MidAPack PaRAllel Iterative Sky EstimatoR vDev, May 2019 ################\n");
    printf("rank = %d, size = %d\n",rank,size);
    fflush(stdout);
  }

  //total length of the time domaine signal
  M = 0;
  for(i=0;i<size;i++){
    M += ((int *)data_size_proc)[i];
    fflush(stdout);
  }
  if(rank==0){
    printf("[rank %d] M=%ld\n", rank, M);
    fflush(stdout);
  }

  //compute distribution indexes over the processes
  m = ((int *)data_size_proc)[rank];
  gif = 0;
  for(i=0;i<rank;i++)
    gif += ((int *)data_size_proc)[i];

  //Print information on data distribution
  int Nb_t_Intervals_loc = nb_blocks_loc;
  MPI_Allreduce(&nb_blocks_loc, &Nb_t_Intervals, 1, MPI_INT, MPI_SUM, comm);
  int nb_proc_shared_one_interval = 1; //max(1, size/Nb_t_Intervals );
  if(rank==0){
    printf("[rank %d] size=%d \t m=%d \t Nb_t_Intervals=%d \n", rank, size, m, Nb_t_Intervals);
    printf("[rank %d] Nb_t_Intervals_loc=%d \n", rank, Nb_t_Intervals_loc );
    fflush(stdout);
  }

  pix_copy = calloc(m*Nnz,sizeof(int));
  memcpy(pix_copy, pix, m*Nnz*sizeof(int));
  pixweights_copy = calloc(m*Nnz,sizeof(double));
  memcpy(pixweights_copy, pixweights, m*Nnz*sizeof(double));

  //Pointing matrix init
  st=MPI_Wtime();
  A.trash_pix =0;
  B.trash_pix =0;
  int *Locshift = (int *) malloc((nb_blocks_loc+1) * sizeof(int));
  Locshift[0] = 0;
  for (int iblk = 0; iblk < nb_blocks_loc; iblk++) {
    Locshift[iblk+1] = Locshift[iblk] + ((int *)local_blocks_sizes)[iblk];
  }
  A.shift = Locshift;
  B.shift = Locshift;
  MatInit( &A, m, Nnz, pix, pixweights, pointing_commflag, comm);
  MatInit( &B, m, Nnz, pix_copy, pixweights_copy, pointing_commflag, comm);
  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Initializing pointing matrix time=%lf \n", rank, t-st);
    fflush(stdout);
  }

  //Build pixel-to-time domain mapping
  st=MPI_Wtime();
  id0pix = (int *) malloc(A.lcount/(A.nnz) * sizeof(int)); //index of the last time sample pointing to each pixel
  id0pix_B = (int *) malloc(A.lcount/(A.nnz) * sizeof(int)); //index of the last time sample pointing to each pixel
  ll = (int *) malloc(m * sizeof(int)); //linked list of time samples indexes
  ll_B = (int *) malloc(m * sizeof(int)); //linked list of time samples indexes
  copy_signal = calloc(A.m,sizeof (double));
  memcpy(copy_signal, signal, m * sizeof(double));

  //initialize the mapping arrays to -1
  for(i=0; i<m; i++){
    ll[i] = -1;
    ll_B[i] = -1;

  }
  for(j=0; j<A.lcount/(A.nnz); j++){
    id0pix[j] = -1;
    id0pix_B[j] = -1;
  }
  //build the linked list chain of time samples corresponding to each pixel
  for(i=0;i<m;i++){
    if(id0pix[A.indices[i*A.nnz]/(A.nnz)] == -1){
      id0pix[A.indices[i*A.nnz]/(A.nnz)] = i;
      id0pix_B[A.indices[i*A.nnz]/(A.nnz)] = i;
      }
    else{
      ll_B[i] = id0pix[A.indices[i*A.nnz]/(A.nnz)];
      ll[i] = id0pix[A.indices[i*A.nnz]/(A.nnz)];
      id0pix_B[A.indices[i*A.nnz]/(A.nnz)] = i;
      id0pix[A.indices[i*A.nnz]/(A.nnz)] = i;
    }
  }
  A.id0pix = id0pix;
  A.ll = ll;
  B.id0pix = id0pix_B;
  B.ll = ll_B;
  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Total pixel-to-time domain mapping time=%lf \n", rank, t-st);
    fflush(stdout);
  }

  //Map objects memory allocation (MatInit gives A.lcount)
  int *lhits;
  double *cond;
  x   = (double *) malloc(A.lcount*sizeof(double));
  x_B   = (double *) malloc(A.lcount*sizeof(double));
  cond = (double *) malloc((int)(A.lcount/3)*sizeof(double));
  lhits = (int *) malloc((int)(A.lcount/3) * sizeof(int));
  int *lhits_B;
  double *cond_B;
  cond_B = (double *) malloc((int)(A.lcount/3)*sizeof(double));
  lhits_B = (int *) malloc((int)(A.lcount/3) * sizeof(int));

  for(j=0; j<A.lcount; j++){
    x[j] = 0.;
    x_B[j] = 0.;
    if(j%3 == 0){
      lhits_B[(int)(j/3)] = 0;
      cond_B [(int)(j/3)] = 0.;
      lhits  [(int)(j/3)] = 0;
      cond   [(int)(j/3)] = 0.;
    }
  }

  //Create piecewise Toeplitz matrix
  //specifics parameters:
  int nb_blocks_tot = Nb_t_Intervals;
  int lambda_block_avg = lambda;

  //flags for Toeplitz product strategy
  Flag flag_stgy;
  flag_stgy_init_auto(&flag_stgy);

  //to print something on screen
  flag_stgy.flag_verbose=1;

  //define Toeplitz blocks list and structure for Nm1
  Block *tpltzblocks;
  Tpltz Nm1;
  Block *tpltzblocks2;
  Tpltz Nm2;

  //dependants parameters:
  int64_t nrow = M;
  int mcol = 1;

  int64_t id0 = gif;
  int local_V_size = m;

  //Block definition
  tpltzblocks  = (Block *) malloc(nb_blocks_loc * sizeof(Block));
  tpltzblocks2 = (Block *) malloc(nb_blocks_loc * sizeof(Block));
  int prct_z = 30;
  defineBlocks_avg(tpltzblocks, tpltzblocks2, invtt, nb_blocks_loc, local_blocks_sizes, lambda_block_avg, id0, prct_z);
  defineTpltz_avg( &Nm1, nrow, 1, mcol, tpltzblocks2, nb_blocks_loc, nb_blocks_tot, id0, local_V_size, flag_stgy, comm);
  if (rank==0) {
    printf("[rank %d] Noise model: Banded block Toeplitz, half bandwidth = %d \n", rank, lambda_block_avg);
    printf("[rank %d] %s  %s \n", rank, outpath, ref);
    fflush(stdout);
  }

  int nbsamples;
  int* sampleIdx = (int*) malloc( sizeof(int)* nb_blocks_loc);
  //print Toeplitz parameters for information
  nbsamples = prepare_Rand_GLS(outpath, ref, &A, Nm1, signal, noise, cond, sampleIdx);

  MPI_Barrier(comm);
  if(rank==0){
    // for (int i = 0; i < 8; i++) {
    //   printf(" %d,", ((int*)Neighbours)[i]);
    // }
    printf(" \n");
    printf("##### Start PCG ####################\n");
  }
  fflush(stdout);

  st=MPI_Wtime();
  // Conjugate Gradient
  //   for (int i = 0; i < nb_blocks_loc; i++) {
  //     printf("Block samples shift %i\n", A.shift[i+1]);
  //   }
  //   printf("Block shift its ok !!!!!!\n");
  // fflush(stdout);

  if(solver == 0){

    normb = PCG_GLS_rand(outpath, ref, &B, Nm1, x_B, copy_signal, noise, cond_B, lhits_B, tol, maxiter, precond, Z_2lvl, nbsamples, sampleIdx);
    int mapsizeB = B.lcount-(B.nnz)*(B.trash_pix);

    // char filename[256];
    // double *mapI_true;
    // mapI_true = (double *) calloc(npix, sizeof(double));
    // sprintf(filename,"%s/Input_U.dat", outpath);
    // fp = fopen(filename,"r");
    // for (i = 0; i < npix; i++) {
    //   fscanf(fp,"%lf", &mapI_true[i]);
    //   // printf("%lf\n", mapI_true[i]);
    // }
    // double nerror = 0.;
    // double truenorm = 0.;
    // double xbnorm = 0.;
    // for(i=0; i< mapsizeB/(B.nnz); i++){
    //   int globidx1 = B.lindices[B.nnz*(i+(B.trash_pix))]/(B.nnz);
    //   nerror += (mapI_true[globidx1]-x_B[i*(B.nnz)+2])*(mapI_true[globidx1]-x_B[i*(B.nnz)+2]);
    //   truenorm += mapI_true[globidx1]*mapI_true[globidx1];
    //   xbnorm += x_B[i*(B.nnz)+2]*x_B[i*(B.nnz)+2];
    //   // printf("%e\n", mapI_true[globidx1]);
    //   }
    //   if (truenorm != 0.) {
    //     nerror = sqrt(nerror/truenorm);
    //     printf("Rank %d, xb norm : %e\n",rank, sqrt(xbnorm));
    //     printf("Rank %d, mapI_norm : %e\n",rank, sqrt(truenorm));
    //     printf("Rank %d, Error Norm on non degenerate Pixels : %e\n",rank, nerror);
    //   }
    // fclose(fp);
    // free(mapI_true);
    defineTpltz_avg( &Nm1, nrow, 1, mcol, tpltzblocks, nb_blocks_loc, nb_blocks_tot, id0, local_V_size, flag_stgy, comm);
    int precond2 = 0;
    PCG_GLS_true(outpath, ref, &A, Nm1, x, signal, noise, cond, lhits, tol, maxiter, precond2, Z_2lvl, x_B, mapsizeB, B.lindices, B.trash_pix, nbsamples, sampleIdx, Neighbours, InterpWeights, normb);

    // if (rank == 0) {
    //   for(int i=A.nnz; i< (A.m-1)*A.nnz; i+=A.nnz){
    //     printf("Scan nb %i, Neighbour 1 : %i, Neighbour 2 : %i\n", A.indices[i]/A.nnz, A.indices[i-A.nnz]/A.nnz, A.indices[i+A.nnz]/A.nnz);
    //   }
    // }
    // if (A.lcount != B.lcount) {
    //   printf("************ %i Trash pixels different for A and B on rank %i\n",(A.lcount - B.lcount)/3, rank);
    // }
    // printf("************ %i Trash pixels on rank %i\n", A.lcount/3, rank);

  }
  else if (solver == 1)
    ECG_GLS(outpath, ref, &A, Nm1, x, signal, noise, cond, lhits, tol, maxiter, enlFac, ortho_alg, bs_red);
  else {
    printf("Whoops! Incorrect solver parameter, please check documentation and try again\n");
    printf("Quick reminder: solver = 0 -> PCG, solver = 1 -> ECG\n");
    exit(1);
  }

  MPI_Barrier(comm);
  t=MPI_Wtime();
  if(rank==0)
    printf("##### End PCG ####################\n");
  if (rank==0) {
    printf("[rank %d] Total PCG time=%lf \n", rank, t-st);
  }
  fflush(stdout);

  //write output to fits files:
  st=MPI_Wtime();
  int mapsize = A.lcount-(A.nnz)*(A.trash_pix);
  int map_id = rank;

  int *lstid;
  lstid = (int *) calloc(mapsize, sizeof(int));
  for(i=0; i< mapsize; i++){
    lstid[i] = A.lindices[i+(A.nnz)*(A.trash_pix)];
  }

  if (rank!=0){
    MPI_Send(&mapsize, 1, MPI_INT, 0, 0, comm);
    MPI_Send(lstid, mapsize, MPI_INT, 0, 1, comm);
    MPI_Send(x, mapsize, MPI_DOUBLE, 0, 2, comm);
    MPI_Send(cond, mapsize/Nnz, MPI_DOUBLE, 0, 3, comm);
    MPI_Send(lhits, mapsize/Nnz, MPI_INT, 0, 4, comm);
  }

  if (rank==0){
    npix = 12*pow(nside,2);
    int oldsize;

    double *mapI;
    mapI    = (double *) calloc(npix, sizeof(double));
    double *mapQ;
    mapQ    = (double *) calloc(npix, sizeof(double));
    double *mapU;
    mapU    = (double *) calloc(npix, sizeof(double));
    int *hits;
    hits = (int *) calloc(npix, sizeof(int));
    double *Cond;
    Cond = (double *) calloc(npix, sizeof(double));

    for (i=0;i<size;i++){
      if (i!=0){
        oldsize = mapsize;
        MPI_Recv(&mapsize, 1, MPI_INT, i, 0, comm, &status);
        if (oldsize!=mapsize){
          lstid = (int *) realloc(lstid, mapsize*sizeof(int));
          x = (double *) realloc(x, mapsize*sizeof(double));
          cond = (double *) realloc(cond, mapsize*sizeof(double));
          lhits = (int *) realloc(lhits, mapsize*sizeof(int));
        }
        MPI_Recv(lstid, mapsize, MPI_INT, i, 1, comm, &status);
        MPI_Recv(x, mapsize, MPI_DOUBLE, i, 2, comm, &status);
        MPI_Recv(cond, mapsize/Nnz, MPI_DOUBLE, i, 3, comm, &status);
        MPI_Recv(lhits, mapsize/Nnz, MPI_INT, i, 4, comm, &status);
      }
      x2map_pol(mapI, mapQ, mapU, Cond, hits, npix, x, lstid, cond, lhits, mapsize);
    }
    printf("Checking output directory ... old files will be overwritten\n");
    char Imap_name[256];
    char Qmap_name[256];
    char Umap_name[256];
    char Condmap_name[256];
    char Hitsmap_name[256];
    char nest = 1;
    char *cordsys = "C";
    int ret,w=1;

    sprintf(Imap_name,"%s/mapI_%s.fits", outpath, ref);
    sprintf(Qmap_name,"%s/mapQ_%s.fits", outpath, ref);
    sprintf(Umap_name,"%s/mapU_%s.fits", outpath, ref);
    sprintf(Condmap_name,"%s/Cond_%s.fits", outpath, ref);
    sprintf(Hitsmap_name,"%s/Hits_%s.fits", outpath, ref);
  //   sprintf(Imap_name,"%s/mapI_test_save.fits", outpath
  // );
  //   sprintf(Qmap_name,"%s/mapQ_test_save.fits", outpath
  // );
  //   sprintf(Umap_name,"%s/mapU_test_save.fits", outpath
  // );
  //   sprintf(Condmap_name,"%s/Cond_test_save.fits", outpath
  // );
  //   sprintf(Hitsmap_name,"%s/Hits_test_save.fits", outpath
  // );

    if( access( Imap_name, F_OK ) != -1 ) {
      ret = remove(Imap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Imap_name);
        w = 0;
      }
    }

    if( access( Qmap_name, F_OK ) != -1 ) {
      ret = remove(Qmap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Qmap_name);
        w = 0;
      }
    }

    if( access( Umap_name, F_OK ) != -1 ) {
      ret = remove(Umap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Umap_name);
        w = 0;
      }
    }

    if( access( Condmap_name, F_OK ) != -1 ) {
      ret = remove(Condmap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Condmap_name);
        w = 0;
      }
    }

    if( access( Hitsmap_name, F_OK ) != -1 ) {
      ret = remove(Hitsmap_name);
      if(ret != 0){
        printf("Error: unable to delete the file %s\n",Hitsmap_name);
        w = 0;
      }
    }

    if(w==1){
      printf("Writing HEALPix maps FITS files ...\n");
      write_map(mapI, TDOUBLE, nside, Imap_name, nest, cordsys);
      write_map(mapQ, TDOUBLE, nside, Qmap_name, nest, cordsys);
      write_map(mapU, TDOUBLE, nside, Umap_name, nest, cordsys);
      write_map(Cond, TDOUBLE, nside, Condmap_name, nest, cordsys);
      write_map(hits, TINT, nside, Hitsmap_name, nest, cordsys);
    }
    else{
      printf("IO Error: Could not overwrite old files, map results will not be stored ;(\n");
    }

  }

  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Write output files time=%lf \n", rank, t-st);
    fflush(stdout);
  }
  st=MPI_Wtime();

  MatFree(&A);
  A.indices = NULL;
  A.values = NULL;                                                //free memory
  free(x);
  free(cond);
  free(lhits);
  free(cond_B);
  free(lhits_B);
  free(copy_signal);
  free(tpltzblocks);
  free(sampleIdx);
  free(Locshift);
  MPI_Barrier(comm);
  t=MPI_Wtime();
  if (rank==0) {
    printf("[rank %d] Free memory time=%lf \n", rank, t-st);
    fflush(stdout);
  }
  // MPI_Finalize();
}

int x2map_pol( double *mapI, double *mapQ, double *mapU, double *Cond, int * hits, int npix, double *x, int *lstid, double *cond, int *lhits, int xsize)
{

  int i;

  for(i=0; i<xsize; i++){
    if(i%3 == 0){
      mapI[(int)(lstid[i]/3)]= x[i];
      hits[(int)(lstid[i]/3)]= lhits[(int)(i/3)];
      Cond[(int)(lstid[i]/3)]= cond[(int)(i/3)];
    }
    else if (i%3 == 1)
      mapQ[(int)(lstid[i]/3)]= x[i];
    else
      mapU[(int)(lstid[i]/3)]= x[i];
  }

  return 0;
}
