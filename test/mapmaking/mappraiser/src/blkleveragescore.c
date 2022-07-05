// Midapack library
// mapmaking code example using the Midapack library - release 1.2b, Nov 2012
// Block Leverage score method for Randomized GLS

/** @file   blkleveragescore.c
 @author Niels Guilbert
 @date   February 2022
 @credit  Adapted from work by Frederic Dauvergne
 @Last_update June 2020 by Aygul Jamal */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <mkl.h>
#include <unistd.h>
#include "fitsio.h"
#include "midapack.h"
#include "mappraiser.h"

double genrandn(double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;

  if (call == 1)
  {
    call = !call;
    return (mu + sigma * (double) X2);
  }

  U1 = -1 + ((double)rand()/(double)RAND_MAX) * 2;
  U2 = -1 + ((double)rand()/(double)RAND_MAX) * 2;
  W = pow (U1, 2) + pow (U2, 2);

  while (W >= 1.0 || W == 0.0)
  {
    U1 = -1 + ((double)rand()/(double)RAND_MAX) * 2;
    U2 = -1 + ((double)rand()/(double)RAND_MAX) * 2;
    W = pow (U1, 2) + pow (U2, 2);
  }


  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (double) X1);
}

/** dichotmic search of an integer in a monotony array
    @param number elemnent array of values
    @param monotony array
    @param element to search
    @return index of searched element*/
int dichotomynnz(int nT, int *T, int e, int nnz){
  int min, max, pivot;
  min=0;
  max=nT-1;
  pivot=(max-min)/2;
  if (e == -1){
    pivot = -1;
  }
  else{
    while((e != (T[pivot*nnz]/nnz)) && (max > min) ){
      if((T[pivot*nnz]/nnz)<e){
        min=pivot+1;
      }
      else{
        max=pivot;
      }
      pivot = min + (max-min)/2;
    }
    if (e != (T[pivot*nnz]/nnz)) {
      pivot = -1;
    }
  }
  return pivot;
}

/** Sequential reindexing
    @param T monotony array
    @param nT number of index
    @param A tab to reindex
    @param nA number of element to reindex
    @return array of indices
    @ingroup matmap_group22*/
int newlocindex(int *T, int nT, int *A, int nA, int nnz)
{
  int i, tmp;
  i=0;
  for(i=0; i<nA; i++)
  {
    tmp = A[i];
    A[i] = dichotomynnz(nT, T, tmp, nnz);
  }
  return 0;
}

// Structure for min and max of a array of doubles
struct Pair
{
    double min;
    double max;
};

// Get min and max of an array of doubles
struct Pair getMinMax(double* list, int n)
{
    struct Pair minmax;
    int i;

    // If there is only one element
    // then return it as min and max both
    if (n == 1)
    {
        minmax.max = list[0];
        minmax.min = list[0];
        return minmax;
    }

    // If there are more than one elements,
    // then initialize min and max
    if (list[0] > list[1])
    {
        minmax.max = list[0];
        minmax.min = list[1];
    }
    else
    {
        minmax.max = list[1];
        minmax.min = list[0];
    }

    for(i = 2; i < n; i++)
    {
        if (list[i] > minmax.max)
            minmax.max = list[i];

        else if (list[i] < minmax.min)
            minmax.min = list[i];
    }
    return minmax;
}

// Get closest entries and its index to an input value (target)
int getMinDist(double target, double *list, int sizelist)
{
  int minidx = 0;
  double mindist = abs(target-list[0]);
  for (int i = 0; i < sizelist; i++) {
    double currentdist = abs(target-list[i]);
    if (currentdist < mindist) {
      minidx = i;
      mindist = currentdist;
    }
  }
  return minidx;
}

// Selection of blocks of data to be kept in the GLS problem
int prepare_Rand_GLS(char *outpath, char *ref, Mat *A, Tpltz Nm1, double *b, double *noise, double *cond, int* sampleIdx)
{
  int rank, size;
  MPI_Comm_rank(A->comm, &rank);
  MPI_Comm_size(A->comm, &size);

  if (rank == 0) {
    printf("******* prepare_Rand_GLS *******\n");
    fflush(stdout);
  }

  int nbsamples = 0;
  FILE *fp;
  int GlobsampleIdx[Nm1.nb_blocks_tot];



  srand(time(0));

  // printf("Test %e \n", genrandn(0.0,1.0));
  // printf("Test %e \n", genrandn(0.0,1.0));
  // printf("Test %e \n", genrandn(0.0,1.0));
  // printf("Test %e \n", genrandn(0.0,1.0));
  // printf("Test %e \n", genrandn(0.0,1.0));
  // printf("Test %e \n", genrandn(0.0,1.0));
  // printf("Test %e \n", genrandn(0.0,1.0));

  // Output of Topelitz block Norm
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++AAAAAAAAAAAAAAAAAAAAA
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++AAAAAAAAAAAAAAAAAAAAA
  double TplzBlocknorms   [Nm1.nb_blocks_tot];
  double TplzBlocknormsLoc[Nm1.nb_blocks_loc];
  for (int iblk = 0; iblk < Nm1.nb_blocks_loc; iblk++) {
    TplzBlocknormsLoc[iblk] = 0.;
    for (int i = 0; i < Nm1.tpltzblocks[iblk].lambda; i++) {
      TplzBlocknormsLoc[iblk] += (Nm1.tpltzblocks[iblk].T_block)[i]*(Nm1.tpltzblocks[iblk].T_block)[i];
    }
    TplzBlocknormsLoc[iblk] = sqrt(TplzBlocknormsLoc[iblk]*Nm1.tpltzblocks[iblk].n);
  }

  // if (rank == 0) {
  //   printf("******* CheckPoint 1 *******\n");
  //   fflush(stdout);
  // }
  // if (rank == 0) {
  //   for (int i = 0; i < Nm1.tpltzblocks[2].lambda; i++) {
  //     printf("-- %e", (Nm1.tpltzblocks[2].T_block)[i]);
  //   }
  //
  // }
  int vec_nb_blocks[size];
  int displacements[size+1];
  displacements[0] = 0;
  MPI_Allgather( &Nm1.nb_blocks_loc, 1, MPI_INT, vec_nb_blocks, 1, MPI_INT, MPI_COMM_WORLD);
  // if (rank == 0) {
  //   printf("******* CheckPoint 2 *******\n");
  //   fflush(stdout);
  // }
  // for (int i = 0; i < size; i++) {
  //   displacements[i+1] = displacements[i] + vec_nb_blocks[i];
  //   if (rank == i) {
  //     for (int j = 0; j < Nm1.nb_blocks_loc; j++) {
  //       sampleIdx[j] = displacements[i] + j;
  //     }
  //   }
  //   for (int j = 0; j < Nm1.nb_blocks_loc; j++) {
  //     GlobsampleIdx[displacements[i] + j] = displacements[i] + j;
  //   }
  // }
  // if (rank == 0) {
  //   printf("******* CheckPoint 3 *******\n");
  //   fflush(stdout);
  // }
  // MPI_Allgatherv(TplzBlocknormsLoc, Nm1.nb_blocks_loc, MPI_DOUBLE, TplzBlocknorms, vec_nb_blocks, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
  // double stddev = 0.0, expmean = 0.0;
  // for (int i = 0; i < Nm1.nb_blocks_tot; i++) {
  //   expmean += TplzBlocknormsLoc[i];
  // }
  // if (rank == 0) {
  //   printf("******* CheckPoint 4 *******\n");
  //   fflush(stdout);
  // }
  // expmean = expmean/Nm1.nb_blocks_tot;
  // for (int i = 0; i < Nm1.nb_blocks_tot; i++) {
  //   stddev += (TplzBlocknormsLoc[i]-expmean)*(TplzBlocknormsLoc[i]-expmean);
  // }
  // stddev = sqrt(stddev/Nm1.nb_blocks_tot);
  // if (rank == 0) {
  //   printf("******* CheckPoint 5 *******\n");
  //   fflush(stdout);
  // }
  // if (rank == 0) {
  //   char filename[256];
  //   sprintf(filename,"%s/Tplzblknormfile_%s.dat", outpath, ref);
  //   fp = fopen(filename,"w");
  //   for (int iblk = 0; iblk < Nm1.nb_blocks_tot; iblk++) {
  //     double normw = TplzBlocknorms[iblk];
  //     fwrite(&normw, sizeof(double), 1, fp);
  //   }
  //   fflush(stdout);
  // }
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++AAAAAAAAAAAAAAAAAAAAA
  //          Selection of the blocks according to their norm
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++AAAAAAAAAAAAAAAAAAAAA
  // int Globnbsamples = (Nm1.nb_blocks_tot*80)/100; // 80% of blocks kept in memory
  // int randidx = 0;
  // struct Pair minmax = getMinMax(&TplzBlocknorms[0], Nm1.nb_blocks_tot); //get min and max for uniform distrib
  // for (int iblk = 0; iblk < Globnbsamples; iblk++) {
  //   int    idxsave   = GlobsampleIdx[iblk];
  //   double normsave  = TplzBlocknorms[iblk];
  //   double unirand   =((double)rand()/(double)RAND_MAX); //random value between 0 and 1
  //   // double randval   = minmax.min+unirand*(minmax.max-minmax.min); //unfirm subset of the data
  //   double randval   = genrandn(expmean,stddev); //unfirm subset of the data
  //   //Look for the closest value in TplzBlocknorms
  //   randidx = getMinDist(randval, &TplzBlocknorms[iblk], Nm1.nb_blocks_tot-iblk); //get closest value from the random one
  //   // exchange values and indices
  //   GlobsampleIdx [iblk        ] = iblk+randidx;
  //   TplzBlocknorms[iblk        ] = TplzBlocknorms[iblk+randidx];
  //   GlobsampleIdx [iblk+randidx] = idxsave;
  //   TplzBlocknorms[iblk+randidx] = normsave;
  //   int isave = -1;
  //   for (int i = nbsamples; i < Nm1.nb_blocks_loc; i++) {
  //     if (GlobsampleIdx[iblk] == sampleIdx[i]) {
  //       // idxsave = sampleIdx[nbsamples];
  //       // sampleIdx[nbsamples] = sampleIdx[i]-displacements[rank];
  //       // sampleIdx[i] = idxsave;
  //       isave = i;
  //     }
  //   }
  //   if (isave > nbsamples) {
  //     idxsave = sampleIdx[nbsamples];
  //     sampleIdx[nbsamples] = sampleIdx[isave]-displacements[rank];
  //     // printf("%i %i %i %i\n", rank, countsp, sampleIdx[countsp], displacements[rank]);
  //     sampleIdx[isave] = idxsave;
  //     nbsamples += 1;
  //   }
  //   else if(isave == nbsamples) {
  //     sampleIdx[isave] = sampleIdx[isave]-displacements[rank];
  //     nbsamples += 1;
  //   }
  // }
  //
  // // for (int iblk = 0; iblk < nbsamples; iblk++) {
  // //   printf("On rank %i sample block : %i\n", rank, sampleIdx[iblk]);
  // // }
  // if (nbsamples == 0) {
  //   printf("On rank %i number of samples = 0\n", rank);
  // }


  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++AAAAAAAAAAAAAAAAAAAAA
  //          Random selection of the blocks
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++AAAAAAAAAAAAAAAAAAAAA
  nbsamples = (Nm1.nb_blocks_loc)*100/100;
  for (int iblk = 0; iblk < Nm1.nb_blocks_loc; iblk++) {
    sampleIdx[iblk] = iblk;
  }
  for (int iblk = 0; iblk < nbsamples; iblk++) {
    int idxsave = sampleIdx[iblk];
    int randidx = iblk+rand()%(Nm1.nb_blocks_loc-iblk);
    sampleIdx[iblk] = sampleIdx[randidx];
    sampleIdx[randidx] = idxsave;
  }
  // if (rank == 0) {
  //   printf("******* CheckPoint 6 *******\n");
  //   fflush(stdout);
  // }

  return nbsamples;
}

// Interpolation for Initializing
void Update_Initial_Guess(double *x_init, int n_init, double *x, int* Neighbours, int* old_lindices, int old_npix, int old_trashpix, Mat *A, int *old2new, double* InterpWeights)
{
  int rank, size;
  FILE *fp;

  MPI_Comm_rank(A->comm, &rank);
  MPI_Comm_size(A->comm, &size);

  if (rank == 0) {
    printf("******* Update_Initial_Guess *******\n");
  }

  // if (rank == 0) {
  //   printf("******* newlocindex Function call *******\n");
  // }
  // fflush(stdout);
  // Update Interpolation Info with new numbering
  newlocindex(A->lindices, A->lcount/A->nnz, Neighbours, old_npix*4, A->nnz);
  // if (rank == 0) {
  //   printf("******* Update intial guess with sol of Randomized GLS problem *******\n");
  // }
  // fflush(stdout);
  // Update intial guess with sol of Randomized GLS problem
  int mapsizeA = A->lcount-(A->nnz)*(A->trash_pix);
  for (int j = 0; j < mapsizeA; j++)
  {
    int globidx2 = A->lindices[j+(A->nnz)*(A->trash_pix)];
    int lcount = 0;
    for(int i=0; i< n_init; i++){
      int globidx1 = old_lindices[i+(A->nnz)*old_trashpix];
      if (globidx1 == globidx2) {
        x[j] = x_init[i];
        lcount += 1;
      }
    }
    // if(lcount > 0)
    // {
    //
    // }
  }
  // if (rank == 0) {
  //   printf("******* Interpolation To complete Initial Guess *******\n");
  // }
  // fflush(stdout);
  // Interpolation To complete Initial Guess
  // for (int j = 0; j < mapsizeA/(A->nnz); j++)
  // {
  //   if (x[j*(A->nnz)] == 0)
  //   {
  //     double preval = x[j*(A->nnz)];
  //     for(int iblk = 0; iblk < (A->nnz); iblk++)
  //     {
  //       int countneighb = 0;
  //       for (int i = 0; i < 4; i++)
  //       {
  //         int idxneighbpix = Neighbours[old2new[j]*4+i]-(A->trash_pix);
  //         if (idxneighbpix >= 0)
  //         {
  //           countneighb += 1;
  //           x[j*(A->nnz)+iblk] += x[idxneighbpix*(A->nnz)+iblk]*InterpWeights[old2new[j]*4+i];
  //           // x[j*(A->nnz)+iblk] += x[idxneighbpix*(A->nnz)+iblk];
  //         }
  //         // if (countneighb>0) {
  //         //   x[j*(A->nnz)+iblk] = x[j*(A->nnz)+iblk]/(1.0*countneighb);
  //         // }
  //       }
  //     }
  //     double postval = x[j*(A->nnz)];
  //     printf("preval = %e *** postval = %e \n", preval, postval);
  //   }
  // }
  // if (rank == 0) {
  //   printf("******* Update_Initial_Guess END *******\n");
  // }
  // fflush(stdout);
  // if (rank == 0) {
  //   for (int i = 1000; i < 1009; i++) {
  //     printf("******* OOOOOO %i\n", Neighbours[i]);
  //   }
  // }

}
