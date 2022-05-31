// Midapack library
// mapmaking code example using the Midapack library - release 1.2b, Nov 2012
// PCG routine applied to the map-making equation
// This can use the block-diagonal jacobi or Two-level preconditionners

/** @file   pcg_true.c
 @author Hamza El Bouhargani
 @date   May 2019
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

int PCG_GLS_true(char *outpath, char *ref, Mat *A, Tpltz Nm1, double *x, double *b, double *noise, double *cond, int *lhits, double tol, int K, int precond, int Z_2lvl, double *x_init, int n_init, int* old_lindices, int old_trashpix, int nbsamples, int *sampleIdx, int *Neighbours, double *InterpWeights, double normb)
{
    int rank, size;
    MPI_Comm_rank(A->comm, &rank);
    MPI_Comm_size(A->comm, &size);
    if (rank == 0)
      printf("##### Start PCG on full GLS Pb ################### \n");
    fflush(stdout);
    int    i, j, k;     // some indexes
    int    m, n;        // number of local time samples, number of local pixels
    double localreduce; // reduce buffer
    double st, t;       // timers
    double solve_time = 0.0;
    double res, res0, res_rel;

    double *_g, *ACg, *Ah, *Nm1Ah; // time domain vectors
    double *g, *gp, *gt, *Cg, *h;  // map domain vectors
    double *AtNm1Ah;               // map domain
    double ro, gamma, coeff;       // scalars
    double g2pix, g2pixp, g2pix_polak;

    struct Precond *p = NULL;
    double *pixpond;


    // if we want to use the true norm to compute the residual
    int TRUE_NORM = 1;  //0: No ; 1: Yes

    FILE *fp;

    m = A->m;

    st = MPI_Wtime();

    if (Z_2lvl == 0) Z_2lvl = size;
    int old_npix = A->lcount/A->nnz;
    int *oldlindices = (int *) malloc(old_npix * sizeof(int)); // old num to new num

    build_precond(&p, &pixpond, &n, A, &Nm1, &x, b, noise, cond, lhits, tol, Z_2lvl, precond, nbsamples, sampleIdx, x_init, n_init, old_lindices, old_trashpix);
    // int mapsizeA = (A->lcount-(A->nnz)*(A->trash_pix))/(A->nnz);
    // int *old2new     = (int *) malloc(mapsizeA * sizeof(int)); // old num to new num
    // for (int i = 0; i < old_npix; i++) {
    //   for (int j = 0; j < mapsizeA; j++) {
    //     if (oldlindices[i] == (A->lindices[(A->nnz)*(j+(A->trash_pix))]/(A->nnz)))
    //     {
    //       old2new[j] = i;
    //       // if (rank == 0) {
    //       //   printf("index old2new %i\n", i);
    //       // }
    //     }
    //   }
    // }
    // Update_Initial_Guess(x_init, n_init, x, Neighbours, old_lindices, old_npix, old_trashpix, A, old2new, InterpWeights);
    int mapsizeA = A->lcount-(A->nnz)*(A->trash_pix);
    for(i=0; i< n_init; i++){
      int globidx1 = old_lindices[i+(A->nnz)*old_trashpix];
      for (int j = 0; j < mapsizeA; j++) {
        int globidx2 = A->lindices[j+(A->nnz)*(A->trash_pix)];
        if (globidx1 == globidx2) {
          x[j] = x_init[i];
        }
      }
    }



    fflush(stdout);
    t = MPI_Wtime();
    if (rank == 0) {
        printf("[rank %d] Preconditioner computation time = %lf \n", rank, t - st);
        fflush(stdout);
    }

    // map domain objects memory allocation
    h = (double *) malloc(n * sizeof(double)); // descent direction
    g = (double *) malloc(n * sizeof(double)); // residual
    gp = (double *) malloc(n * sizeof(double)); // residual of previous iteration
    AtNm1Ah = (double *) malloc(n * sizeof(double));

    // time domain objects memory allocation
    Ah = (double *) malloc(m * sizeof(double));

    _g = Ah;
    Cg = AtNm1Ah;
    Nm1Ah = Ah;

    st = MPI_Wtime();

    // Compute RHS
    MatVecProd(A, x, _g, 0);

    for (i = 0; i < m; i++) // To Change with Sequenced Data
      _g[i] = b[i] + noise[i] - _g[i];
        // _g[i] = b[i] + noise[i];

    stbmmProd(Nm1, _g); // _g = Nm1 (Ax-b)

    TrMatVecProd(A, _g, g, 0); // g = At _g

    apply_precond(p, A, &Nm1, g, Cg);

    for (j = 0; j < n; j++) // h = -Cg
        h[j] = Cg[j];

    g2pix = 0.0; // g2 = "res"
    localreduce = 0.0;
    for (i = 0; i < n; i++) // g2 = (Cg, g)
        localreduce += Cg[i] * g[i] * pixpond[i];

    MPI_Allreduce(&localreduce, &g2pix, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    t = MPI_Wtime();
    solve_time += (t - st);

    // Just to check with the true norm:
    if (TRUE_NORM == 1) {
        res = 0.0; // g2 = "res"
        localreduce = 0.0;
        for (i = 0; i < n; i++) // g2 = (Cg, g)
            localreduce += g[i] * g[i] * pixpond[i];

        MPI_Allreduce(&localreduce, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    else {
        res = g2pix;
    }



    double g2pixB = g2pix;
    if (normb > 0) {
      res0 = normb;
    }
    else{
      res0 = res;
    }
    double tol2rel = tol * tol * res0; //tol*tol*g2pixB; //*g2pixB; //tol; //*tol*g2;

    // Test if already converged
    if (rank == 0) {
        res_rel = sqrt(res) / sqrt(res0);
	      printf("k = %d, res = %e, g2pix = %e, res_rel = %e, time = %lf\n", 0, res, g2pix, res_rel, t - st);
        char filename[256];
        sprintf(filename,"%s/pcg_residuals_%s.dat", outpath, ref);
        fp = fopen(filename, "wb");
        fwrite(&res_rel, sizeof(double), 1, fp);
        fflush(stdout);
    }

    if (res <= tol) {
        if (rank == 0)
            printf("--> converged (%e < %e)\n", res, tol);
        k = K; // to not enter inside the loop
    }

    st = MPI_Wtime();
    fflush(stdout);

    // PCG Descent Loop *********************************************
    for (k = 1; k < K ; k++){

        // Swap g backup pointers (Ribière-Polak needs g from previous iteration)
        gt = gp;
        gp = g;
        g  = gt;
        MatVecProd(A, h, Ah, 0);                            // Ah = A h
        // MatVecProdwGaps(A, h, Ah, 0, sampleIdx, nbsamples); // Ah = A h
        stbmmProd(Nm1, Nm1Ah);                              // Nm1Ah = Nm1 Ah   (Nm1Ah == Ah)
        // stbmmProdwGaps(Nm1, Nm1Ah, nbsamples, sampleIdx);                              // Nm1Ah = Nm1 Ah   (Nm1Ah == Ah)
        TrMatVecProd(A, Nm1Ah, AtNm1Ah, 0);                 // AtNm1Ah = At Nm1Ah
        // TrMatVecProdwGaps(A, Nm1Ah, AtNm1Ah, 0, sampleIdx, nbsamples);                 // AtNm1Ah = At Nm1Ah
        coeff = 0.0;
        localreduce = 0.0;
        for (i = 0; i < n; i++)
            localreduce += h[i] * AtNm1Ah[i] * pixpond[i];
        MPI_Allreduce(&localreduce, &coeff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        ro = g2pix / coeff;


        for (j = 0; j < n; j++) // x = x + ro * h
            x[j] = x[j] + ro * h[j];

        for (j = 0; j < n; j++)             // g = g + ro * (At Nm1 A) h
            g[j] = gp[j] - ro * AtNm1Ah[j]; // Use Ribière-Polak formula

	      apply_precond(p, A, &Nm1, g, Cg);
        g2pixp = g2pix; // g2p = "res"
        localreduce = 0.0;
        for (i = 0; i < n; i++) // g2 = (Cg, g)
            localreduce += Cg[i] * g[i] * pixpond[i];

        MPI_Allreduce(&localreduce, &g2pix, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        localreduce = 0.0;
        for (i = 0; i < n; i++) // g2 = (Cg, g)
            localreduce += Cg[i] * (g[i] - gp[i]) * pixpond[i];

        MPI_Allreduce(&localreduce, &g2pix_polak, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        t = MPI_Wtime();
        solve_time += (t - st);

        // Just to check with the true norm:
        if (TRUE_NORM == 1) {
            localreduce = 0.0;
            for (i = 0; i < n; i++) // g2 = (Cg, g)
                localreduce += g[i] * g[i] * pixpond[i];

            MPI_Allreduce(&localreduce, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        else {
            res = g2pix_polak;
        }
        if (rank == 0){ //print iterate info
	          res_rel = sqrt(res) / sqrt(res0);
            printf("k = %d, res = %e, g2pix = %e, res_rel = %e, time = %lf\n", k, res, g2pix_polak, res_rel, t - st);
            fwrite(&res_rel, sizeof(double), 1, fp);
        }

        fflush(stdout);

        if (res <= tol2rel) {
            if (rank == 0) {
                printf("--> converged (%e < %e) \n", res, tol2rel);
                printf("--> i.e. \t (%e < %e) \n", res_rel, tol);
                printf("--> solve time = %lf \n", solve_time);
                fclose(fp);
            }
            break;
        }

        if (g2pix_polak > g2pixp) {
            if (rank == 0)
                printf("--> g2pix > g2pixp pb (%e > %e) \n", g2pix, g2pixp);
        }

        st = MPI_Wtime();

        //gamma = g2pix / g2pixp;
        gamma = g2pix_polak / g2pixp;

        for (j = 0; j < n; j++) // h = h * gamma + Cg
            h[j] = h[j] * gamma + Cg[j];

    } // End loop

    if (k == K) { // check unconverged
        if (rank == 0) {
            printf("--> unconverged, max iterate reached (%le > %le)\n", res, tol2rel);
            fclose(fp);
        }
    }

    if (rank == 0)
        printf("--> g2pix = %e\n", g2pix);
    fflush(stdout);
    free(h);
    free(g);
    free(gp);
    free(AtNm1Ah);
    free(Ah);
    // free(old2new);
    free(oldlindices);
    free_precond(&p);

    return 0;
}

double PCG_GLS_rand(char *outpath, char *ref, Mat *A, Tpltz Nm1, double *x, double *b, double *noise, double *cond, int *lhits, double tol, int K, int precond, int Z_2lvl, int nbsamples, int *sampleIdx)
{
    int    rank, size;
    MPI_Comm_rank(A->comm, &rank);
    MPI_Comm_size(A->comm, &size);
    if (rank == 0)
      printf("##### Start PCG on randomized GLS Pb ################### \n");

    int    i, j, k;     // some indexes
    int    m, n;        // number of local time samples, number of local pixels
    double localreduce; // reduce buffer
    double st, t;       // timers
    double solve_time = 0.0;
    double res, res0, res_rel;
    // double *bcopy;
    double *_g, *ACg, *Ah, *Nm1Ah; // time domain vectors
    double *g, *gp, *gt, *Cg, *h;  // map domain vectors
    double *AtNm1Ah;               // map domain
    double ro, gamma, coeff;       // scalars
    double g2pix, g2pixp, g2pix_polak;

    struct Precond *p = NULL;
    double *pixpond;


    // if we want to use the true norm to compute the residual
    int TRUE_NORM = 1;  //0: No ; 1: Yes

    FILE *fp;

    m = A->m;

    st = MPI_Wtime();
    // bcopy = (double *) malloc(m * sizeof(double)); // copy of signal
    // for (i = 0; i < m; i++) {
    //   bcopy[i] = b[i];
    // }
    if (Z_2lvl == 0) Z_2lvl = size;
    build_precond4rand(&p, &pixpond, &n, A, &Nm1, &x, b, noise, cond, lhits, tol, Z_2lvl, precond, nbsamples, sampleIdx);


    fflush(stdout);
    t = MPI_Wtime();
    if (rank == 0) {
        printf("[rank %d] Preconditioner computation time = %lf \n", rank, t - st);
        fflush(stdout);
    }

    // map domain objects memory allocation
    h = (double *) malloc(n * sizeof(double)); // descent direction
    g = (double *) malloc(n * sizeof(double)); // residual
    gp = (double *) malloc(n * sizeof(double)); // residual of previous iteration
    AtNm1Ah = (double *) malloc(n * sizeof(double));

    // time domain objects memory allocation
    Ah = (double *) malloc(m * sizeof(double));

    _g = Ah;
    Cg = AtNm1Ah;
    Nm1Ah = Ah;

    st = MPI_Wtime();

    // Compute RHS
    // MatVecProd(A, x, _g, 0);
    MatVecProdwGaps(A, x, _g, 0, sampleIdx, nbsamples);

    // for (i = 0; i < m; i++) // To Change with Sequenced Data
    //     _g[i] = b[i] + noise[i] - _g[i];

    for (int ispl = 0; ispl < nbsamples; ispl++) // To Change with Sequenced Data
    {
      int begblk = A->shift[sampleIdx[ispl]  ];
      int endblk = A->shift[sampleIdx[ispl]+1];
      for (int j = begblk; j < endblk; j++) {
        _g[j] = b[j] + noise[j] - _g[j];
      }
    }

    // stbmmProd(Nm1, _g); // _g = Nm1 (Ax-b)
    stbmmProdwGaps(Nm1, _g, nbsamples, sampleIdx); // _g = Nm1 (Ax-b)
    // TrMatVecProd(A, _g, g, 0); // g = At _g
    TrMatVecProdwGaps(A, _g, g, 0, sampleIdx, nbsamples); // g = At _g

    apply_precond(p, A, &Nm1, g, Cg);

    for (j = 0; j < n; j++) // h = -Cg
        h[j] = Cg[j];

    g2pix = 0.0; // g2 = "res"
    localreduce = 0.0;
    for (i = 0; i < n; i++) // g2 = (Cg, g)
        localreduce += Cg[i] * g[i] * pixpond[i];

    MPI_Allreduce(&localreduce, &g2pix, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    t = MPI_Wtime();
    solve_time += (t - st);

    // Just to check with the true norm:
    if (TRUE_NORM == 1) {
        res = 0.0; // g2 = "res"
        localreduce = 0.0;
        for (i = 0; i < n; i++) // g2 = (Cg, g)
            localreduce += g[i] * g[i] * pixpond[i];

        MPI_Allreduce(&localreduce, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    else {
        res = g2pix;
    }

    double g2pixB = g2pix;
    double tol2rel = tol * tol * res; //tol*tol*g2pixB; //*g2pixB; //tol; //*tol*g2;
    res0 = res;
    // Test if already converged
    if (rank == 0) {

        res_rel = sqrt(res) / sqrt(res0);
	      printf("k = %d, res = %e, g2pix = %e, res_rel = %e, time = %lf\n", 0, res, g2pix, res_rel, t - st);
        char filename[256];
        sprintf(filename,"%s/rand_pcg_residuals_%s.dat", outpath, ref);
        fp = fopen(filename, "wb");
        fwrite(&res_rel, sizeof(double), 1, fp);
        fflush(stdout);
    }

    if (res <= tol) {
        if (rank == 0)
            printf("--> converged (%e < %e)\n", res, tol);
        k = K; // to not enter inside the loop
    }

    st = MPI_Wtime();
    fflush(stdout);


    // PCG Descent Loop *********************************************
    for (k = 1; k < K ; k++){

        // Swap g backup pointers (Ribière-Polak needs g from previous iteration)
        gt = gp;
        gp = g;
        g = gt;

        // MatVecProd(A, h, Ah, 0);                            // Ah = A h
        MatVecProdwGaps(A, h, Ah, 0, sampleIdx, nbsamples); // Ah = A h
        // stbmmProd(Nm1, Nm1Ah);                              // Nm1Ah = Nm1 Ah   (Nm1Ah == Ah)
        stbmmProdwGaps(Nm1, Nm1Ah, nbsamples, sampleIdx);                              // Nm1Ah = Nm1 Ah   (Nm1Ah == Ah)
        // TrMatVecProd(A, Nm1Ah, AtNm1Ah, 0);                 // AtNm1Ah = At Nm1Ah
        TrMatVecProdwGaps(A, Nm1Ah, AtNm1Ah, 0, sampleIdx, nbsamples);                 // AtNm1Ah = At Nm1Ah

        coeff = 0.0;
        localreduce = 0.0;
        for (i = 0; i < n; i++)
            localreduce += h[i] * AtNm1Ah[i] * pixpond[i];
        MPI_Allreduce(&localreduce, &coeff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        ro = g2pix / coeff;


        for (j = 0; j < n; j++) // x = x + ro * h
            x[j] = x[j] + ro * h[j];

        for (j = 0; j < n; j++)             // g = g + ro * (At Nm1 A) h
            g[j] = gp[j] - ro * AtNm1Ah[j]; // Use Ribière-Polak formula

	      apply_precond(p, A, &Nm1, g, Cg);

        g2pixp = g2pix; // g2p = "res"
        localreduce = 0.0;
        for (i = 0; i < n; i++) // g2 = (Cg, g)
            localreduce += Cg[i] * g[i] * pixpond[i];

        MPI_Allreduce(&localreduce, &g2pix, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        localreduce = 0.0;
        for (i = 0; i < n; i++) // g2 = (Cg, g)
            localreduce += Cg[i] * (g[i] - gp[i]) * pixpond[i];

        MPI_Allreduce(&localreduce, &g2pix_polak, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        t = MPI_Wtime();
        solve_time += (t - st);

        // Just to check with the true norm:
        if (TRUE_NORM == 1) {
            localreduce = 0.0;
            for (i = 0; i < n; i++) // g2 = (Cg, g)
                localreduce += g[i] * g[i] * pixpond[i];

            MPI_Allreduce(&localreduce, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        else {
            res = g2pix_polak;
        }

        if (rank == 0){ //print iterate info
	          res_rel = sqrt(res) / sqrt(res0);
            printf("k = %d, res = %e, g2pix = %e, res_rel = %e, time = %lf\n", k, res, g2pix_polak, res_rel, t - st);
            fwrite(&res_rel, sizeof(double), 1, fp);
        }

        fflush(stdout);

        if (res <= tol2rel) {
            if (rank == 0) {
                printf("--> converged (%e < %e) \n", res, tol2rel);
                printf("--> i.e. \t (%e < %e) \n", res_rel, tol);
                printf("--> solve time = %lf \n", solve_time);
                fclose(fp);
            }
            break;
        }

        if (g2pix_polak > g2pixp) {
            if (rank == 0)
                printf("--> g2pix > g2pixp pb (%e > %e) \n", g2pix, g2pixp);
        }

        st = MPI_Wtime();

        //gamma = g2pix / g2pixp;
        gamma = g2pix_polak / g2pixp;

        for (j = 0; j < n; j++) // h = h * gamma + Cg
            h[j] = h[j] * gamma + Cg[j];

    } // End loop

    // MatVecProd(A, x, _g, 0);
    // for (i = 0; i < m; i++) // To Change with Sequenced Data
    //     _g[i] = bcopy[i] + noise[i] - _g[i];
    //
    // stbmmProd(Nm1, _g); // _g = Nm1 (Ax-b)
    // TrMatVecProd(A, _g, g, 0); // g = At _g
    // localreduce = 0.0;
    // for (i = 0; i < n; i++) // g2 = (Cg, g)
    //     localreduce += g[i] * g[i] * pixpond[i];
    // MPI_Allreduce(&localreduce, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // if (rank == 0)
    //     printf("--> VERIF of RES on True LS PB : %e \n", res);

    if (k == K) { // check unconverged
        if (rank == 0) {
            printf("--> unconverged, max iterate reached (%le > %le)\n", res, tol2rel);
            fclose(fp);
        }
    }
    if (rank == 0)
      printf(" ----------------------- > End of PCG on randomized GLS \n");
    fflush(stdout);

    if (rank == 0)
        printf("--> g2pix = %e\n", g2pix);
    free(h);
    free(g);
    free(gp);
    free(AtNm1Ah);
    free(Ah);
    // free(bcopy);
    free_precond(&p);

    if (rank == 0)
      printf(" ----------------------- > End of PCG on randomized GLS + Free temporary vectors DONE \n");
    fflush(stdout);

    return res0;
}
