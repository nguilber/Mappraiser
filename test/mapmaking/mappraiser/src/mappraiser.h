/** @file mappraiser.h
    @brief <b> Declaration of the backbone routines of the map-making code.</b>
    @author Hamza El Bouhargani
    @date May 2019 */

/* Define the banded block Toeplitz matrix */
int defineTpltz_avg( Tpltz *Nm1, int64_t nrow, int m_cw, int m_rw, Block *tpltzblocks, int nb_blocks_loc, int nb_blocks_tot, int64_t idp, int local_V_size, Flag flag_stgy, MPI_Comm comm);

int defineBlocks_avg(Block *tpltzblocks, double *T, int nb_blocks_loc, void *local_blocks_sizes, int lambda_block_avg, int64_t id0 );

/* IO routines */
int ioWritebinfile( int mapsize, int mappart_id, int *lstid, double *map, double *cond, int *lhits);

int ioReadbinfile( int mapsize, int mappart_id, int *lstid, double *map, double *cond, int *lhits);

//Import HEALPix routines with little tweaks to pass datatypes in arguments
static void util_fail_ (const char *file, int line, const char *func,
  const char *msg);

#if defined (__GNUC__)
#define UTIL_FUNC_NAME__ __func__
#else
#define UTIL_FUNC_NAME__ "unknown"
#endif
#define UTIL_ASSERT(cond,msg) \
  if(!(cond)) util_fail_(__FILE__,__LINE__,UTIL_FUNC_NAME__,msg)
#define UTIL_FAIL(msg) \
  util_fail_(__FILE__,__LINE__,UTIL_FUNC_NAME__,msg)

static void setCoordSysHP(char coordsys,char *coordsys9);

static void printerror (int status);

void write_map (void *signal, int type, long nside, const char *filename,
  char nest, const char *coordsys);

/* Preconditioner routines */

// Block-Jacobi preconditioner
int precondblockjacobilike(Mat *A, Tpltz Nm1, Mat *BJ, double *b, double *cond, int *lhits);

// Point Jacobi preconditioner
int precondjacobilike(Mat A, Tpltz Nm1, int *lhits, double *cond, double *vpixDiag);

// Local product A^T * diagNM1 * A
int getlocalW(Mat *A, Tpltz Nm1, double *vpixBlock, int *lhits);

int getlocDiagN(Mat *A, Tpltz Nm1, double *vpixDiag);

int DiagAtA(Mat *A, double *diag, int pflag);












// Check is there are some NaN or Inf in an array of double
int Check_Inf_NaN(double *array, int size_array, int rank);

// Build (A_i.T * A_i) bloc of size nnz*nnz operator
int Build_ATA_bloc(Mat *A, Tpltz Nm1, double *ATA, int row_indice, int nb_rows, int np, int rank, int bloc);

// Apply the product A_i.T * v for a timestream of one block
int Apply_ATr_bloc(Mat *A, double *x, double *y, int size_y, int row_indice, int nb_rows);

// Apply the product (A_i.T * A_i)^{-1} * v for a timestream of one block
int Apply_ATA_bloc(Mat *A, double *ATA, double *y, double *z, int np);

// Get the fourier mode of order index_mode through fftw3 package
int get_Fourier_mode(fftw_complex *in, fftw_complex *out, int size, int index_mode);

// Build the coarse space from eigenvector of the blocks
int Build_ALS_proc(Mat *A, Tpltz Nm1, double *CS, int nb_defl, int n, int rank);

// Build the coarse space from eigenvector of the blocks
int Build_ALS(Mat *A, Tpltz Nm1, double *Z, int nb_defl, int np, int rank);

// Build a orthonormal basis of a coarse space Z
int Orthogonalize_Space_loc(double **Z, int nb_rows, int nb_cols, double tol_svd, int rank);

// Get the total size of the coarse over all the processes
int Communicate_sizes(int new_size, MPI_Comm comm);

// Communicate the coarse space array between procs
double * Communicate_CS(double *Z, int new_size, int* tot_size_CS, MPI_Comm comm, int nb_rows);

/* // Compute the Cholesky factor of the Coarse matrix E (ref. Tang, Nabben, Vuik, Erlan notation) */
/* int Factorize_CS(Mat *A, Tpltz Nm1, double *CS, double *E); */

/* // Apply the ADEF1 2lvl preconditionner to a vector (ref. Tang, Nabben, Vuik, Erlan notation) */
/* int Apply_ADEF1(Mat BJ, Mat *A, Tpltz Nm1, double *CS, double *E, double *in, double *out); */











// Communication routine for building the pixel blocks of the preconditioner
int commScheme(Mat *A, double *vpixDiag, int pflag);

/* PCG routines */

// Pixel share ponderation to deal with overlapping pixels between multiple MPI procs
int get_pixshare_pond(Mat *A, double *pixpond);

//PCG routine
int PCG_GLS_true(char *outpath, char *ref, Mat *A, Tpltz Nm1, double *x, double *b, double *noise, double *cond, int *lhits, double tol, int K);

//ECG routine
int ECG_GLS(char *outpath, char *ref, Mat *A, Tpltz Nm1, double *x, double *b, double *noise, double *cond, int *lhits, double tol, int maxIter, int enlFac, int ortho_alg, int bs_red);
