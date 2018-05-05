// Midapack library
// mapmaking code example using the Midapack library - release 1.2b, Nov 2012 
// The routine read data from binaries files and write the result in distributed binaries files

/** @file   test_pcg.c
    @author Frederic Dauvergne
    @date   November 2012 */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <midapack.h>


void usage(){
    printf("usage...\n");
}

//cluster Adamis:
extern const char *WORKDIR="/data/dauvergn/Test_mapmaking/new_data_set/";
//double FKNEE=1.00;
//extern const char *WORKDIR="/data/dauvergn/Test_mapmaking/fred_pack_data/";
double FKNEE=0.25;
//Hoppper:
//extern const char *WORKDIR="/global/homes/d/dauvergn/data/fred_pack_data/";


int main(int argc, char *argv[])
{

  int		M, N, Nnz;  //Global number of rows, of columns, 
                            //of non-zeros values per column for the pointing matrix A
  int		m, n;  //local number of rows, of columns for the pointing matrix A 
  int 		gif;			//global indice for the first local line
  int		i, j, k;
  int           K;	                //maximum number of iteration for the PCG
  double 	tol;			//residual tolerence for the PCG
  Mat	A;			        //pointing matrix structure
  int 		*indices;
  double 	*values;
  int 		pointing_commflag ;	//option for the communication scheme for the pointing matrix
  double	*b, *Ag, *Ad; 	 	//temporal domain vectors
  double	*x, *g, *d, *Ax_b;	//pixel domain vectors
  double        alpha, beta, gamma, resold, resnew;
  double 	localreduce;
  double	st, t;		 	//timer, start time
  int 		output, timer, info;          
  int 		rank, size;	
  
  MPI_Init(&argc, &argv);               
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
  MPI_Comm_size(MPI_COMM_WORLD, &size); 

//communication scheme for the pointing matrix  (to move in .h)
  pointing_commflag=1; //2==BUTTERFLY - 1==RING


//global data caracteristics
  int Nb_t_Intervals = 128;//8;//8;//256;//8;           //total number of stationnary intervals
  int t_Interval_length = pow(2,20);          //length for each stationnary interval
  int LambdaBlock = pow(2,14);//pow(2,14)+1;  //lambda length for each stationnary interval
  double fknee=FKNEE; //0.25;
  Nnz=1;

// PCG parameters
  tol=pow(10,-7);
  K=100;


//total length of the time domaine signal
  M = Nb_t_Intervals*t_Interval_length ;
//compute distribution indexes over the processes
  partition(&gif, &m, M, rank, size);


//  double Nb_t_Intervals_loc_dble = Nb_t_Intervals/size;
  int Nb_t_Intervals_loc = ceil( Nb_t_Intervals/size );
  int nb_proc_shared_one_interval = max(1, size/Nb_t_Intervals ); //same as ceil( (size*1.0)/Nb_t_Intervals );
  printf("[rank %d] nb_proc_shared_one_interval=%d\n", rank, nb_proc_shared_one_interval );

  int t_Interval_length_loc = t_Interval_length/nb_proc_shared_one_interval; //just to check
//should be equal to min(m ,t_Interval_length)

  printf("[rank %d] size=%d \t m=%d \t t_Interval_length=%d\n", rank, size, m, t_Interval_length );
  printf("[rank %d] Nb_t_Intervals_loc=%d \t t_Interval_length_loc=%d\n", rank, Nb_t_Intervals_loc , t_Interval_length_loc); 


//input data memory allocation
  indices  = (int *) malloc(Nnz*m * sizeof(int));     //for pointing matrix indices 
  values  = (double *) malloc(Nnz*m * sizeof(double));//for pointing matrix values  
  b   = (double *) malloc(m*sizeof(double));    //full raw data vector for the signal


  //Read data from files:
//note: work only if the number of processes is a multiple of the number of stationary intervals

//Definition for the input data
  int part_id;      // stationnaly period id number
  int *point_data;  // scann strategy input data for the pointing matrix 
  double *signal;   // signal input data


//Init the pointing matrix values to unity
  for(i=0; i<(Nnz*m); i++) 
    values[i] = 1.;

  int number_in_interval;

//for the case we share the stationary intervals in severals processes
if (nb_proc_shared_one_interval>1) {

  part_id = rank/nb_proc_shared_one_interval; //floor(rank/nb_proc_shared_one_interval);
  number_in_interval = rank%nb_proc_shared_one_interval;
  printf("[rank %d] part_id=%d\n", rank, part_id );  //interval id number
  printf("[rank %d] number_in_interval=%d\n", rank, number_in_interval );

  point_data = (int *) malloc(Nnz*t_Interval_length * sizeof(int));
  signal     = (double *) malloc(t_Interval_length * sizeof(double));

//read the all stationary interval
  ioReadfile(t_Interval_length, part_id, point_data, signal);

  for (i=0; i<(Nnz*m); i++)
    indices[i]=point_data[i+number_in_interval*t_Interval_length_loc];

  for(i=0; i<(m); i++)
    b[i] = signal[i+number_in_interval*t_Interval_length_loc];

  free(point_data);
  free(signal);

}
else { //for the case we dont need to share
//Read the relevants raw inputs data from files distributed by stationary period
//note: Work only for no sharing stationnary interval

  for (k=0; k < Nb_t_Intervals_loc; ++k) { 
    point_data = indices + t_Interval_length*Nnz*k;
    signal = b + t_Interval_length*k;

    part_id = Nb_t_Intervals_loc*rank + k;

    ioReadfile(t_Interval_length, part_id, point_data, signal);

  }//end of the loop over the intervals

}//end if

/*
//m = m/2;
if (rank==0) {
    for( i = 0 ; i < 5 ; ++i) {
      printf(" indices[%d] = %d\n", i, indices[Nnz*(m-5) + i] );
//      printf("values[%d] = %lf\n", i, values[m-5 + i] );
      printf("b[%d] = %lf\n", i, b[m-5 +i] );
    }
}
*/
//sleep(1);

/*
if (rank==0) {
    for( i = 0 ; i < 5 ; ++i) {
      printf(" indices[%d] = %d\n", i, indices[Nnz*(m) + i] );
//      printf("values[%d] = %lf\n", i, values[m + i] );
      printf("b[%d] = %lf\n", i, b[m+i] );
    }
}
*/

/*
if (rank==1) {
    for( i = 0; i < 5; ++i) {
      printf(" indices[%d] = %d\n", i, indices[i] );
      printf("values[%d] = %lf\n", i, values[i] );
      printf("b[%d] = %lf\n", i, b[i] );
    }
}
*/

/*
    for( i = (Nnz*m - 5) ; i < (Nnz*m); ++i) {
      printf(" indices[%d] = %d\n", i, indices[i] );
      printf("values[%d] = %lf\n", i, values[i] );
      printf("b[%d] = %lf\n", i, b[i] );
    }
*/

//MPI_Finalize();
//return 0;


//Pointing matrix init
  MatInit( &A, m, Nnz, indices, values, pointing_commflag, MPI_COMM_WORLD);


// PCG begining vector input definition for the pixel domain map (MatInit gives A.lcount)
  x   = (double *) malloc(A.lcount*sizeof(double));
  for(j=0; j<A.lcount; j++)
    x[j] = 0.;


//Create piecewise Toeplitz matrix

//specifics parameters:
  int nb_blocks_tot = Nb_t_Intervals;
  int n_block_avg = M/nb_blocks_tot;  //should be equal to t_Intervals_length in this example
                                      //because we dont have flotting blocks
  int lambda_block_avg = LambdaBlock; 

//flags for Toeplitz product strategy
  Flag flag_stgy;
  flag_stgy_init_auto(&flag_stgy);

//to print something on screen
  flag_stgy.flag_verbose=1;

//to define fixed bs:
//  flag_stgy.flag_bs = 1;
//  flag_stgy.fixed_bs = pow(2,17);

//define Toeplitz blocks list and structure for Nm1
  Block *tpltzblocks;
  Tpltz Nm1;

//dependants parameters:
  int nrow = M;
  int mcol = 1;  

  int id0 = gif;
  int local_V_size = m;

  int Tsize = lambda_block_avg;
  double *T;  //toeplitz data storage
  T  = (double *) calloc(Tsize ,sizeof(double));

//For one identical block 
  ioReadTpltzfile( Tsize, fknee, T);

//  createT(T, Tsize);


  int nb_blocks_loc;
  nb_blocks_loc = ceil( local_V_size*1.0/n_block_avg );

  double nb_blocks_loc_part =  (local_V_size*1.0)/(n_block_avg) ;


// check special cases to have exact number of local blocks
  if ((id0/n_block_avg + nb_blocks_loc) * n_block_avg < (id0+local_V_size))
    nb_blocks_loc=nb_blocks_loc+1;

  if (rank==0 | rank==1) {
    printf("M=%d, m=%d \n", M, m);
    printf("gif = %d \n", gif);
  }

  int nb_proc_shared_a_block = ceil( size*1.0/nb_blocks_tot );
  int nb_comm = (nb_proc_shared_a_block)-1 ;


//Block definition 
  tpltzblocks = (Block *) malloc(nb_blocks_loc * sizeof(Block));

  defineBlocks_avg(tpltzblocks, T, nb_blocks_loc, n_block_avg, lambda_block_avg, id0 );


  defineTpltz_avg( &Nm1, nrow, 1, mcol, tpltzblocks, nb_blocks_loc, nb_blocks_tot, id0, local_V_size, flag_stgy, MPI_COMM_WORLD);



//print Toeplitz parameters fro check
  if (rank==0 | rank==1) {
    printf("[rank %d] size=%d, nrow=%ld, local_V_size=%d, id0=%ld \n", rank, size, nrow, local_V_size, id0);
    printf("[rank %d] nb_blocks_tot=%d, nb_blocks_loc=%d, n_block_avg=%d, lambda_block_avg=%d \n", rank, nb_blocks_tot, nb_blocks_loc, n_block_avg, lambda_block_avg);
    printf("[rank %d] nb_proc_shared_a_block=%d, nb_comm=%d \n", rank, nb_proc_shared_a_block, nb_comm);
  }


  MPI_Barrier(MPI_COMM_WORLD);
   if(rank==0)
 printf("##### Start PCG ####################\n");
  st=MPI_Wtime();

// Conjugate Gradient
  PCG_GLS_true( A, Nm1, x, b, tol, K);


  MPI_Barrier(MPI_COMM_WORLD);
  t=MPI_Wtime();
   if(rank==0)
 printf("##### End PCG ####################\n");
  if (rank==0) {
    printf("[rank %d] Total PCG time=%lf \n", rank, t-st);
  }


//write output to binaries files:
  int mapsize= A.lcount;
  int map_id=rank;

  int *lstid;
  lstid = (int *) calloc(A.lcount, sizeof(int));

  for(i=0; i< A.lcount; i++)
    lstid[i] = A.lindices[i];


  ioWritebinfile( mapsize, map_id, lstid, x);


//Write some parameters in txt file:
  //output file:
  FILE* file;
  char filenametxt [1024];
  sprintf(filenametxt,"mapout%01d.txt", map_id);
  file = fopen(filenametxt, "w");

  fprintf(file, "%d\n", size );
  fprintf(file, "%d\n", gif );
  fprintf(file, "%d\n", m );
  fprintf(file, "%d\n", A.lcount );
  fprintf(file, "%d\n", Nb_t_Intervals );
  fprintf(file, "%d\n", t_Interval_length );
  fprintf(file, "%d\n", LambdaBlock );
  fprintf(file, "%d\n", Nb_t_Intervals_loc );
  fprintf(file, "%d\n", rank );
  fprintf(file, "size idp m A.lcount Nb_t_Intervals t_Interval_length LambdaBlock Nb_t_Intervals_loc rank\n" );
  fclose(file);


  MatFree(&A);                                                //free memory  

  free(indices);
  free(values);

  free(b);
  free(x);
  MPI_Finalize();
 
 
  return 0;
}



int partition(int *fi, int *wi, int N, int me, int us){
  int r, k;
  k = N / us;
  r = N - k*us;
  if( me < r){
    *fi = (k+1) * me;
    *wi = k+1;
  }
  else{
    *fi = r*(k+1) + k*(me-r);
    *wi = k;
    }
  return 0;
}



