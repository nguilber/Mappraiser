// Midapack library
// mapmaking code example using the Midapack library - release 1.2b, Nov 2012 
// Utilitary IO routines for the mapmaking example. This is not contains healpix dependancy.

/** @file   iofiles.c
    @author Frederic Dauvergne
    @date   November 2012 */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <midapack.h>

#include <stdbool.h>
#include <errno.h>
#include <unistd.h>


//extern char *WORKDIR="/data/dauvergn/Test_mapmaking/fred_pack_data/";
//char *WORKDIR="/data/dauvergn/Test_mapmaking/fred_pack_data/";
char *WORKDIR;
int IOVERBOSE=0;

int ioReadfile( int block_size, int part_id, unsigned int *point_data, double *signal)
{
        int i;

        char p_vectorFile[256];
        char *p_vectorFileNameBasis = "point_data_0_";

        char s_vectorFile[256];
//        char *s_vectorFileNameBasis = "signal_";
        char *s_vectorFileNameBasis = "pure_signal_";

        FILE *fp;

        sprintf(p_vectorFile, "%s%s%01d.dat", WORKDIR, p_vectorFileNameBasis, part_id);
        sprintf(s_vectorFile, "%s%s%01d.dat", WORKDIR, s_vectorFileNameBasis, part_id); 

        if (IOVERBOSE>1) {
          printf(" Pointing file name: %s\n", p_vectorFile);
          printf("   Signal file name: %s\n", s_vectorFile);
	}

        fp=fopen(p_vectorFile, "rb");
        fread(point_data, sizeof(unsigned int), block_size, fp);
        fclose(fp);

        fp=fopen(s_vectorFile, "rb");
        fread(signal, sizeof(double), block_size, fp);
        fclose(fp);

        return 0;
}


int ioReadfilePure( int block_size, int part_id, unsigned int *point_data, double *signal)
{
        int i;

        char p_vectorFile[256];
        char *p_vectorFileNameBasis = "point_data_0_";

        char s_vectorFile[256];
//        char *s_vectorFileNameBasis = "signal_";
        char *s_vectorFileNameBasis = "pure_signal_";

        FILE *fp;

        sprintf(p_vectorFile, "%s%s%01d.dat", WORKDIR, p_vectorFileNameBasis, part_id);
        sprintf(s_vectorFile, "%s%s%01d.dat", WORKDIR, s_vectorFileNameBasis, part_id);

        printf(" Pointing file name: %s\n", p_vectorFile);
        printf("   Signal file name: %s\n", s_vectorFile);


        fp=fopen(p_vectorFile, "rb");
        fread(point_data, sizeof(unsigned int), block_size, fp);
        fclose(fp);

        fp=fopen(s_vectorFile, "rb");
        fread(signal, sizeof(double), block_size, fp);
        fclose(fp);

        return 0;
}



int ioReadrandom( int block_size, int part_id, unsigned int *point_data, double *signal)
{
        int i;

//Random generator:
  srand(part_id);                                 //initialize the random generator
  for(i=0; i<block_size; i++)
    signal[i] = 1.0 + (10*((double) rand()) / RAND_MAX -1);


  for(i=0; i<block_size; i++)
    point_data[i] = i;


  return 0;
}



int ioReadTpltzfile( int lambda, double num, double *Tblock)
{

        int i;

        char N_vectorFile[256];
        char *N_vectorFileNameBasis = "inv_tt_x";

        FILE *fp;

        sprintf(N_vectorFile, "%s%s%1.2f.bin", WORKDIR, N_vectorFileNameBasis, num);

        printf(" Block Toeplitz values file name: %s\n", N_vectorFile);

        fp=fopen(N_vectorFile, "rb");
        fread(Tblock, sizeof(double), lambda, fp);

        fclose(fp);


        return 0;
}


int ioReadTpltzrandom( int lambda, double *Tblock)
{
        int i;
	double lambdareduce=10;

    srand (lambda); //init seed

  //input matrix definition of T
    for(i=1;i<lambdareduce;i++) {
      Tblock[i]= -1.0/((double) i);
    }
    for(i=lambdareduce;i<lambda;i++) {
      Tblock[i]= rand()/((double) 100*RAND_MAX);
    }

    Tblock[0] = 10;


        return 0;
}



int ioWritebinfile( int mapsize, int mappart_id, int *lstid, double *map)
{
        int i;

        char lstid_vectorFile[256];
        char x_vectorFile[256];
        char *lstid_vectorFileNameBasis = "mapout_lstid_";
        char *x_vectorFileNameBasis = "mapout_";

        FILE *fp;

        sprintf(lstid_vectorFile, "%s%01d.dat", lstid_vectorFileNameBasis, mappart_id);
        sprintf(x_vectorFile, "%s%01d.dat", x_vectorFileNameBasis, mappart_id);

//        printf(" Map file name: %s\n", lstid_vectorFile);
//        printf(" Map file name: %s\n", x_vectorFile);


        fp=fopen(lstid_vectorFile, "wb");
        fwrite(lstid, sizeof(int), mapsize, fp);
        fclose(fp);

        fp=fopen(x_vectorFile, "wb");
        fwrite(map, sizeof(double), mapsize, fp);
	fclose(fp);


        return 0;
}




int ioReadbinfile( int mapsize, int mappart_id, int *lstid, double *map)
{

        int i;

        char lstid_vectorFile[256];
        char x_vectorFile[256];
        char *lstid_vectorFileNameBasis = "mapout_lstid_";
        char *x_vectorFileNameBasis = "mapout_";

        FILE *fp;

        sprintf(lstid_vectorFile, "%s%01d.dat", lstid_vectorFileNameBasis, mappart_id);
        sprintf(x_vectorFile, "%s%01d.dat", x_vectorFileNameBasis, mappart_id);

        printf(" Map id file name: %s\n", lstid_vectorFile);
        printf(" Map file name: %s\n", x_vectorFile);


        fp=fopen(lstid_vectorFile, "rb");
        fread(lstid, sizeof(int), mapsize, fp);
        fclose(fp);

        fp=fopen(x_vectorFile, "rb");
        fread(map, sizeof(double), mapsize, fp);
        fclose(fp);


        return 0;
}


