/* Gaussian elimination without pivoting.
 *  * Compile with "gcc gauss.c" 
 *   */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 *  * You need not submit the provided code.
 *   */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>
/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N;  /* Matrix size */

/*-------------------New Variables as part of this Project-----------------*/
int NUMTHREADS; /* Number of threads to use */
int NORM = 0; /* Variable shared by threads to keep a track of current norm */
pthread_mutex_t mutex; /* Mutex to synchronize NORM calculations */
pthread_cond_t cv; /* Condition Variable to Broadcast / Wait for calculations from other threads */

/* Structure to send data to pthreads */
typedef struct info_t {
	int start; /* Starting row */
	int end; /* Ending row */
} info;
/* ------------------------------------ Done -------------------------------*/

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
		* It is this routine that is timed.
		* 		* It is called only on the parent.
		* 				*/

/* returns a seed for srand based on the time */
unsigned int time_seed() {
	struct timeval t;
	struct timezone tzdummy;

	gettimeofday(&t, &tzdummy);
	return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
	int seed = 0;  /* Random seed */
	char uid[32]; /*User name */

	/* Read command-line arguments */
	srand(time_seed());  /* Randomize */

	if (argc == 4) {
		seed = atoi(argv[2]);
		srand(seed);
		printf("Random seed = %i\n", seed);
	} 
	if (argc >= 3) {
		N = atoi(argv[1]);
		if (N < 1 || N > MAXN) {
			printf("N = %i is out of range.\n", N);
			exit(0);
		}
	}
	else {
		printf("Usage: %s <matrix_dimension> [random seed]\n",
				argv[0]);    
		exit(0);
	}

	/* Print parameters */
	printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
	int row, col;

	printf("\nInitializing...\n");
	for (col = 0; col < N; col++) {
		for (row = 0; row < N; row++) {
			A[row][col] = (float)rand() / 32768.0;
		}
		B[col] = (float)rand() / 32768.0;
		X[col] = 0.0;
	}

}

/* Print input matrices */
void print_inputs() {
	int row, col;
	if (N< 10) {
		printf("\nA =\n\t");
		for (row = 0; row < N; row++) {
			for (col = 0; col < N; col++) {
				printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
			}
		}
		printf("\nB = [");
		for (col = 0; col < N; col++) {
			printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
		}
	}
}

void print_X() {
	int row;

	if (N < 100) {
		printf("\nX = [");
		for (row = 0; row < N; row++) {
			printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
		}
	}
}

int main(int argc, char **argv) {
	/* Timing variables */
	struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
	struct timezone tzdummy;
	clock_t etstart2, etstop2;  /* Elapsed times using times() */
	unsigned long long usecstart, usecstop;
	struct tms cputstart, cputstop;  /* CPU times for my processes */

	/* Initialize the number of threads */
	if (argv[3] == NULL) {
		printf("Usage: %s <matrix_dimension> [random seed] <number of threads>\n",
				argv[0]);    
		exit(0);
	}
	NUMTHREADS = atoi(argv[3]);

	/* Process program parameters */
	parameters(argc, argv);

	/* Initialize A and B */
	initialize_inputs();

	/* Print input matrices */
	print_inputs();

	/* Start Clock */
	printf("\nStarting clock.\n");
	gettimeofday(&etstart, &tzdummy);
	etstart2 = times(&cputstart);

	/* Gaussian Elimination */
	gauss();

	/* Stop Clock */
	gettimeofday(&etstop, &tzdummy);
	etstop2 = times(&cputstop);
	printf("Stopped clock.\n");
	usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
	usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

	/* Display output */
	print_X();

	/* Display timing results */
	printf("\nElapsed time = %g ms.\n",
			(float)(usecstop - usecstart)/(float)1000);

	printf("(CPU times are accurate to the nearest %g ms)\n",
			1.0/(float)CLOCKS_PER_SEC * 1000.0);
	printf("My total CPU time for parent = %g ms.\n",
			(float)( (cputstop.tms_utime + cputstop.tms_stime) -
				(cputstart.tms_utime + cputstart.tms_stime) ) /
			(float)CLOCKS_PER_SEC * 1000);
	printf("My system CPU time for parent = %g ms.\n",
			(float)(cputstop.tms_stime - cputstart.tms_stime) /
			(float)CLOCKS_PER_SEC * 1000);
	printf("My total CPU time for child processes = %g ms.\n",
			(float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
				(cputstart.tms_cutime + cputstart.tms_cstime) ) /
			(float)CLOCKS_PER_SEC * 1000);
	/* Contrary to the man pages, this appears not to include the parent */
	printf("--------------------------------------------\n");

	exit(0);
}

void *parallel_row(info * row_data) {
	int col;
	int row, norm;

	for (norm = 0 ; norm < row_data->end ; norm++) {
		/* All threads calculate norm 0 and wait */
		for (row = row_data->start; row <= row_data->end; row++) {
			if (row > norm) {
				float multiplier = A[row][norm] / A[norm][norm]; 
				for (col = norm; col < N; col++) {
					A[row][col] -= A[norm][col] * (multiplier);
				}
				B[row] -= B[norm] * multiplier;
			}
		}

		pthread_mutex_lock(&mutex);
		if ((row_data->start <= NORM+1) && (NORM+1 <= row_data->end)) { 
			/* I need to broadcast and I am not the last thread in the list */
			if (!(NORM == row_data->end)) {
				NORM = NORM + 1;
				pthread_cond_broadcast(&cv);
			}
		}  else {
			/* I need to wait for norm calculation from a previous thread */
			if ((!(row_data->end == NORM) || NORM<=norm) && ((NORM + 1) < N)) {
				pthread_cond_wait(&cv, &mutex); 
			}
		}
		pthread_mutex_unlock(&mutex);
	}
	free(row_data);
	pthread_exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 *  * defined in the beginning of this code.  X[] is initialized to zeros.
 *   */
void gauss() {
	int norm, row, col, i;  /* Normalization row, and zeroing
				 * element row and col */
	float multiplier;

	printf("Computing Using Pthreads.\n");

	int remainder, total_num_rows, rows_per_process, sent, num_rows_to_send, num_threads = 0;

	/* Gaussian elimination */
	
	/* Keep track of spawned threads */
	pthread_t tid[MAXN];

	/* Calculate number of rows to distribute */
	total_num_rows = N;

	/* Calculate how many rows each process will get */
	rows_per_process = total_num_rows/NUMTHREADS;
	if (total_num_rows > (NUMTHREADS * rows_per_process)) {
		remainder = (total_num_rows) % (NUMTHREADS);
	}

	/* Variable to keep track of rows distributed so far */
	/* Initial value is 1 as no calculation is needed for row 0 */
	sent = 1;

	for (i = 0; i< NUMTHREADS; i++) {
		if (remainder > 0 ) {
			num_rows_to_send = rows_per_process + 1;
			remainder--;
		} else {
			num_rows_to_send = rows_per_process;
		}

		if (num_rows_to_send > (N- sent)) {
			num_rows_to_send = N - sent;
		}

		info * data = malloc(sizeof(info));
		/* Calculate starting row */
		data->start = sent;

		/* Calculate ending row */
		data->end = data->start + num_rows_to_send - 1;
		if (data->end >= N) data->end = N-1; 

		/* Lets send the data */
		if (data->start <= data->end) {
			pthread_create(&tid[num_threads], NULL, (void *)parallel_row, data);
			sent = sent + num_rows_to_send;
			num_threads++;
		}
	}

	/* Distribution Done.. Let's wait for threads to finish computation */
	for (i = 0; i < num_threads; i++) {
		pthread_join(tid[i], NULL);
	}

	/* (Diagonal elements are not normalized to 1.  This is treated in back
	 *    * substitution.)
	 *       */


	/* Back substitution */
	for (row = N - 1; row >= 0; row--) {
		X[row] = B[row];
		for (col = N-1; col > row; col--) {
			X[row] -= A[row][col] * X[col];
		}
		X[row] /= A[row][row];
	}
}
