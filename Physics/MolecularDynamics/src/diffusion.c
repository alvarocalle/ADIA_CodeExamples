/* ------------------------------------------------------------
   MOLECULAR DINAMICS
---------------------------------------------------------------
   DIFFUSION

   Study the diffusion of particles in a dilute system. 

   a) Take 16 particles in a 10x10 box and calculate the
   mean-square displacement of an atom as a function of time.
   Show that the mpotion is diffusive and find the constant D.

   b) Study how D varies with density.

---------------------------------------------------------------
   
   x(i,n),y(i,n) = position of particle i
   n = 1,2,3 = oldest, current, and new positions
   vx,vy = velocity components
   n_particles = number of particles
   len = size of box (use periodic boundary conditions)
   dt = time step
   tagged particles = 1 and 2

--------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rand.h"

#define NMAX 50
#define STEPS 3
#define STOP 10
#define TIME 500
#define SEED (-8743)

float sgn(float x);
float init_velocity(float *vx, float *vy, float v0);
void pair_force(float r, float *f, float *u);
void initialize(float x[STEPS][NMAX],float y[STEPS][NMAX],
		float vx[NMAX], float vy[NMAX], int n_particles,
		int len, float dt, int n_plot);
void find_r(int i, int n, int flag, float x[STEPS][NMAX], float y[STEPS][NMAX],
	    int len, float *r, float *dx, float *dy);
void force(int n, float x[STEPS][NMAX], float y[STEPS][NMAX],
	   float n_particles, int len, float *fx, float *fy);
void update(float x[STEPS][NMAX], float y[STEPS][NMAX], float vx[NMAX],
	    float vy[NMAX], int n_particles, int len, float dt,
	    float *x1, float *y1, float *r2, float *x2, float *y2, float *rp2);

long ldum = SEED;
int n_plot = 3;
int n_particles = 20, len = 10;
float dt = 0.02;

int main ()
{
    float x[STEPS][NMAX], y[STEPS][NMAX], vx[NMAX], vy[NMAX];
    float energy[TIME], mytime[TIME], temperature[TIME];
    float sqpos[TIME], sqpair[TIME];
    float t, x1, y1, x2, y2, r2, rp2;
    int i, j, n_p, t_loop;
    FILE *fich;

    initialize(x, y, vx, vy, n_particles, len, dt, n_plot);

    r2 = 0.0;
    x1 = 0.0;
    y1 = 0.0;
    x2 = x[2][2] - x[2][1];
    y2 = y[2][2] - y[2][1];

    t_loop = 0;
    n_p = 0;
    i = 0;
    j = 0;

    while( t_loop <= TIME ){
	
	t_loop++;

	// update system
	update(x,y,vx,vy,n_particles,len,dt,&x1,&y1,&r2,&x2,&y2,&rp2);

	t = t + dt;
	j = j + 1;

	//  display_screen if j >= nplot
	if ( j >= n_plot ){  

	    j = 0;
	    i ++ ;

	    mytime[i] = t;
	    sqpos[i] = r2;
	    sqpair[i] = rp2;

	    n_p += 1;  
	}
    } 

    /* Outputs files */

    // Square displacement time series 
    fich = fopen("displac.dl","w");   
    fprintf(fich,"# Tagged particle square displacement time series\n"); 
    fprintf(fich,"# Time (in L-J units), r^2 (in sigma^2)\n"); 
    for(i = 1; i <= n_p; i++)
	fprintf(fich," %f %f \n", mytime[i], sqpos[i]);
    fclose(fich);

    // Pair square distance time series
    fich = fopen("pair-dist.dl","w");   
    fprintf(fich,"# Tagged pair square distance time series\n"); 
    fprintf(fich,"# Time (in L-J units), dr^2 (in sigma^2)\n"); 
    for(i = 1; i <= n_p; i++)
	fprintf(fich," %f %f \n", mytime[i], sqpair[i]);
    fclose(fich);

    return 0;
}

void initialize(float x[STEPS][NMAX],float y[STEPS][NMAX],
		float vx[NMAX], float vy[NMAX], int n_particles,
		int len, float dt, int n_plot){
/*
    Initialize variables: I consider a fairly diluite gas.
    Even in a gas, the atoms are not arranged completely at random. To take
    this into account it is convenient in our initialization to give the atoms
    an approximately regular arrangement. Then first placing the atoms on a
    square lattice in which the spacing between nearest neighbors is grater 
    than 2s (s = 1) and then displacing the atom from these locations at random
    by a distance <= s/2.
*/    
    int n, i, j;
    float vmax = 1.;
    float grid;
    float rnd;
    
    grid = len / (int) (sqrt(n_particles) + 1);
    rnd = (float) ran2(&ldum);

    n = 0;
    i = 0;
    
    while( i < len ){ 
	j = 0;
	while (j < len ){
	    n ++ ;
	    if ( n <= n_particles ){ 

//	     	rnd = (float) ran2(&ldum);

		x[2][n] = i + (rnd - 0.5) * grid / 2.;
		y[2][n] = j + (rnd - 0.5) * grid / 2.;

		init_velocity(&vx[n],&vy[n],vmax);

		x[1][n] = x[2][n] - vx[n] * dt;
		y[1][n] = y[2][n] - vy[n] * dt;
	    }
	    j += grid;
	}
	i += grid;
    }
}

float init_velocity(float *vx, float *vy, float v0){
/*  Initialize velocities randomly */

    float rnd;
    rnd = (float) ran2(&ldum);

    // Random velocities
    (*vx) = v0 * (rnd - 0.5) * sqrt(2.);
    (*vy) = v0 * (rnd - 0.5) * sqrt(2.); 

    // All velocities v0
//    (*vx) = (*vy) = v0; 

    // Semi-random velocities
//    (*vx) = v0; 
//    (*vy) = v0 * (rnd - 0.5); 
}

void pair_force(float r, float *f, float *u){
/*  Lennard-Jones force */

    (*u) = 4. * ( 1./pow(r,12) - 1./pow(r,6) ); //  LJ potential in scaled units
    (*f) = 24. * ( 2./pow(r,13) - 1./pow(r,7) ); //  LJ force in scaled units
}

void find_r(int i, int n, int flag, float x[STEPS][NMAX], float y[STEPS][NMAX],
	    int len, float *r, float *dx, float *dy){
/*  Find spacing taking periodic boundary conditions into account */
    
    (*dx) = x[flag][n] - x[flag][i]; 
    (*dy) = y[flag][n] - y[flag][i];

    if ( abs(*dx) > (len / 2.) ) (*dx) -= sgn(*dx) * len;
    if ( abs(*dy) > (len / 2.) ) (*dy) -= sgn(*dy) * len;

    (*r) = sqrt( (*dx)*(*dx) + (*dy)*(*dy) );
}

void force(int n, float x[STEPS][NMAX], float y[STEPS][NMAX],
	    float n_particles, int len, float *fx, float *fy){
/*  Compute forces on all of the particles */

    float r, f, u, dx, dy;
    int i;

    (*fx) = 0.0,
    (*fy) = 0.0;

    for (i = 1; i<= n_particles; i++){
	if ( i != n ){
	    find_r(i, n, 2, x, y, len, &r, &dx, &dy);
	    if (r < 3. ){
		pair_force(r, &f, &u);
		(*fx) += f * dx / r;
		(*fy) += f * dy / r;
	    }
	}
    }
}

void update(float x[STEPS][NMAX], float y[STEPS][NMAX], float vx[NMAX],
	    float vy[NMAX], int n_particles, int len, float dt,
	    float *x1, float *y1, float *r2, float *x2, float *y2, float *rp2){

/*  Move forward one time step */

    float x_new[100], y_new[100];
    float fx, fy;
    int i;

    for (i = 1; i <= n_particles; i++){

	// compute the forces
	force(i, x, y, n_particles, len, &fx, &fy);

	// use Verlet method
	x_new[i] = 2. * x[2][i] - x[1][i] + fx * dt*dt;
	y_new[i] = 2. * y[2][i] - y[1][i] + fy * dt*dt;

	// keep track of velocities
	vx[i] = (x_new[i] - x[1][i]) / (2. * dt);
	vy[i] = (y_new[i] - y[1][i]) / (2. * dt);

	if(i == 1){ 
	    (*x1) = (*x1) + x_new[1] - x[2][1];
	    (*y1) = (*y1) + y_new[1] - y[2][1];
	    (*r2) = (*x1) * (*x1) + (*y1) * (*y1);
	}
	else if(i == 2){
	    (*x2) = (*x2) + x_new[2] - x[2][2];
	    (*y2) = (*y2) + y_new[2] - y[2][2];
	    (*rp2) = pow((*x2 - *x1),2) + pow((*y2 - *y1),2);
	}

	// periodic boundary conditions
	if ( x_new[i] < 0 ){
	    x_new[i] = x_new[i] + len;
	    x[2][i] = x[2][i] + len;
	}
	else if ( x_new[i] > len ){
	    x_new[i] = x_new[i] - len;
	    x[2][i] = x[2][i] - len;
	}
	if ( y_new[i] < 0 ){
	    y_new[i] = y_new[i] + len;
	    y[2][i] = y[2][i] + len;
	}
	else if ( y_new[i] > len ){
	    y_new[i] = y_new[i] - len; 
	    y[2][i] = y[2][i] - len;
	}
    }

    // update current and old values
    for ( i = 1; i <= n_particles; i++ ){
	x[1][i] = x[2][i];                           
	x[2][i] = x_new[i];
	y[1][i] = y[2][i];
	y[2][i] = y_new[i];
    }
}

float sgn(float x) {                    
/* signed function */

    if( x < 0 )
	return -1.0;
    else if( x > 0 )
	return 1.0; 
}
