/* ---------------------------------------------------------------
   MOLECULAR DINAMICS
------------------------------------------------------------------
   TEMPERATURE

   Calculate the speed distrubution for a dilute gas and compare
   the results with the Maxwell distribution. Extract the 
   temperature and compare the value found with the calculated
   directly from the equipartition theorem.

------------------------------------------------------------------
   
   x(i,n),y(i,n) = position of particle i
   n = 1,2,3 = oldest, current, and new positions
   vx,vy = velocity components
   n_particles = number of particles
   len = size of box (use periodic boundary conditions)
   dt = time step

------------------------------------------------------------------ */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rand.h"

#define PI 3.141592654
#define NMAX 100
#define STEPS 3
#define TIME 5000
#define SEED (-8743)
#define Lmax 100

long ldum = SEED;
int n_plot = 3;
int n_particles = 20, len = 10;
float dt = 0.02;

/* --------------------------------------------------------------- */

float sgn(float x);
void pair_force(float r, float *f, float *u);
float init_velocity(float *vx, float *vy, float v0);
void initialize(float x[STEPS][NMAX],float y[STEPS][NMAX],
		float vx[NMAX], float vy[NMAX], int n_particles,
		int len, float dt, int n_plot);
void find_r(int i, int n, int flag, float x[STEPS][NMAX], float y[STEPS][NMAX],
	    int len, float *r, float *dx, float *dy);
void force(int n, float x[STEPS][NMAX], float y[STEPS][NMAX],
	   float n_particles, int len, float *fx, float *fy);
void update(float x[STEPS][NMAX], float y[STEPS][NMAX],
	    float vx[NMAX], float vy[NMAX],
	    int n_particles, int len, float dt);
void calc_energy(float x[STEPS][NMAX], float y[STEPS][NMAX],
		 float vx[NMAX], float vy[NMAX], int n_particle,
		 int len, float *e, float *pot_e, float *k_e, float *temp);
void to_histogram(float v[Lmax], float x, float a, float b, int n);

/* --------------------------------------------------------------- */

int main ()
{
    float energy[TIME], mytime[TIME], temperature[TIME];
    float x[STEPS][NMAX], y[STEPS][NMAX], vx[NMAX], vy[NMAX], v[NMAX];
    float t, e, pot_e, k_e, temp, T;
    int i, np, tmin, tmax;

    FILE *fich_histo;

    int L = 100;
    float hvx, hvy, hv, pos, Pv;
    float dh, Norm, LH = 0., GH = 5.;
    float histo_vx[Lmax], histo_vy[Lmax], histo_v[Lmax];

    fich_histo = fopen("histogram_v.sal","w");
    fprintf(fich_histo,"# SPEED HISTOGRAM \n# v, P(vx), P(vy), P(v), Pmaxw(v)\n");

    /* Initial configuration */
    initialize(x, y, vx, vy, n_particles, len, dt, n_plot);

    for(i = 1; i <= n_particles; i++){
	
	v[i] = sqrt( vx[i]*vx[i] + vy[i]*vy[i]);

	to_histogram (histo_vx, vx[i], LH, GH, L);
	to_histogram (histo_vy, vy[i], LH, GH, L);
	to_histogram (histo_v, v[i], LH, GH, L);
    }

    tmin = 2500.0;
    tmax = 4500.0;

    np = 0;
    t = 0.0;

    /* Start evolution */
    while(np < TIME){
	
	np ++;
	
 	/* update system */
	update(x,y,vx,vy,n_particles,len,dt);

	t += dt ;

	/* calculate and record time, energy and temperature */
	mytime[np] = t;    
	calc_energy(x, y, vx, vy, n_particles, len, &e, &pot_e, &k_e, &temp);
	energy[np] = e;
	temperature[np] = temp; 
		
	/*  make a velocity histogram when tmin < t < tmax */
	if( (np >= tmin) && (np <= tmax) ){
	    for(i = 1; i <= n_particles; i++){
		v[i] = sqrt( vx[i]*vx[i] + vy[i]*vy[i]);	
		to_histogram (histo_vx, vx[i], LH, GH, L);
		to_histogram (histo_vy, vy[i], LH, GH, L);
		to_histogram (histo_v, v[i], LH, GH, L);
	    }
	}
    }
    
    dh = (GH - LH) / L;
    Norm = (float) ( L * (tmax - tmin) * n_particles);

    for(i = 0; i < L; i++){

	hvx = histo_vx[i];
	hvy = histo_vy[i];
	hv = histo_v[i];

	pos = LH + (float)i * dh;
	Pv = sqrt(2./PI) * (pos * pos) * exp( -  pos * pos / 2. );

	fprintf(fich_histo,"%f %f %f %f %f\n",pos,hvx/Norm,hvy/Norm,hv/Norm,Pv);
    }

    fclose(fich_histo);

    for(i = 0; i < np; i++)
	T += temperature[i];     

    printf("\nTemperature via equipartition theorem:\n\n\t T = %f\n\n", T/np);

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
    float vx_aux, vy_aux;
    float grid;
    float rnd;
    
    grid = len / (int) (sqrt(n_particles) + 1);
//    rnd = (float) ran2(&ldum);

    n = 0;
    i = 0;
    
    while( i < len ){ 
	j = 0;
	while (j < len ){
	    n ++ ;
	    if ( n <= n_particles ){ 

	     	rnd = (float) ran2(&ldum);

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
//    (*vx) = (*vy) = v0 * (rnd - 0.5);

    // All velocities v0
    (*vx) = (*vy) = v0; 
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
	    if (r < 30 ){
		pair_force(r, &f, &u);
		(*fx) += f * dx / r;
		(*fy) += f * dy / r;
	    }
	}
    }
}

void update(float x[STEPS][NMAX], float y[STEPS][NMAX],
	    float vx[NMAX], float vy[NMAX],
	    int n_particles, int len, float dt){
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

void calc_energy(float x[STEPS][NMAX], float y[STEPS][NMAX],
		 float vx[NMAX], float vy[NMAX], int n_particles,
		 int len, float *e, float *pot_e, float *k_e, float *temp){
/* 
  calculate current energy = potential + kinetic
  also compute temperature via equipartition
*/
    int i, j;
    float r, f, u, dx, dy;

    (*pot_e) = 0.0;    // potential energy
    (*k_e) = 0.0;      // kinetic energy

    for (i = 1; i <= n_particles; i++){
	for (j = i+1; j <= n_particles; j++){

	    find_r(i,j,1,x,y,len,&r,&dx,&dy);  // find spacing of two atoms
	    if (r < 30){                     // using nearest separation rule
		pair_force(r,&f,&u);
		(*pot_e) = (*pot_e) + u;
	    }
	}
	(*k_e) = (*k_e) + ( pow(vx[i],2) + pow(vy[i],2) ) / 2.;  // kinetic energy
    }
    (*e) = (*k_e) + (*pot_e);
    (*temp) = (*k_e) / n_particles;          // equipartition in two dimensions
}

float sgn(float x) {                    
/* signed function */

    if( x < 0 )
	return -1.0;
    else if( x > 0 )
	return 1.0; 
}

void to_histogram (float v[Lmax], float x, float a, float b, int n)
{
    int j = 0;
    float h = (b - a)/n;
    int eureka = 0;

    while (!eureka)
    {
	if ((x<a)||(x>b))
	    eureka=1;
	
	if ( (x>j*h+a) && (x<=(j+1)*h+a))
	{
	    v[j]++;
	    eureka=1;
	}
	else
	    j++;
    }
}
