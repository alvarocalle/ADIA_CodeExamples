/**********************************************************/
/***************  Schrodinger equation  *******************/
/**********************************************************/
/*         Propagacion del paquete de ondas en 1D         */
/*--------------------------------------------------------*/
/* Nota: este programa esta escrito siguiendo bastante    */
/*       fielmente el libro de N. Giordano.               */
/*--------------------------------------------------------*/
/*                                                        */
/* -> uso arrays de dimension 2 para los nºs complejos <- */ 
/*                                                        */
/**********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NMAX 3000

void Init_psi_packet(float p[NMAX][2], int m, float dx,
		     float x0, float sigma, float k);
void Normalize(float p[NMAX][2], int m, float dx);
void Calculate(float p_old[NMAX][2], float p_new[NMAX][2],
	       float e[NMAX][2], float f[NMAX][2], int max,
	       float dx, float dt, float lambda, float dx2);
void Calc_psi(float p_old[NMAX][2],float p_new[NMAX][2],
	      float e[NMAX][2], float f[NMAX][2], int max);
void Calc_ef(float p[NMAX][2], float e[NMAX][2], float f[NMAX][2],
	     float max, float dx, float dt, float lambda, float dx2);
void Plot(float p[NMAX][2], int m, float dx, FILE *fich);
float Check_normalization(float p[NMAX][2], int m);
float Velocity(float p[NMAX][2], int max, float dx, float (*x0), float t);
float Potential(float x);

float psi_old[NMAX][2],
      psi_new[NMAX][2],
      e[NMAX][2],
      f[NMAX][2];

int main ()
{
    int i, num;
    float t, N;

    int max = 2000;       // lattice dimension
    int n_display = 150;  // display psi after every (n_display * dt)
    float dx = 1./max;     
    float dt = 2. * pow(dx, 2);   // lambda = 1

    float x0 = 0.2;    // wave packet is centered here
    float sigma = 0.01;  // wave packet width 
    float k0 = 800.;     // this is the wave vector
    float E = k0*k0/2.0; // energy of the wave packet

    FILE *fich_plot;

    fich_plot = fopen("probabilidad.sal","w");
    fprintf(fich_plot,"# x, psi(x)^2 \n");

    // funcion de onda inicial
    Init_psi_packet(psi_old, max, dx, x0, sigma, k0);
    Plot(psi_old, max, dx, fich_plot);

    N = Check_normalization(psi_old, max);
    printf("\n Normalizacion: %f \n", N);
    printf(" Centro del paquete x0 = %f \n", x0);
    printf(" Velocidad imput v0 = %f \n", k0);
    printf(" Energia del paquete E = %f \n", k0*k0/2.);

    num = 2;
    t = 0.0;
    while (num != 1){
	
	printf("\n-------------------\n");
	printf("\n evolucionando ....\n");
      	printf("\n-------------------\n");

	// evolucionamos el sistema
	for(i = 1; i<= n_display; i++)
	    Calculate(psi_old, psi_new, e, f, max, dx, dt, 2.*dx*dx/dt, dx*dx);
	
	// comprobamos normalizacion
	Plot(psi_old, max, dx, fich_plot);
	N = Check_normalization(psi_old, max);
	printf(" Comprobando la normalizacion: N = %f\n", N);

	t += n_display * dt;

	// calculamos la velocidad y centro
	k0 = Velocity(psi_old, max, dx, &x0, t);
	printf(" Centro del paquete x0 =  %f \n", x0);
	printf(" Comprobando velocidad: v = %f \n", k0);
	printf(" Energia del paquete E = %f \n", k0*k0/2.);

	printf("\n\t***********************************************\n");
	printf("\n\tPulse 1 si desea salir / 2 si desea continuar: \n");
	printf("\n\t***********************************************\n");
	scanf("%d",&num);

    }

    fclose(fich_plot);
}

/*================================================================*/
/*============= Funciones de inicializacion ======================*/
/*================================================================*/

// paquete de onda gaussiano 
void Init_psi_packet(float p[NMAX][2], int m, float dx,
		     float x0, float sigma, float k){
/*  
    psi(x) = exp[ikx] * exp[-(x - x0)^2/ sigma^2]
           =  exp[-(x - x0)^2/ sigma^2] * ( cos(kx) + i sin(kx) )
*/
    int i;
    float a;

    for(i = 0; i <= m; i++){
	a = exp( - pow( (i*dx - x0), 2) / pow(sigma,2) );   // a gaussian packet
	p[i][1] = a * cos(k*i*dx);            // real part of psi
	p[i][2] = a * sin(k*i*dx);            // imaginary part of psi
    }
    Normalize(p,m,dx);
}

// normalize psi - part of initialization
void Normalize(float p[NMAX][2], int m, float dx){

    int i;
    float sum = 0.;

    for(i = 0; i <= m; i++)
	sum += p[i][1]*p[i][1] + p[i][2]*p[i][2];  

    sum = sqrt(sum*dx);

    for(i = 0; i <= m; i++){
	p[i][1] = p[i][1] / sum;
	p[i][2] = p[i][2] / sum;
    }
}

/*================================================================*/
/*=============== Funciones de calculo ===========================*/
/*================================================================*/

// first calculate the new e and f factors, the update psi
void Calculate(float p_old[NMAX][2], float p_new[NMAX][2],
	       float e[NMAX][2], float f[NMAX][2], int max,
	       float dx, float dt, float lambda, float dx2){
    int i;

    //declare def potential
    Calc_ef(p_old, e, f, max, dx, dt, lambda, dx2);
    Calc_psi(p_old, p_new, e, f, max);

    for(i = 0; i <= max; i++){
	p_old[i][1] = p_new[i][1];
	p_old[i][2] = p_new[i][2];
    }
}

// calculate the new wave function
void Calc_psi(float p_old[NMAX][2],float p_new[NMAX][2],
	      float e[NMAX][2], float f[NMAX][2], int max){
/*
  -------------------------------------------
  p_new(.,1) = parte real de p_new(m,n)
  p_new(.,2) = parte imaginaria de p_new(m,n)
  -------------------------------------------
*/
    float emod;
    int i;

    emod = pow(e[max-1][1],2) + pow(e[max-1][2],2);
    p_new[max-1][1] = -(f[max-1][1]*e[max-1][1] + f[max-1][2]*e[max-1][2]) / emod;
    p_new[max-1][2] = -(f[max-1][2]*e[max-1][1] - f[max-1][1]*e[max-1][2]) / emod;

    for(i =max-1; i>= 1; i--){
	emod = pow(e[i][1],2) + pow(e[i][2],2);
	p_new[i][1] = ( ( p_new[i+1][1] - f[i][1] ) * e[i][1]
		    +   ( p_new[i+1][2] - f[i][2] ) * e[i][2] ) / emod;
	p_new[i][2] = ( ( p_new[i+1][2] - f[i][2] ) * e[i][1] 
		    -   ( p_new[i+1][1] - f[i][1] ) * e[i][2] ) / emod;
    }
}

// calculate the new e and f factors
void Calc_ef(float p[NMAX][2], float e[NMAX][2], float f[NMAX][2],
	     float max, float dx, float dt, float lambda, float dx2){
/*
  ------------------------------------
  e(.,1) = parte real de e(m,n)
  e(.,2) = parte imaginaria de e(m,n)
  f(.,1) = parte real de f(m,n)
  f(.,2) = parte imaginaria de f(m,n)
  ------------------------------------
*/
    int i;
    float emod;

    e[1][1] = 2.0 + 2.0 * dx2 * Potential(dx);  
    e[1][2] = - 2.0 * lambda;
    f[1][1] = - p[2][1] + (2.0 * dx2 * Potential(dx) + 2.0) * p[1][1]
	- 2.0 * lambda * p[1][2] - p[0][1];
    f[1][2] = - p[2][2] + (2.0 * dx2 * Potential(dx) + 2.0) * p[1][2]
	+ 2.0 * lambda * p[1][1] - p[0][2];
    
    for(i = 2; i <= max-1; i++){
	emod = pow(e[i-1][1],2) + pow(e[i-1][2],2);
	e[i][1] = 2.0 + 2.0 * dx2 * Potential(i*dx) - e[i-1][1] / emod;
	e[i][2] = - 2.0 * lambda + e[i-1][2] / emod;
	f[i][1] = - p[i+1][1] + (2.0 * dx2 * Potential(i*dx) + 2.0) * p[i][1]
	    - 2.0 * lambda * p[i][2] - p[i-1][1]
	    + ( f[i-1][1] * e[i-1][1] + f[i-1][2] * e[i-1][2] ) / emod;
	f[i][2] = - p[i+1][2] + (2.0 * dx2 * Potential(i*dx) + 2.0) * p[i][2]
	    + 2.0 * lambda * p[i][1] - p[i-1][2]
	    + ( f[i-1][2] * e[i-1][1] - f[i-1][1] * e[i-1][2] ) / emod;
    }
}

// calculate the wave function velocity
float Velocity(float p[NMAX][2], int max, float dx, float (*x0), float t){

    int i;
    float P2, P2max;
    float v, xmed;

    xmed = 0.0;
    P2max = 0.0;

    for(i = 0; i <= max; i++){
	P2 = pow(p[i][1],2) + pow(p[i][2],2);	
	if(P2 >= P2max){
	    P2max = P2;
	    xmed = (float)i * dx;
	}
    }

    v = ( xmed - (*x0) ) / t;    
    (*x0) = xmed;

    return v;
}

// Print psi*psi
void Plot(float p[NMAX][2], int m, float dx, FILE *fich){

    int i;
    float P2;

    for(i = 0; i <= m; i++){
	P2 = pow(p[i][1],2) + pow(p[i][2],2);
	fprintf(fich,"%f %f\n", (i * dx), P2 );
    }    
}

float Potential(float x){

    float V, L, h, xp;
    int opcion;

    xp = 0.5;

    // Eleccion del tipo de potencial:
    opcion = 4;

    switch(opcion){
	
	case 1: // Potencial nulo
	    V = 0.0;
	    break;
	case 2: // Potencial lineal
	    h = 1.0e4;
	    if(x < xp) V = 0.0;
	    else V = h * x;
	    break;	   
	case 3: // Barrera de potencial
	    h = 1.0e6;
	    if(x > xp) V = h;
	    else V = 0.0;
	    break;
	case 4: // Pozo de potencial
	    h = - 1.0e6;
	    if(x > xp) V = h;
	    else V = 0.0;
	    break;
    }
    return V;
}

// compute the normalization of psi*psi
// to be sure that the algorithm is ok
float Check_normalization(float p[NMAX][2], int m){

    float sum = 0.;
    int i;
    
    for(i = 0; i <= m; i++)
	sum = sum + p[i][1]*p[i][1] + p[i][2]*p[i][2];
    
    return sum/m;
}
