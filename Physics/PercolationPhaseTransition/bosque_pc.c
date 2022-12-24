/**************************************************************************/
/*           PERCOLACION: CRITICALIDAD EN EL EQUILIBRIO                   */
/*   $$$$$$$$$$$$$$ CALCULO DE LA DENSIDED CRITICA $$$$$$$$$$$$$$$$$$$$   */
/*------------------------------------------------------------------------*/
/*                                                                        */
/*  Descripcion:                                                          */
/*                                                                        */
/*  - Red cuadrada LxL. Cada nudo de la red representa posible arbol.     */
/*  - Cada nudo de la red esta ocupado por un arbol con probabilidad p    */
/*    y desocupado con probabilidad 1-p.                                  */
/*  - Hay interaccion a vecinos proximos.                                 */
/*  - Se lanza una chispa y se inicia el fuego forestal si el nudo esta   */
/*    ocupado. El fuego se propaga a traves de los VP.                    */
/*  - B (beneficio) = arboles que quedan sin quemar despues de saltar la  */
/*    chispa promediado sobre todas las posibles configuraciones.         */
/*  - Leyenda:                                                            */
/*        1 = arbol ardiendo                                              */
/*        0 = arbol apagado (definido)                                    */
/*        2 = arbol no definido                                           */
/*                                                                        */
/**************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rand.h"

#define SEED (-843)
#define LMAX 10000
#define TIME 100000
#define NPOIN 50
#define NPERCO 1

int bosque[LMAX][LMAX];
int nax[LMAX], nay[LMAX];  // Tabla de vecinos adyacentes

long ldum = SEED;
int L = 300;

int Cuenta(int bosque[LMAX][LMAX], int L);
void Print_dllinux(int MPOS[LMAX][LMAX], int L, FILE *fich);
void Percolacion(int bosque[LMAX][LMAX], int L, float p);
void Mediavar (float data[NPERCO], int N, float *med, float *var);
void Sizeclaster(int lattice[LMAX][LMAX], int N, float *sizex, float *sizey);
void Adyacentes(int nx, int ny, int bosque[LMAX][LMAX],
		int nax[LMAX], int nay[LMAX], int *k, int L);

int main()
{
    float p, pini, pfin, dp;
    float sx, sy, sizemed, sizevar;
    int i;

    FILE *file_out;
    FILE *file_size;

    float size[NPERCO];

    file_size = fopen("p-size.dat","w");    
    fprintf(file_size,"# Curva probabilidad - size medio del claster\n"); 
    fprintf(file_size,"# p(ON), size(max), varianzas:\n");	    

//  Repite para distintas probabilidades
    pini = 0.50;
    pfin = 0.60;
    dp = (pfin - pini) / (float) NPOIN;

    for(p=pini; p<=pfin; p+=dp){
	
	printf("#erase p = %f\n", p);	    

//	for(i=0;i<NPERCO;i++){

	    Percolacion(bosque, L, p);
/*	    Sizeclaster(bosque, L, &sx, &sy);
	    
            if(sx >= sy) size[i] = sx;
            else if(sx < sy) size[i] = sy;
	    }
    
//  Calculo los tamaños medios:
	Mediavar(size, NPERCO, &sizemed, &sizevar);
	fprintf(file_size,"%f %f %f\n", p, sizemed, sizevar);
*/
    }

    fclose(file_size);    
    
    
//    file_out = fopen("conig.dl","w");    
//    quemados = Cuenta(bosque, L);
//    printf("%f %d\n", p, quemados/(L*L));
//    Print_dllinux(bosque, L, file_out);
//    fclose(file_out);    

    return 0;
}

/**********************************************************/
/* ----------------  FUNCIONES  ------------------------- */
/**********************************************************/

void Percolacion(int bosque[LMAX][LMAX], int L, float p)
{
/* --------------------------------------------------------
    Funcion basica que realiza el incendio del bosque de 
    tamaño L con pobabilidad  p de arder.
    -------------------------------------------------------
     -> Constantes GLOBALES usadas dentro de la funcion <-
                ------ LMAX, TIME ---------
    Remark:
    - Esta funcion retorna a la ppal cuando t = TIME
    - Si la probabilidad es critica entonces el sistema
      crece hasta t = TIME
 --------------------------------------------------------- */    

    int i, j, nc, nx, ny;
    int k, nv;
    long t;
    float r;

//  Inicializo todos los nodos de la red como no definidos (2)

    for(i=0;i<L;i++)
    for(j=0;j<L;j++)
	bosque[i][j] = 2;
    
    for(i=0;i<LMAX;i++){ nax[i] = 0; nay[i] = 0; }
        
//  Escojo inicialmente el nudo central y lo quemo (1)

    nc = (int) L/2;
    nx = nc;
    ny = nc;

    bosque[nx][ny]=1; 
    k = 0;

//  Cuento los adyacentes
    Adyacentes(nx, ny, bosque, nax, nay, &k, L);

//  Recorro el bosque y paro si no quedan vecinos sin definir

    for(t=0;t<TIME;t++){

	if(k>0){ 
	
//  Tomo un vecino de la tabla
	    nv = (int) ( k* ran1(&ldum) ) + 1; 
	    nx = nax[nv];
            ny = nay[nv];

//  Debo arreglar la tabla para eliminarlo de ella
	    nax[nv] = nax[k];
	    nay[nv] = nay[k];
	    nax[k] = 0;
	    nay[k] = 0;
	    k--;
/* ------------------------------------------------------
    Sorteo el fuego con probabilidad p:
    El arbol arde (-1) o no arde pero queda definido (1).
    Aplicamos ademas condiciones periodicas.
 ------------------------------------------------------- */ 
	    r = ran1(&ldum);

	    if(r > p){
		bosque[nx][ny] = 0;
//		printf("%d %d %d\n",nx ,ny, 0);
	    }
            else{
		bosque[nx][ny] = 1;
		printf("%d %d %d\n",nx ,ny, 1);
		Adyacentes(nx, ny, bosque, nax, nay, &k, L);
	    }

	} // Bucle k>0
    } // Bucle temporal
} 

void Adyacentes(int nx, int ny, int bosque[LMAX][LMAX], int nax[LMAX], 
		int nay[LMAX], int *k, int L) {
/*----------------------------------------------------
  Miro los 4 VP del punto y si no estan definidos(0)
  los pongo como adyacentes definidos(1) en cuyo caso
  podran o no arder(-1) con cierta probabilidad.
  ----------------------------------------------------*/

    int i, vpx, vpy;

    for(i=-1;i<=1;i+=2){		    
	vpx = nx + i;
	vpy = ny;

//  Condiciones periodicas: 
	vpx = (vpx + L) % L;
	vpy = (vpy + L) % L;

	if( bosque[vpx][vpy] == 2 ){ // Lo acumulo como vecino adyacente
            (*k)++;	    
            nax[*k] = vpx;
            nay[*k] = vpy;
	}
    }

    for(i=-1;i<=1;i+=2){
	vpx = nx;
	vpy = ny + i;	

//  Condiciones periodicas: 
	vpx = (vpx + L) % L;
	vpy = (vpy + L) % L;

	if( bosque[vpx][vpy] == 2 ){ // Lo acumulo como vecino adyacente
            (*k)++;
            nax[*k] = vpx;
            nay[*k] = vpy;
	}
    }  
}     
 
int Cuenta(int bosque[LMAX][LMAX], int L)
{
// Cuenta el nº de arboles quemados en el bosque

    int i, j, n = 0;

    for(i=0;i<L;i++)
    for(j=0;j<L;j++)
	if(bosque[i][j] == 1) n++;   

    return n;
}

void Print_dllinux(int MPOS[LMAX][LMAX], int L, FILE *fich)
{
    int i, j;

    fich = fopen("config.dl","w");    
    fprintf(fich,"# Estado del bosque\n");	    

    for(i=0;i<L;i++)
    for(j=0;j<L;j++)
	if(MPOS[i][j]==1) 
	    fprintf(fich,"%d %d %d\n",i, j, 1);
    fclose(fich);
}

void Sizeclaster(int lattice[LMAX][LMAX], int N, float *sizex, float *sizey)
{
/*  Calcula el tamaño del claster en direcciones x e y */

    int nc, imax, jmax;
    int i, j;

    nc = (int) N/2;
    i = j = imax = jmax = nc;
     
//  Tamaño en x    

    while(i < N){
	if(lattice[i][nc]==1) imax = i;
	i+=1;
    }

//  Tamaño en y    

    while(j < N){
	if(lattice[nc][j]==1) jmax = j;
	j+=1;
    }
         
    *sizex = (float) abs(imax-nc);
    *sizey = (float) abs(jmax-nc);    

}

void Mediavar (float data[NPERCO], int N, float *med, float *var)
{
/* --------------------------------------------
   Esta funcion calcula la madia y varianza 
   - data: vector de reales de longitud N
   - med: almacena la media calculada
   - var: almacena la varianza calculada
   --------------------------------------------
   Remark: Para calcular la varianza hacemos
   
   Var = (1/N-1) [S(x) - N*Med(x)] 
   -------------------------------------------- */

    float s, scua;
    int i;

    s = scua = 0.0;

    for(i=1;i<N;i++){
	s += data[i];
	scua += data[i]*data[i];
    }

    *med = s / N;
    *var = (1.0 / (N-1)) * ( scua - (N*(*med)*(*med)) );
}
