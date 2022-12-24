/**************************************************************************/
/*           PERCOLACION: CRITICALIDAD EN EL EQUILIBRIO                   */
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
#define TIME 10000
#define PROB 0.5

int bosque[LMAX][LMAX];
int nax[LMAX], nay[LMAX];  // Tabla de vecinos adyacentes

long ldum = SEED;
int L = 300;
float p = PROB;

void Print_dllinux(int MPOS[LMAX][LMAX], int L);
void Adyacentes(int nx, int ny, int bosque[LMAX][LMAX], int nax[LMAX], 
		int nay[LMAX], int *k, int L);

int main(void)
{
    int i, j, nc, nx, ny;
    int k, nv, quemados;
    long t;
    float r;

//  Inicializo todos los nodos de la red como no definidos (2)

    for(i=0;i<L;i++)
    for(j=0;j<L;j++)
	bosque[i][j] = 2;
    
    for(i=0;i<LMAX;i++){
	nax[i] = 0; nay[i] = 0; }
        
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
		printf("%d %d %d\n",nx ,ny, 0);
	    }
            else{
		bosque[nx][ny] = 1;
		printf("%d %d %d\n",nx ,ny, 1);
		Adyacentes(nx, ny, bosque, nax, nay, &k, L);
	    }

	} // Bucle k>0
    } // Bucle temporal

    return 0;
}

/**********************************************************/
/* ----------------  FUNCIONES  ------------------------- */
/**********************************************************/

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
 
void Print_dllinux(int MPOS[LMAX][LMAX], int L)
{
    int i, j;
    FILE *fich;

    fich = fopen("config.dl","w");    
    fprintf(fich,"# Estado del bosque\n");	    

    for(i=0;i<L;i++)
    for(j=0;j<L;j++)
	if(MPOS[i][j]==1) 
	    fprintf(fich,"%d %d %d\n",i, j, 1);
    fclose(fich);
}
