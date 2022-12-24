set term postscript eps enhanced 14 
set size 1,1 
set encoding iso_8859_1

# ======= GRAFICO TAMAÑO-PROBABILIDAD ==================

set output 'percolacion_pc.eps' 
       
set nokey 
       
       set size 0.5,0.5
       set origin 0,0.5

set xlabel "{/Symbol r}" 0.5,0.0 font "Helvetica,16"
set ylabel "{/Symbol D}x_{mitad}" 0,0.5 font "Helvetica,16" 

set label "Cambio de fase" at 0.56,10 font "Helvetica,18"
plot 'p-size.dat' w l lw 1.5
       
set nolabel
       
# ======= GRAFICO B(rho) ==================


set output 'percolacion_soc.eps' 
       
set nokey 
       
       set size 0.5,0.5
       set origin 0,0.5

set xlabel "{/Symbol r}" 0.5,0.0 font "Helvetica,16"
set ylabel "B({/Symbol r}) = {/Symbol r} - < s >" 0,0.5 font "Helvetica,16" 

set label "SOC" at 0.75,0.4 font "Helvetica,15"
set label "Pecolation" at 0.6,0.44 font "Helvetica,15"
plot [][0:] 'config.dl' u 2:1 w l lw 1.5
       
set nolabel

