set terminal postscript enhanced color 'Times-Roman' 20
set encoding iso_8859_1
    
set output 'propagation.eps' 

set title  "Propagacion del paquete" font "Times-Roman,30"

set size 1.0,1.0
set origin 0.0,0.0

set label "V(x) = 0, para todo x " at 0.7, 70 font "Times-Roman,15" 
set label "k_0 = 500" at 0.7, 65 font "Times-Roman,15" 
set label "x_0 = 0.4" at 0.7, 60 font "Times-Roman,15" 

set xlabel "x" font "Times-Roman,20" 
set ylabel "{/Symbol F}{/Symbol F}^{*}" font "Times-Roman,20" 
 
set nokey

plot [0.3:0.9] 'probabilidad.sal' w l
