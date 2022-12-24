set terminal postscript enhanced color 'Times-Roman' 20
set encoding iso_8859_1
    
set output 'ensanche.eps' 

set title  "Ensanchamiento del paquete" font "Times-Roman,30"

set size 1.0,1.0
set origin 0.0,0.0

set label "V(x) = 0, para todo x " at 0.25, 60 font "Times-Roman,15" 
set label "k_0 = 0" at 0.25, 55 font "Times-Roman,15" 
set label "x_0 = 0.4" at 0.25, 50 font "Times-Roman,15" 

set xlabel "x" font "Times-Roman,20" 
set ylabel "{/Symbol F}{/Symbol F}^{*}" font "Times-Roman,20" 
 
set nokey

plot [0.2:0.6][] 'probabilidad.sal' w l
