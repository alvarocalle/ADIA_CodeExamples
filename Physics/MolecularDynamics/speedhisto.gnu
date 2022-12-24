set terminal postscript enhanced color 'Times-Roman' 20
set encoding iso_8859_1
    
set output 'speedhisto.eps' 

set title  "Speed distribution" font "Times-Roman,30"

set size 1.0,1.0
set origin 0.0,0.0

set label "v(initial) = 1.0" at 3,0.55 font "Times-Roman,15" 
set label "Lattice 10x10" at 3,0.5 font "Times-Roman,15" 
set label "N(particles) = 20" at 3,0.45 font "Times-Roman,15" 

set xlabel "v" font "Times-Roman,20" 
set ylabel "P(v)" font "Times-Roman,20" 
 
plot \
'histogram_v.sal' u 1:4 t "Histogram" w boxes,\
'histogram_v.sal' u 1:5 t "Maxwell distribution 2D" w l
