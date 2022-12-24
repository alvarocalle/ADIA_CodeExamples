set term postscript eps enhanced color 'Helvetica' 20
set encoding iso_8859_1
    
set output 'figura2.eps' 

set size 1,1
set origin 0.0,0.0

set title  "Efecto del ruido en la trayectoria" font "Helvetica,25"
    
set xlabel "paso temporal ( = t)" 0,0 font "Helvetica,20"
set ylabel "h(t)" 0,0 font "Helvetica,20" 

plot [][-1000:]'fort.G.0.1' u 1:2 t "{/Symbol G} = 0.1" w l lw 2,\
     'fort.G.1' u 1:2 t "{/Symbol G} = 1" w l lw 2,\
     'fort.G.10' u 1:2 t "{/Symbol G} = 10" w l lw 2,\
     'fort.G.100' u 1:2 t "{/Symbol G} = 100" w l lw 2
