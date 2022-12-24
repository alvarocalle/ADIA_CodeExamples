set term postscript eps enhanced color 'Helvetica' 10
set encoding iso_8859_1
    
set output 'figura.eps' 

set multiplot       

set size 0.4,0.35
set nokey
#---------------------------------------------
# Gamma = 0.1:

  set origin 0.0,0.665

  set title  "Evolucion de la altura ({/Symbol G} = 0.1)" font "Helvetica,12"
    
  set xlabel "paso temporal ( = t)" 0.5,0.5 font "Helvetica,12"
  set ylabel "h(t)" 0.5,0.5 font "Helvetica,12" 

  set xtics ( 0, 100, 200, 300, 400, 500)
  set ytics ( 200, 0, -200, -400, -600)

  plot [0:500][-600:200]'fort.G.0.1' u 1:2 w l lw 2 

#---------------------------------------------
  set origin 0.5,0.665

  set title  "Evolucion de la velocidad ({/Symbol G} = 0.1)" font "Helvetica,12"
    
  set xlabel "paso temporal ( = t)" 0.5,0.5 font "Helvetica,12"
  set ylabel "v(t)" 0.5,0.5 font "Helvetica,12" 

  set xtics ( 0, 100, 200, 300, 400, 500)
  set ytics (0.4, 0, -0.4, -0.8, -1.2, -1.6, -2)

  plot [0:500][-2:0.4] 'fort_v.0.1' u 1:3 w l lw 1, 0

set nolabel


#-------------------------------------------
 
# Gamma = 1:

  set origin 0.0,0.33

  set title  "Evolucion de la altura ({/Symbol G} = 1)" font "Helvetica,12"
    
  set xlabel "paso temporal ( = t)" 0.5,0.5 font "Helvetica,12"
  set ylabel "h(t)" 0.5,0.5 font "Helvetica,12" 

  set xtics ( 0, 100, 200, 300, 400, 500)
  set ytics ( 200, 0, -200, -400, -600)

  plot [0:500][-600:200]'fort_v.1' u 1:2 w l lw 2 

#-------------------------------------------
  set origin 0.5,0.33

  set title  "Evolucion de la velocidad ({/Symbol G} = 1)" font "Helvetica,12"
    
  set xlabel "paso temporal ( = t)" 0.5,0.5 font "Helvetica,12"
  set ylabel "v(t)" 0.5,0.5 font "Helvetica,12" 

  set xtics ( 0, 100, 200, 300, 400)
  set ytics (-4, -3, -2, -1, 0, 1, 2, 3 )

  plot [0:500][-4:3]'fort_v.1' u 1:3 w l lw 1, 0

set nolabel
#-------------------------------------------

# Gamma = 10:

  set origin 0.0,0.0

  set title  "Evolucion de la altura ({/Symbol G} = 10)" font "Helvetica,12"
    
  set xlabel "paso temporal (= t)" 0.5,0.5 font "Helvetica,12"
  set ylabel "h(t)" 0.5,0.5 font "Helvetica,12" 

  set xtics ( 0, 100, 200, 300, 400, 500)
  set ytics ( 200, 0, -200, -400, -600)

  plot [0:500][-600:200]'fort.G.10' u 1:2 w l lw 2 

#-----------------------

  set origin 0.5,0.0

  set title  "Evolucion de la velocidad ({/Symbol G} = 10)" font "Helvetica,12"
    
  set xlabel "paso temporal ( = t)" 0.5,0.5 font "Helvetica,12"
  set ylabel "v(t)" 0.5,0.5 font "Helvetica,12" 

  set xtics ( 0, 100, 200, 300, 400, 500)
  set ytics (-8, -4, 0, 4, 8 )

  plot [0:500][-8:8]'fort_v.10' u 1:3 t "v(t)" w l lw 1, 0

set nomultiplot


