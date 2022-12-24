set term postscript eps enhanced 14 
set size 1,1 
set encoding iso_8859_1

set output 'incendios.eps' 

set nokey
set multiplot

# Incendio 1

set size 0.5,0.5
set origin 0.0,0.5      

set title "Red L = 300, p = 0.5" font "Helvetica,16"   
plot [120:180][120:180] 'p0.5.dat' pt 7 ps 0.5

# Incendio 2
      
set size 0.5,0.5
set origin 0.5,0.5      
       
set title "Red L = 300, p = 0.53" font "Helvetica,16"
       
plot [100:220][100:220] 'p0.53.dat' pt 7 ps 0.3

# Incendio 3
     
set size 0.5,0.5
set origin 0.0,0.0      
       
set title "Red L = 300, p = 0.54" font "Helvetica,16"
       
plot 'p0.54.dat' pt 7 ps 0.1

# Incendio 4

set size 0.5,0.5
set origin 0.5,0.0      
       
set title "Red L = 300, p = 0.58" font "Helvetica,16"
       
plot 'p0.58.dat' pt 7 ps 0.1

set nomultiplot

