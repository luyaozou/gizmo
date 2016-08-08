set datafile separator ','
cat = ARG1
temp = real(ARG2)
boost = real(ARG3)
equa(boost,t) = sprintf("($1*1e-3):(10**($3+%f)*(300/%f)**(3./2)/(exp(-$5/0.695/300)-exp(-($5+$1*3.33564*1e-5)/0.695/300)))*exp(-($5+$1*3.33564*1e-5)/0.695/%f)",boost,t,t)
catp = equa(boost,temp)
set xlabel 'Frequency (GHz)'
set ylabel 'Intensity (a.u.)'
plot cat u @catp w impulses ls 2 lw 3 title sprintf('Simulation Temperature %.1f K', temp)
