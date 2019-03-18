set datafile separator ','
cat = ARG1      # catalog file name *.csv
temp = real(ARG2)   # simulation Temperature
alpha = real(ARG3)  # 1.5 for non-linear & 1 for linear molecule
boost = real(ARG4)  # magnitude booster, arb.
equa(boost,t,a) = sprintf("($1*1e-3):(10**($3+%f)*(300/%f)**(%f)/(exp(-$5/0.695/300)-exp(-($5+$1*3.33564*1e-5)/0.695/300)))*(exp(-$5/0.695/%f)-exp(-($5+$1*3.33564*1e-5)/0.695/%f))",boost,t,a,t,t)
catp = equa(boost,temp,alpha)
set xlabel 'Frequency (GHz)'
set ylabel 'Intensity (a.u.)'
plot cat u @catp w impulses ls 2 lw 2 title sprintf('Simulation Temperature %.1f K', temp)
