reset

filename=ARG1

HATREE_2_CM = 219474.63

set xlabel 'Dihedral Angle (deg)'
set ylabel 'Relative Energy (cm^{-1})'
set xrange [-180:180]
set xtics out -180, 60 nomirror
set ytics out nomirror 

stats filename u 1:2 nooutput
print(sprintf('V3 | %.2f cm^{-1}  %.3f GHz', (STATS_max_y-STATS_min_y) * HATREE_2_CM,(STATS_max_y-STATS_min_y) * HATREE_2_CM * 29.9792458))

plot filename u 1:($2-STATS_min_y)*HATREE_2_CM w linespoints lw 2 pt 6 ps 1 t ''

