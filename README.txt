# Scripts to replicate the data presented in
#  IEEE Transactions on Big Data.

In order to replicate our traces and plots, we supply
the Python classes used for this study.

1) Points (see file make_points.py):

Python class that samples the shapefiles according
to a maximum population and a maximum area each sampled point
represents. The shapefiles are stored in the folder * ./Ireland/ *

2) Adjacency (see file make_adj.py):

Python class to calculate basic network parameters for a cellular network,
i.e. RSSI, SINR, Attenuation. This file makes use of the points generated
by the Points class and the base_station.csv file that contains
essential information about the cellular network deployment for two operators
(i.e. base stations coordinates, power, frequency, etc.)

3) Plotter (see file plotter.py):

Python class to generate all plots in the paper
"Assembling and Using a Cellular Dataset for Mobile Network Analysis and Planning".
This file needs all the adjacencies and points file generate
by Points and Adjacency.

3) run.py

Create all databases, i.e. all points files, all adjacency files,
and creates all plots.

<code>
> chmod +x run.py
> ./run.py
</code>

*Benchmarking Results*

A useful way to check the benchmarking results, would be to visualize with the 
shakeviz python library the .cprof files stored in ./profiling/.