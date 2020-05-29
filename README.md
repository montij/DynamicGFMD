# DynamicGFMD
Dynamic implementation of Green's functions molecular dynamics

Compatible with LAMMPS (21 Sep 2016) and likely earlier versions

Invocation of dynamic solver:\
```fix ID group-ID gfmd gfmd fcc100 1 SPRINGCONSTANT height 0 solver dynamic NMAX NMIN 0 MASS GAMMA```

**Arguments**:\
SPRINGCONSTANT - Spring constant for nearest-neighbor interactions\
NMAX - Number of layers kept for q = 0\
NMIN - Minimum number of layers kept for all q\
MASS - Mass of subsurface GFMD atoms\
GAMMA - Damping coefficient for subsurface Kelvin Damping\

At present, this code works only for FCC100 faces with single atom unit cells.\
