# PySSaGE

PySSaGE is a collection of python versions of analyses originally part of the *PASSaGE* (Rosenberg 2001) and [*PASSaGE* 
2]((https://www.passagesoftware.net)) (Rosenberg and Anderson 2011) software packages for spatial analysis.

See the [PySSaGE wiki]((https://github.com/msrosenberg/pyssage/wiki)) for more information.

---

## Analyses

So far the following analyses have been implemented:

### Quadrat-Variance

#### One-dimensional

* TTLQV: Two-Term Local Quadrat Variance Analysis
* 3TLQV: Three-Term Local Quadrat Variance Analysis
* PQV: Paired Quadrat Variance Analysis
* tQV: Triplet Quadrat Variance Analysis
* 2NLV: Two-term New Local Variance Analysis
* 3NLV: Three-term New Local Variance Analysis

#### Two-dimensional

* 4TLQV: Four-Term Local Quadrat Variance Analysis
* 9TLQV: Nine-Term Local Quadrat Variance Analysis
* 5QV: Pentuplet Quadrat Variance

### Distance Calculations
* Euclidean distances/angles from 1, 2, and 3-dimensional points
* Spherical distances/angles from latitudes and longitudes
* Shortest-path/Geodesic distances  
* Distances from arrays of data
  * Euclidean distances
  * Squared Euclidean distances
  * Manhattan distances
  * Canberra distances
  * Hamming distances
  * Jaccard distances
  * Cosine distances
  * Czekanowski distances
  * Correlation distances
  * Squared correlation distances
* Distance class determination

### Connections/Links
* Delaunay/Voronoi Tessellation
* Minimum-spanning Tree
* Relative Neighborhood Network
* Gabriel Graph 
* Least-diagonal Network
* Range-based Connections
* *k*-nearest Neighbors

### Correlograms
* Correlograms
  * Moran's *I* 
  * Geary's *c* 
* Mantel Correlograms 

### Anisotropy
* Bearing Analysis 
* Bearing Correlograms
* Windrose Correlograms
* Windrose Mantel Correlograms

### Miscellaneous
* Mantel tests 
