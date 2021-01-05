# PySSaGE

PySSaGE is a collection of python versions of analyses originally part of the *PASSaGE* (Rosenberg 2001) and [*PASSaGE* 
2]((https://www.passagesoftware.net)) (Rosenberg and Anderson 2011) software packages for spatial analysis.

A more throrough explanation of how to use the code is forthcoming.

## Analyses

So far the following analyses have been implemented:

### Quadrat-Variance

* TTLQV: Two-Term Local Quadrat Variance Analysis (Hill 1973)
* 3TLQV: Three-Term Local Quadrat Variance Analysis (Hill 1973)
* PQV: Paired Quadrat Variance Analysis
* tQV: Triplet Quadrat Variance Analysis (Dale 1999)
* 2NLV: Two-term New Local Variance Analysis (Galiano 1982)
* 3NLV: Three-term New Local Variance Analysis (Galiano 1982)

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
* Delaunay/Voronoi Tessellation (Delaunay 1928, 1934)
* Minimum-spanning Tree
* Relative Neighborhood Network
* Gabriel Graph (Gabriel and Sokal 1969)
* Least-diagonal Network (Fraser and van den Driessche 1972)
* Range-based Connections
* *k*-nearest Neighbors

### Correlograms
* Correlograms (Cliff and Ord 1973, 1981; Sokal and Oden 1978)
  * Moran's *I* (Moran 1950)
  * Geary's *c* (Geary 1954)
* Mantel Correlograms (Sokal *et al.* 1987)

### Anisotropy
* Bearing Analysis (Falsetti and Sokal 1993)
* Bearing Correlograms (Rosenberg 2000)
* Windrose Correlograms (Oden and Sokal 1986)
* Windrose Mantel Correlograms

### Miscellaneous
* Mantel tests (Mantel 1967; Mantel and Valand 1970)


## References

* Cliff, A.D., and J.K. Ord (1973) *Spatial Autocorrelation*. London: Pion.

* Cliff, A.D., and J.K. Ord (1981) *Spatial Processes*. London: Pion.

* Dale, M.R T. (1999) *Spatial Pattern Analysis in Plant Ecology.* Cambridge: Cambridge University Press.

* Delaunay, B. (1928) Sur la sphère vide. Pp. 695-700 in Proceedings of the International Mathematical Congress 
  held in Toronto, August 11-16. Toronto: University of Toronto Press.

* Delaunay, B. (1934) Sur la sphère vide. Bulletin de L'Academie des Sciences de L'URSS Classe des Sciences 
  Mathématiques et Naturelles 7:793-800.

* Falsetti, A.B., and R.R. Sokal (1993) Genetic structure of human populations in the British Isles. *Annals of 
  Human Biology* 20:215-229.

* Fraser, A.R., and P. van den Driessche (1972) Triangles, density and pattern in point populations. Pp. 277-286 in 
  *Proceedings of the 3rd Conference of the Advisory Group of Forest Statisticians*. Jouy-en-Josas, France: 
  International Union for Research organization, Institut National de la Recherche Agronomique.

* Gabriel, K.R., and R.R. Sokal (1969) A new statistical approach to geographic variation analysis. *Systematic 
  Zoology* 18:259-278.

* Galiano, E.F. (1982) Détection et mesure de l'hétérogénéité spatiale des espèces dans les pâturages. *Acta 
  OEcologia / OEcologia Plantarum* 3:269-278.

* Geary, R.C. (1954) The contiguity ratio and statistical mapping. *Incorporated Statistician* 5:115-145.

* Moran, P.A.P. (1950) Notes on continuous stochastic phenomena. *Biometrika* 37:17-23.

* Hill, M.O. (1973) The intensity of spatial pattern in plant communities. *Journal of Ecology* 61:225-235.

* Mantel, N. (1967) The detection of disease clustering and a generalized regression approach. *Cancer Research* 
  27:209-220.

* Mantel, N., and R.S. Valand (1970) A technique of nonparametric multivariate analysis. *Biometrics* 26:547-558.

* Oden, N.L., and R.R. Sokal (1986) Directional autocorrelation: An extension of spatial correlograms to two 
  dimensions. *Systematic Zoology* 35:608-617.

* Rosenberg, M.S. (2000) The bearing correlogram: A new method of analyzing directional spatial autocorrelation. 
  *Geographical Analysis* 32:267-278.

* Rosenberg, M.S. (2001) *PASSAGE. Pattern Analysis, Spatial Statistics and Geographic Exegesis.* Version 1.

* Rosenberg, M.S., and C.D. Anderson (2011) *PASSaGE. Pattern Analysis, Spatial Statistics and Geographic 
Exegesis.* Version 2. *Methods in Ecology and Evolution* 2(3):229–232. 
[DOI: 10.1111/j.2041-210x.2010.00081.x](https://dx.doi.org/10.1111/j.2041-210x.2010.00081.x)

* Sokal, R.R., and N.L. Oden (1978) Spatial autocorrelation in biology 1. Methodology. *Biological Journal of the 
  Linnean Society* 10:199-228.

* Sokal, R.R., N.L. Oden, and J.S.F. Barker (1987) Spatial structure in *Drosophila buzzatii* populations: Simple 
  and directional spatial autocorrelation. *American Naturalist* 129:122-142.
