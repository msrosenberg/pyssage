# Quadrat Variance Methnods

The following methods are all based on a similar principle. They calculate the variance of differences among 
blocks of different sizes or scales and use the pattern of the variance estimates to determine the scale of 
pattern. The methods differ primarily in the number and distribution of blocks being compared (the shape of the 
logical spatial template).

See Usher (1975), Ludwig and Goodall (1978), Lepš (1990), and Dale (1999) for comparisons and contrasts among 
quadrat variance and related methods.

## One-dimensional Methods

All one-dimensional quadrat variance methods have the identical function calls, for example: 

```
def ttlqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
          wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
```

The only required input, *transect*, is a one-dimensional numpy.ndarray containing the transect to be analyzed. 

Optional parameters include:

*min_block_size:* the smallest block size of the analysis (default = 1)

*max_block_size:* the largest block size of the analysis (default = 0, indicating 50% of the transect length for two
term analyses, 33% for three term analyses)

*block_step:* the incremental size increase of each block size (default = 1)

*wrap:* treat the transect as a circle where the ends meet (default = False)

*unit_scale:* represents the unit scale of a single block (default = 1). Can be used to rescale the units of 
the output, *e.g.*, if the blocks are measured in centimeters, you could use a scale of 0.01 to have the
output expressed in meters.

All quadrat variance analyses return a two-column numpy matrix, with the first column containing the scaled block 
size and the second the calculated variance.


### TTLQV: Two-Term Local Quadrat Variance Analysis 

```
def ttlqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
          wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
```

Originally developed by Hill (1973), TTLQV essentially calculates the mean square difference between adjacent blocks 
for each scale. The variance at block size (scale) *b* is

<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_2%5Cleft%28b%5Cright%29%3D%5Cfrac%7B%5Cdisplaystyle%5Csum_%7Bi%3D1%7D%5E%7Bn_x%2B1-2b%7D+%5Cleft%28%7B%5Cdisplaystyle%5Csum_%7Bj%3Di%7D%5E%7Bi%2Bb-1%7Dd_j+-+%5Cdisplaystyle%5Csum_%7Bj%3Di%2Bb%7D%5E%7Bi%2B2b-1%7Dd_j%7D%5Cright%29%5E2%7D%7B2b%5Cleft%28n_x%2B1-2b%5Cright%29%7D" alt="V_2\left(b\right)=\frac{\displaystyle\sum_{i=1}^{n_x+1-2b} \left({\displaystyle\sum_{j=i}^{i+b-1}d_j - \displaystyle\sum_{j=i+b}^{i+2b-1}d_j}\right)^2}{2b\left(n_x+1-2b\right)}">

The value in one block of width *b* is contrasted against the value in the following block, also of width *b*. The 
scale at which the variance peaks is interpreted as the scale of the pattern being investigated.

### 3TLQV: Three-Term Local Quadrat Variance Analysis

```
def three_tlqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
               wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
```

Originally developed by Hill (1973), 3TLQV is similar to TTLQV, except that it is based on three blocks, rather than 
two. This method looks at the squared difference between the sum of the first and third blocks and twice the 
second block (the second block is multiplied by two so that the expected value if all blocks were equal is zero). 
The variance of 3TLQV is

<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_3%5Cleft%28b%5Cright%29%3D%5Cfrac%7B%5Cdisplaystyle%5Csum_%7Bi%3D1%7D%5E%7Bn_x%2B1-3b%7D+%5Cleft%28%7B%5Cdisplaystyle%5Csum_%7Bj%3Di%7D%5E%7Bi%2Bb-1%7Dd_j+-+2%5Cdisplaystyle%5Csum_%7Bj%3Di%2Bb%7D%5E%7Bi%2B2b-1%7Dd_j%7D+%2B+%5Cdisplaystyle%5Csum_%7Bj%3Di%2B2b%7D%5E%7Bi%2B3b-1%7Dd_j+%5Cright%29%5E2%7D%7B8b%5Cleft%28n_x%2B1-3b%5Cright%29%7D" alt="V_3\left(b\right)=\frac{\displaystyle\sum_{i=1}^{n_x+1-3b} \left({\displaystyle\sum_{j=i}^{i+b-1}d_j - 2\displaystyle\sum_{j=i+b}^{i+2b-1}d_j} + \displaystyle\sum_{j=i+2b}^{i+3b-1}d_j \right)^2}{8b\left(n_x+1-3b\right)}">

The scale at which the variance peaks is interpreted as the scale of the pattern being investigated.


### PQV: Paired Quadrat Variance Analysis

```
def pqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
        wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
```

Paired Quadrat Variance differs from the local quadrat variances in that it tests for scale by changing the 
distance between tested quadrats without increasing block size. The formula for PQV is

<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_P%5Cleft%28b%5Cright%29+%3D+%5Cfrac%7B%5Cdisplaystyle%5Csum_%7Bi%3D1%7D%5E%7Bn_x+-+b%7D%5Cleft%28d_i+-+d_%7Bi%2Bb%7D%5Cright%29%5E2%7D%7B2%5Cleft%28n_x+-+b%5Cright%29%7D" alt="V_P\left(b\right) = \frac{\displaystyle\sum_{i=1}^{n_x - b}\left(d_i - d_{i+b}\right)^2}{2\left(n_x - b\right)}">

As with the above methods, peaks in variance are interpreted as the scale of pattern. PQV is essentially identical 
to a semivariogram analysis and is closely related to the calculation of autocorrelation (Dale and Mah 1998; 
ver Hoef *et al.* 1993).


### tQV: Triplet Quadrat Variance Analysis

```
def tqv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
        wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
```

Dale (1999) proposed a variation of paired quadrat variance called Triplet Quadrat Variance (tQV). This method is 
akin to PQV in the way 3TLQV is akin to TTLQV. Rather than looking at pairs separated by a set distance as in PQV, 
tQV examines triplets. The formula for tQV is

<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_t%5Cleft%28b%5Cright%29+%3D+%5Cfrac%7B%5Cdisplaystyle%5Csum_%7Bi%3D1%7D%5E%7Bn_x+-+2b%7D%5Cleft%28d_i+-+2d_%7Bi%2Bb%7D+%2B+d_%7Bi%2B2b%7D%5Cright%29%5E2%7D%7B4%5Cleft%28n_x+-+2b%5Cright%29%7D%0A" alt="V_t\left(b\right) = \frac{\displaystyle\sum_{i=1}^{n_x - 2b}\left(d_i - 2d_{i+b} + d_{i+2b}\right)^2}{4\left(n_x - 2b\right)}">

### 2NLV: Two-term New Local Variance Analysis

```
def two_nlv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
            wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
```

Galiano (1982) proposed a pair of methods (a two-term and a three-term method) for detecting the patch size rather 
than the scale of pattern. These methods calculate the average size of either gaps or patches (whichever is 
smaller); the previously described quadrat variance methods average across both gaps and patches (Dale 1999). 
Galiano’s methods, known as New Local Variance (NLV), are closely related to TTLQV and 3TLQV. The NLVs calculate 
the differences among neighboring TTLQV (or 3TLQV) blocks. The variance for the two-term version of NLV is

<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_%7BN2%7D%5Cleft%28b%5Cright%29+%3D+%5Cfrac%7B%5Cdisplaystyle%5Csum_%7Bi%3D1%7D%5E%7Bn_x+-2b%7D+%5Cleft%7C+%5Cleft%28%5Cdisplaystyle%5Csum_%7Bj%3Di%7D%5E%7Bi%2Bb-1%7Dd_j+-+%5Cdisplaystyle%5Csum_%7Bj%3Di%2Bb%7D%5E%7Bi%2B2b-1%7Dd_j%5Cright%29%5E2+-+%5Cleft%28%5Cdisplaystyle%5Csum_%7Bj%3Di%2B1%7D%5E%7Bi%2Bb%7Dd_j+-+%5Cdisplaystyle%5Csum_%7Bj%3Di%2Bb%2B1%7D%5E%7Bi%2B2b%7Dd_j%5Cright%29%5E2+%5Cright%7C+%7D%7B2b%5Cleft%28n_x+-+2b%5Cright%29%7D" alt="V_{N2}\left(b\right) = \frac{\displaystyle\sum_{i=1}^{n_x -2b} \left| \left(\displaystyle\sum_{j=i}^{i+b-1}d_j - \displaystyle\sum_{j=i+b}^{i+2b-1}d_j\right)^2 - \left(\displaystyle\sum_{j=i+1}^{i+2b}d_j - \displaystyle\sum_{j=i+b+1}^{i+2b}d_j\right)^2 \right| }{2b\left(n_x - 2b\right)}">

### 3NLV: Three-term New Local Variance Analysis

```
def three_nlv(transect: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
              wrap: bool = False, unit_scale: Number = 1) -> numpy.ndarray:
```

Galiano's (1982) three-term version of New Local Variance.

<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_%7BN3%7D%5Cleft%28b%5Cright%29+%3D+%5Cfrac%7B%5Cdisplaystyle%5Csum_%7Bi%3D1%7D%5E%7Bn_x+-3b%7D+%5Cleft%7C+%5Cleft%28%5Cdisplaystyle%5Csum_%7Bj%3Di%7D%5E%7Bi%2Bb-1%7Dd_j+-+2%5Cdisplaystyle%5Csum_%7Bj%3Di%2Bb%7D%5E%7Bi%2B2b-1%7Dd_j+%2B+%5Cdisplaystyle%5Csum_%7Bj%3Di%2B2b%7D%5E%7Bi%2B3b-1%7Dd_j%5Cright%29%5E2+-+%5Cleft%28%5Cdisplaystyle%5Csum_%7Bj%3Di%2B1%7D%5E%7Bi%2Bb%7Dd_j+-+2%5Cdisplaystyle%5Csum_%7Bj%3Di%2Bb%2B1%7D%5E%7Bi%2B2b%7Dd_j+%2B+%5Cdisplaystyle%5Csum_%7Bj%3Di%2B2b%2B1%7D%5E%7Bi%2B3b%7Dd_j%5Cright%29%5E2+%5Cright%7C+%7D%7B8b%5Cleft%28n_x+-+3b%5Cright%29%7D" alt="V_{N3}\left(b\right) = \frac{\displaystyle\sum_{i=1}^{n_x -3b} \left| \left(\displaystyle\sum_{j=i}^{i+b-1}d_j - 2\displaystyle\sum_{j=i+b}^{i+2b-1}d_j + \displaystyle\sum_{j=i+2b}^{i+3b-1}d_j\right)^2 - \left(\displaystyle\sum_{j=i+1}^{i+b}d_j - 2\displaystyle\sum_{j=i+b+1}^{i+2b}d_j + \displaystyle\sum_{j=i+2b+1}^{i+3b}d_j\right)^2 \right| }{8b\left(n_x - 3b\right)}">

---

## Two-dimensional Methods
```

```

### 4TLQV: Four-Term Local Quadrat Variance Analysis (Dale 1990, 1999)

```
def four_tlqv(surface: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
              unit_scale: Number = 1) -> numpy.ndarray:
```

### 9TLQV: Nine-Term Local Quadrat Variance Analysis (Dale 1990, 1999)

```
def nine_tlqv(surface: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
              unit_scale: Number = 1) -> numpy.ndarray:
```

### 5QV: Pentuplet Quadrat Variance (Fortin and Dale 2005)

```
def five_qv(surface: numpy.ndarray, min_block_size: int = 1, max_block_size: int = 0, block_step: int = 1,
            unit_scale: Number = 1) -> numpy.ndarray:
```


---

## References

* Dale, M.R T. (1999) *Spatial Pattern Analysis in Plant Ecology.* Cambridge: Cambridge University Press.

* Dale, M.R.T., and M. Mah (1998) The use of wavelets for spatial pattern analysis in ecology. *Journal of 
  Vegetation Science* 9:805-814.

* * Galiano, E.F. (1982) Détection et mesure de l'hétérogénéité spatiale des espèces dans les pâturages. *Acta 
  OEcologia / OEcologia Plantarum* 3:269-278.

* Hill, M.O. (1973) The intensity of spatial pattern in plant communities. *Journal of Ecology* 61:225-235.

* Lepš, J. (1990) Comparison of transect methods for the analysis of spatial pattern. Pp. 71-82 in 
  *Spatial Processes in Plant Communities*, F. Krahulec, A.D.Q. Agnew, S. Agnew and J.H. Willem, eds. Prague: 
  Academia Press.

* Ludwig, J.A., and D.W. Goodall (1978) A comparison of paired- with blocked-quadrat variance methods for the 
  analysis of spatial pattern. *Vegetatio* 38:49-59.

* Usher, M.B. (1975) Analysis of pattern in real and artificial plant populations. *Journal of Ecology* 63:569-586.

* ver Hoef, J.M., N.A.C. Cressie, and D.C. Glenn-Lewin (1993) Spatial models for spatial statistics: Some 
  unification. *Journal of Vegetation Science* 4:441-452.
