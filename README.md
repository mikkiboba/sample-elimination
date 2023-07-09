# Sample-Elimination
This is the repository for the Computer Graphics project. The project choosen is **Sample Elimination** and is made in python.
Libraries used: `numpy`, `opencv`, `math`, `heapq`, `scipy` and `matplotlib`
---
> file: `random_sampling.py`
This algorithm places samples in a 2D surface randomly.

```Python
n_points = 5000
x = np.random.rand(n_points)
y = np.random.rand(n_points)
```
_this generates 5000 random samples_

---
> file: `2D_weighted_elimination.py`
This algorithm places samples in a 2D surface using the weighted elimination formula.

---
> file: `3D_weighted_elimination.py`
This algorithm places samples in a 3D surface using the weighted elimination formula.

---
> file: `adaptive_elimination.py`
This algorithm places samples in a 2D surface using an image as weight map.


