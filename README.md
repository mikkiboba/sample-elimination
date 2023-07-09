# Sample-Elimination

This is the repository for the Computer Graphics project.<br>
The project choosen is **Sample Elimination** and is made in python.<br>
Libraries used: `numpy`, `opencv`, `math`, `heapq`, `scipy` and `matplotlib`<br>

Document used for this project: http://www.cemyuksel.com/research/sampleelimination/sampleelimination.pdf

---
> file: `random_sampling.py`
This algorithm places samples in a 2D surface randomly.

```Python
n_points = 5000
x = np.random.rand(n_points)
y = np.random.rand(n_points)
```
_this generates 5000 random samples_<br>

<div align="center">
<h3>Result</h3>
<img src="https://i.postimg.cc/MHgVpVcT/random-sampling.png">
</div>


---
> file: `2D_weighted_elimination.py`
This algorithm places samples in a 2D surface using the weighted elimination.

* The program generates samples randomly, like the `random sampling`.

```Python
points = np.random.rand(input_size,2)
```

* The samples are inserted in a 2-d tree.

```Python
tree = spatial.cKDTree(points)
```

* Weights are assigned to the samples.

```Python
weights = np.zeros(len(points))
for i in range(len(points)):
  indexes = tree.query_ball_point(points[i], 2*r_max)
  for idx in indexes:
    if i != idx:
      distance = calculateDistance(points[i],points[idx],r_max,r_min)
      weights[i] += (1-(distance/(2*r_max)))**alpha
```

* The weights are put in a MAX-heap.

* The samples with the highest weight get eliminated until there are around 1/3 - 1/5 of the number of the starting samples.

```Python
 while len(heap) > output_size:
    pop = heapq.heappop(heap)
    indexes = tree.query_ball_point(points[pop[1]], 2*r_max)
    for idx in indexes:
      if idx != pop[1]:
        distance = calculateDistance(points[pop[1]],points[idx],r_max,r_min)
        app_weight = weights[idx]
        weights[idx] -= (1-(distance/(2*r_max)))**alpha
        if not math.isnan(weights[idx]):
          heap.remove((-1*app_weight,idx))
          heap.append((-1*weights[idx],idx))
    points[pop[1],0] = np.nan
    points[pop[1],1] = np.nan
    weights[pop[1]] = np.nan
    heapq.heapify(heap)
```

<div align="center">
<h3>Result</h3>
<img src="https://i.postimg.cc/Hs7xbjvQ/2-D-weighted-elimination.png">
</div>

---
> file: `3D_weighted_elimination.py`
This algorithm places samples in a 3D surface using the weighted elimination formula.

The code is similar to the one for the `2D weighted elimination`, changing only the number of dimensions of points (from 2 to 3) and the formula to calculate the distance between two points (adapting it to the third dimension).

<div align="center">
<h3>Result</h3>
<img height=480 width=640 src="https://i.postimg.cc/XJJ2zRT0/3-D-weighted-elimination.png">
</div>

---
> file: `adaptive_elimination.py`
This algorithm places samples in a 2D surface using an image as weight map.

The difference between this and `2D weighted elimination` is how the weight is calculated.<br>
It takes an image, converts it to gray scale and gets the weight map from it.<br>

```Python
def transformDistanceDensity(distance:float,point:np.ndarray,gray_img:np.ndarray,d_max:float) -> float:
  if math.isnan(point[0]):
    return d_max
  y = int(point[0] * gray_img.shape[0])
  x = int(point[1] * gray_img.shape[1])
  density = gray_img[y,x]
  distance = distance * (3 - 2*(density/255))
  return min(distance,d_max)
```

<div align="center">
<h3>Results</h3>
  <table>
    <tr>
      <th><img width=300 height=300 src="https://i.postimg.cc/vZRgc5hR/google.png"></th>
      <th><img width=300 height=300 src="https://i.postimg.cc/qM2ncmzB/adaptive-google.png"></th>
    </tr>
    <tr>
      <th><img width=300 height=300 src="https://i.postimg.cc/yNsT3Mjz/gerry.png"></th>
      <th><img width=300 height=300 src="https://i.postimg.cc/yxdQgMVN/adaptive-gerry.png"></th>
    </tr>
    <tr>
      <th><img width=300 height=300 src="https://i.postimg.cc/8Czv4V0d/man.jpg"></th>
      <th><img width=300 height=300 src="https://i.postimg.cc/g2C9qYJQ/adaptive-man.png"></th>
    </tr>
    <tr>
      <th><img width=300 height=300 src="https://i.postimg.cc/xTLFDKTx/photo.jpg"></th>
      <th><img width=300 height=300 src="https://i.postimg.cc/3wqZDRx0/adaptive-photo.png"></th>
    </tr>
  </table>

</div>

***

More informations (in italian) can be found in `Relazione Computer Graphics.pdf`.

