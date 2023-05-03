import numpy as np
import heapq
import math
import scipy.spatial as spatial
import matplotlib.pyplot as plt


def getNeighbours(points, tree:spatial.KDTree, r_max):
    neighbours = []
    for i in range(len(points)):
        indexes = tree.query_ball_point(points[i], 2*r_max)
        neighbours.append(indexes)
    print(neighbours[0])
    return neighbours


def generateHeap(weights: np.ndarray) -> list:
    """Generate a MAX-heap from a list of weights.

    Args:
        weights (np.ndarray): the weights to be used to generate the heap

    Returns:
        list: the MAX-heap
    """
    heap = []
    heapq.heapify(heap)  # O(n)
    for i in range(len(weights)):
        if not math.isnan(weights[i]):
            # push the negative of the weight to make it a MAX-heap
            heapq.heappush(heap, (-1*weights[i], i))
    return heap


def calculateDistance(p1: np.ndarray, p2: np.ndarray, r_max: float, r_min: float) -> float:
    """Calculate the distance between two points.

    Args:
        p1 (np.ndarray): the first point
        p2 (np.ndarray): the second point
        r_max (float): the maximum distance
        r_min (float): the minimum distance

    Returns:
        float: the distance between the two points
    """
    distance = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    if distance < 2*r_min:
        return 2*r_min
    return min(distance, 2*r_max)


'''
def calculateDistance(p1:np.ndarray,p2:np.ndarray,r_max:float) -> float:
    """Calculate the distance between two points.

    Args:
        p1 (np.ndarray): the first point
        p2 (np.ndarray): the second point
        r_max (float): the maximum distance

    Returns:
        float: the distance between the two points
    """
    if math.isnan(p1[0]) or math.isnan(p2[0]):
        return 2*r_max
    distance = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return min(distance, 2*r_max)
'''

# PARAMETERS
# axis_size is the size of the image (in this case we want points with the coordinates between 0 and 1)
axis_size = 1
# input_size is the number of sample we want to generate
input_size = 800
# output_size is the number of sample we want to keep at the end
# (usually between 1/3 and 1/5 of the input_size)
output_size = int(input_size/3)
# alpha is the exponent of the weight function (it need to be even)
alpha = 8
# beta is the exponent of the minimum radius function
beta = 0.65
# theta is the factor of the minimum radius function
theta = 1.5

# set the random seed
np.random.seed()

# calculate r_max
r_max = math.sqrt((axis_size*2)/(2*math.sqrt(3)*output_size))
# calculate r_min
r_min = r_max * (1-(input_size/output_size)**beta)*theta

# generate random points
points = np.random.rand(input_size, 2)

# build a 2-d tree
tree = spatial.KDTree(points)
neighbours = getNeighbours(points,tree,r_max)
# assign the weights to each point (0 at the beginning)
weights = np.zeros(len(points))
for i in range(len(points)):
    print(i, "/", len(points))
    # get the index of the points within the radius
    indexes = tree.query_ball_point(points[i], 2*r_max)
    for idx in indexes:
        # checking if the point is not the same as the one we are calculating the weight for
        if i != idx and (not math.isnan(points[idx][0])):
            # calculate the distance between the two points
            distance = calculateDistance(points[i], points[idx], r_max, r_min)
            # calculate the weight
            weights[i] += (1-(distance/(2*r_max)))**alpha

# build the max-heap of the weights
heap = generateHeap(weights)

# eliminate the points with the highest weight until the output size is reached
while len(heap) > output_size:
    print(len(heap), "/", output_size, len(heap)*100/output_size, "%")
    # pop the highest weight (it has the index of the point)
    pop = heapq.heappop(heap)
    # the the index of the points within the radius from the popped point
    indexes = tree.query_ball_point(points[pop[1]], 2*r_max) 
    for idx in indexes:
        # checking if the point is not the same as the popped one
        print(points[idx])
        if idx != pop[1] and (not math.isnan(points[idx][0])):
            # get the distance between the two points
            distance = calculateDistance(
                points[pop[1]], points[idx], r_max, r_min)
            # save the current weight
            app_weight = weights[idx]
            # decrease the weight of the point since we are eliminating the popped point
            weights[idx] -= (1-(distance/(2*r_max)))**alpha
            # if the new weight is not nan, remove the old weight from the heap and add the new one
            if not math.isnan(weights[idx]):
                heap.remove((-1*app_weight, idx)) # log n
                heap.append((-1*weights[idx], idx)) # log n
    # set the points and weights to nan
    points[pop[1], 0] = np.nan
    points[pop[1], 1] = np.nan
    weights[pop[1]] = np.nan
    # the list is not longer a heap since we made changes, so we need to re-heapify it
    heapq.heapify(heap)
# plot the points using matplotlib
plt.scatter(points[:, 0], points[:, 1], s=output_size/1000)
print("Generated a plot with", output_size, "points.")
plt.savefig('./pictures/2D_weighted_elimination.png', dpi=500)
plt.show()
