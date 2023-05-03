import numpy as np
import heapq
import math
import cv2
import scipy.spatial as spatial


def calculateDistance(p1:np.ndarray,p2:np.ndarray,r_max:float,r_min:float) -> float:
    """Calculate the distance between two points.

    Args:
        p1 (np.ndarray): the first point
        p2 (np.ndarray): the second point
        r_max (float): the maximum distance
        r_min (float): the minimum distance

    Returns:
        float: _description_
    """
    if math.isnan(p1[0]) or math.isnan(p2[0]):
        return 2*r_min
    distance = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return min(distance,2*r_max)

def transformDistanceDensity(distance:float,point:np.ndarray,gray_img:np.ndarray,d_max:float) -> float:
    """Transform the distance with the density of the image at the specified point.

    Args:
        distance (float): the distance to be transformed
        point (np.ndarray): the point at which the density is calculated
        gray_img (np.ndarray): the grayscaled image
        d_max (float): the maximum value of the transformed distance

    Returns:
        float: the transformed distance
    """
    if math.isnan(point[0]):
        return d_max
    y = int(point[0] * gray_img.shape[0])
    x = int(point[1] * gray_img.shape[1])
    density = gray_img[y,x]
    distance = distance * (3 - 2*(density/255)) 
    return min(distance,d_max)

def generateHeap(weights: np.ndarray) -> list:
    """Generate a MAX-heap from a list of weights.

    Args:
        weights (np.ndarray): the weights to be used to generate the heap

    Returns:
        list: the MAX-heap
    """
    heap = []
    heapq.heapify(heap)
    for i in range(len(weights)):
        if not math.isnan(weights[i]):
            # push the negative of the weight to make it a MAX-heap
            heapq.heappush(heap, (-weights[i],i)) 
    return heap

# read the image and convert it to grayscale
img = cv2.imread('./pictures/gerry.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# PARAMETERS
# axis_size is the size of the image (in this case we want points with the coordinates between 0 and 1)
axis_size = 1
# input_size is the number of sample we want to generate
input_size = 12000
# output_size is the number of sample we want to keep at the end
# (usually between 1/3 and 1/5 of the input_size)
output_size = input_size/3
# alpha is the exponent of the weight function (it need to be even)
alpha = 8
# beta is the exponent of the minimum radius function
beta = 0
# theta is the factor of the minimum radius function
theta = 1.5

# calculate r_max
r_max = 3*math.sqrt((axis_size*2)/(2*math.sqrt(3)*output_size))
# calculate r_min
r_min = r_max * (1-(input_size/output_size)**beta)*theta
# calculate d_max
d_max = 2*r_max
print(d_max)
# generate the random seed
np.random.seed()
# generate random points
points = np.random.rand(input_size,2)

# build a 2-d tree (cKDTree is a KDTree but with a c++ implementation wrapped in Cython)
tree = spatial.cKDTree(points)

# assign the weights to each point (0 at the beginning)
weights = np.zeros(len(points))
for i in range(len(points)):
    # get the index of the points within the radius
    indexes = tree.query_ball_point(points[i], 2*r_max)
    for j in range(len(indexes)):
        # checking if the point is not the same as the one we are calculating the weight for
        if i != indexes[j]:
            # calculate the distance between the two points
            distance = calculateDistance(points[i],points[indexes[j]],r_max,r_min)
            # transform the distance 
            distance = transformDistanceDensity(distance,points[i],img,d_max)
            # calculate the weight
            weights[i] += (1-(distance/d_max))**alpha
            
# build the max-heap of the weights
heap = generateHeap(weights)

# eliminate the points with the highest weight until the output size is reached
while len(heap) > output_size:
    # pop the highest weight (it has the index of the point)
    pop = heapq.heappop(heap)
    # the the index of the points within the radius from the popped point
    indexes = tree.query_ball_point(points[pop[1]], 2*r_max)
    for idx in indexes: 
        # checking if the point is not the same as the popped one
        if idx != pop[1]:
            # get the distance between the two points
            distance = calculateDistance(points[pop[1]],points[idx],r_max,r_min)
            # transform the distance
            distance = transformDistanceDensity(distance,points[idx],img,d_max)
            # save the current weight
            app_weight = weights[idx]
            # decrease the weight of the point since we are eliminating the popped point
            weights[idx] -= (1-(distance/d_max))**alpha
            # if the new weight is not nan, remove the old weight from the heap and add the new one
            if not math.isnan(weights[idx]):
                heap.remove((-app_weight,idx))
                heap.append((-weights[idx],idx))
    # set the points and weights to nan
    points[pop[1],0] = np.nan
    points[pop[1],1] = np.nan
    weights[pop[1]] = np.nan
    # the list is not longer a heap since we made changes, so we need to re-heapify it
    heapq.heapify(heap)
    
# plot the points
# make a white image
img_size = 10000
img = np.ones((img_size,img_size,3), np.uint8) * 255

# draw the points
for i in range(len(points)):
    if math.isnan(points[i,0]) == False:
        y = int(points[i,0] * img_size)
        x = int(points[i,1] * img_size)
        cv2.circle(img,(x,y), 30, (0,0,0), -1)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.imwrite('./pictures/adaptive_elimination.png',img)