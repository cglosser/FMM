import numpy as np
from numpy.linalg import norm
from scipy.special import hankel2
from collections   import namedtuple

NUM_ANGLE_QUADRATURE  = 32
DISCRETE_ANGLES       = np.linspace(0, 2*np.pi, NUM_ANGLE_QUADRATURE)
DISCRETE_KHAT_VECTORS = np.transpose([np.cos(DISCRETE_ANGLES), 
                                      np.sin(DISCRETE_ANGLES)])
K_NORM = 1.0

class Grid(object):
    def __init__(self, grid_dim, pts):
        """Construct a Grid of grid_dim Boxes on the interval [0, 1] in both x
        and y.
        """
        self.grid_dim     = grid_dim
        self.num_boxes    = grid_dim**2
        self.grid_spacing = 1/float(grid_dim)

    def __box_id(self, location):
        """Convert an integral (x, y) box coordinate to its unique integer id
        """
        return location[0] + self.grid_dim*location[1]

    def __box_points(self, points):
        """Determine the (x, y) integral coordinates of the box containing each
        item in points
        """
        return np.floor(points/self.grid_spacing).astype(int) 

    def partition_points(self, pts):
        box_ids = np.array([self.__box_id(b) for b in self.__box_points(pts)])
        pts, box_ids = pts[box_ids.argsort()], sort(box_ids)

        rle = [(i, len(list(j))) for (j, i) in groupby(box_ids)]
        box_points = [[] for _ in range(self.num_boxes)]
        total = 0
        for box, count in rle:
            box_points[box].append(pts[total:total + count])
            count += total

        return box_points



class Box(object):
    def __init__(self, location, points):
        self.location = location #bottom-left corner if box is in first quadrant
        self.points = points

    def planewaves(self):
        norms = np.array([np.linalg.norm(pt - self.location)
            for pt in self.points])
        point_angles = np.arctan2(self.points[:,1], self.points[:,0])
        cos_arg = np.array([alpha - point_angles 
            for alpha in DISCRETE_ANGLES])
        print cos_arg
        return np.sum(np.exp(-1j*K_NORM*norms*np.cos(cos_arg)), 1)

def translation_operator(box1, box2):
    def hankel_terms(p_max):
        for idx in range(-p_max, p_max + 1):
            yield hankel2(idx, dist)*np.exp(
                    -1j*idx*angle - DISCRETE_ANGLES - np.pi/2)
        
    dist  = norm(box2.location - box1.location)
    angle = np.arccos(np.dot(box1.location, box2.location)/
                (norm(box1.location)*norm(box2.location)))

    return np.sum(hankel_terms(5))




def cos_polar_angle(point):
    return point[0]/norm(point)
