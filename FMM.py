import numpy as np
from numpy.linalg  import norm
from scipy.special import hankel2
from collections   import namedtuple
from itertools     import groupby

NUM_ANGLE_QUADRATURE  = 32
DISCRETE_ANGLES       = np.linspace(0, 2*np.pi, NUM_ANGLE_QUADRATURE)
DISCRETE_KHAT_VECTORS = np.transpose([np.cos(DISCRETE_ANGLES), 
                                      np.sin(DISCRETE_ANGLES)])
K_NORM = 1.0
HARMONIC_MAX = 5

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
        pts     = pts[box_ids.argsort()]
        box_ids = sorted(box_ids)

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
        self.wavefunctions = self.planewaves()

    def planewaves(self):
        """Construct terms in the planewave expansion for each point in Box,
        evaluated at each alpha angle in DISCRETE_ANGLES. Uses the (-i*k.r)
        convention for *source* points; requires a conjugation for
        *observation* points.
        """
        norms = np.array([np.linalg.norm(pt - self.location)
            for pt in self.points])
        point_angles = np.arctan2(self.points[:, 1], self.points[:, 0])
        cos_arg = np.array([alpha - point_angles 
            for alpha in DISCRETE_ANGLES])
        return np.exp(-1j*K_NORM*norms*np.cos(cos_arg))

    def source_expansion(self):
        """Accumulate all of the planewave expansions, weighted by each 
        point-source's current, for the box acting as a *source*.
        """
        pass

    def observation_expansion(self):
        """Accumulate all of the planewave expansions, weighted by each
        angle's quadrature weight, for the box acting as an *observer*.
        """
        pass

def translation_operator(box1, box2):
    delta_r = box2.location - box1.location
    dist    = norm(delta_r)
    angle   = np.arctan2(delta_r[1], delta_r[0])

    hankel_terms = np.array(list(hankel2(idx, dist)*np.exp(-1j*idx*(angle - DISCRETE_ANGLES 
        - np.pi/2)) for idx in range(-HARMONIC_MAX, HARMONIC_MAX + 1)))

    return np.sum(hankel_terms, axis=0)
