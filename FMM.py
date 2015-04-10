import numpy as np
from scipy.special import hankel2
from collections   import namedtuple
from itertools     import groupby

NUM_ANGLE_QUADRATURE  = 64
DELTA_THETA           = 2*np.pi/(NUM_ANGLE_QUADRATURE - 1)
DISCRETE_ANGLES       = np.linspace(0, 2*np.pi, NUM_ANGLE_QUADRATURE)
DISCRETE_KHAT_VECTORS = np.transpose([np.cos(DISCRETE_ANGLES), 
                                      np.sin(DISCRETE_ANGLES)])
K_NORM = 1.0
HARMONIC_MAX = 15

PointCurrent = namedtuple("PointCurrent", ["location","current"])

class Grid(object):
    def __init__(self, grid_length, sources):
        self.grid_length   = float(grid_length)
        self.sources       = sources
        self.grid_density  = len(self.sources)/self.grid_length**2
        self.boxes_per_row = np.ceil(len(self.sources)**0.25)
        self.box_length    = self.grid_length/self.boxes_per_row

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
    def __init__(self, location, sources):
        self.location = location #bottom-left corner if box is in first quadrant
        self.points   = np.array([p.location for p in sources])
        self.currents = np.array([p.current for p in sources])
        self.outgoing_rays = self.source_expansion()
        self.incoming_rays = self.observation_expansion()

    def planewaves(self):
        """Construct terms in the planewave expansion for each point in Box,
        evaluated at each alpha angle in DISCRETE_ANGLES. Uses the (-i*k.r)
        convention for *source* points; requires a conjugation for
        *observation* points.
        """
        delta_r = np.array([p - self.location for p in self.points])
        norms = np.array([np.linalg.norm(p) for p in delta_r])
        point_angles = np.arctan2(delta_r[:, 1], delta_r[:, 0])
        cos_arg = np.array([alpha - point_angles 
            for alpha in DISCRETE_ANGLES])
        return np.exp(-1j*K_NORM*norms*np.cos(cos_arg))

    def source_expansion(self):
        """Accumulate all of the planewave expansions, weighted by each 
        point-source's current, for the box acting as a *source*.
        """
        return np.dot(self.planewaves(), self.currents)

    def observation_expansion(self):
        """Accumulate all of the planewave expansions, weighted by each
        angle's quadrature weight, for the box acting as an *observer*.
        """
        return np.transpose(np.conjugate(self.planewaves()))

def translation_operator(box1, box2):
    """Give the sum-of-harmonics translation operator evaluated between a pair
    of Box objects. Constructs each term in a generator object for memory
    efficiency.
    """
    delta_r = box2.location - box1.location
    dist    = np.linalg.norm(delta_r)
    angle   = np.arctan2(delta_r[1], delta_r[0])

    hankel_terms = (hankel2(idx, K_NORM*dist)*np.exp(-1j*idx*
        (angle - DISCRETE_ANGLES - np.pi/2)) 
        for idx in range(-HARMONIC_MAX, HARMONIC_MAX + 1))

    return np.sum(hankel_terms, axis=0)

def construct_sources(num, box_dim = 1):
    """Assemble a collection of random pointlike current sources within 
    [0, box_dim] in both x and y.
    """
    return [PointCurrent(current = np.random.rand(), 
        location = np.random.rand(2)*box_dim) for _ in range(num)]
