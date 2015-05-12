import numpy as np
from scipy.special import hankel2
from collections   import namedtuple
from itertools     import groupby

NUM_ANGLE_QUADRATURE  = 32
DELTA_THETA           = 2*np.pi/(NUM_ANGLE_QUADRATURE - 1)
DISCRETE_ANGLES       = np.linspace(0, 2*np.pi, NUM_ANGLE_QUADRATURE)
DISCRETE_KHAT_VECTORS = np.transpose([np.cos(DISCRETE_ANGLES),
                                      np.sin(DISCRETE_ANGLES)])
K_NORM = 1.0
HARMONIC_MAX = 16

PointCurrent = namedtuple("PointCurrent", ["location","current"])

class Grid(object):
    def __init__(self, sources, num_boxes = None):
        self.sources = sources
        self.dimensions = 1.02*np.amax([s.location for s in sources], 0)
        # 1.02 is a fudge to include maximal points in x & y; assumes
        # everything lies in the first quadrant and boxes start at the origin

        if num_boxes is None:
            # gives optimal sqrt(n) particles-per-box
            boxes_per_row = int(np.ceil(len(self.sources)**0.25)) 
            self.num_boxes      = np.array([boxes_per_row, boxes_per_row])
            self.box_dimensions = self.dimensions/self.num_boxes
        else:
            self.num_boxes      = np.array(num_boxes)
            self.box_dimensions = self.dimensions/self.num_boxes

        self.boxes = self.__create_boxes()

    def __box_id(self, location):
        """Convert an integral (x, y) box coordinate to its unique integer id
        """
        return int(location[0] + self.num_boxes[0]*location[1])

    def __box_coords(self, box_id):
        """Convert a unique box id to integral coordinates in the box grid
        """
        boxes_per_row = self.num_boxes[0]
        col_id = np.floor(box_id/boxes_per_row)
        return np.array([box_id - col_id*boxes_per_row, col_id]).astype(int)

    def __box_points(self):
        """Determine the (x, y) integral coordinates of the box containing each
        source point
        """
        coords = np.array([i.location for i in self.sources])
        return np.floor(coords/self.box_length).astype(int)

    def __create_boxes(self):
        def source_boxid(source):
            box_ij = np.floor(source.location/self.box_dimensions).astype(int)
            return self.__box_id(box_ij)

        box_ids = np.array([source_boxid(s) for s in self.sources])
        self.sources = [self.sources[i] for i in box_ids.argsort()]

        groups = groupby(self.sources, source_boxid)

        return [Box(self.__box_coords(idx)*self.box_dimensions, list(sources))
                for idx, sources in groups]

class Box(object):
    def __init__(self, location, sources):
        self.location = location #bottom-left corner if box is in first quadrant
        self.points   = np.array([p.location for p in sources])
        self.currents = np.array([p.current for p in sources])

        #pre-compute these -- they get used a lot
        self.outgoing_rays = self.__source_expansion()
        self.incoming_rays = self.__observation_expansion()

    def __planewaves(self):
        """Construct terms in the planewave expansion for each point in Box,
        evaluated at each alpha angle in DISCRETE_ANGLES. Uses the (-i*k.r)
        convention for *source* points; requires a conjugation for
        *observation* points.
        """
        delta_r = self.points - self.location
        norms = np.array([np.linalg.norm(p) for p in delta_r])
        point_angles = np.arctan2(delta_r[:, 1], delta_r[:, 0])
        cos_arg = np.array([alpha - point_angles
            for alpha in DISCRETE_ANGLES])
        return np.exp(-1j*K_NORM*norms*np.cos(cos_arg))

    def __source_expansion(self):
        """Accumulate all of the planewave expansions, weighted by each
        point-source's current, for the box acting as a *source*.
        """
        return np.dot(self.__planewaves(), self.currents)

    def __observation_expansion(self):
        """Accumulate all of the planewave expansions, weighted by each
        angle's quadrature weight, for the box acting as an *observer*.
        """
        return np.transpose(np.conjugate(self.__planewaves()))

def box_interaction(src_box, obs_box):
    """Evaluate the H2 potential *from* every point in src_box *to*
    obs_box by way of the FMM algorithm (consequently one-directional).
    """
    box_to_box = translation_operator(src_box, obs_box)
    planewaves = obs_box.incoming_rays*box_to_box*src_box.outgoing_rays

    return np.trapz(planewaves, dx = DELTA_THETA)/(2*np.pi)

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
    return [PointCurrent(current = 1,
        location = np.random.rand(2)*box_dim) for _ in range(num)]
