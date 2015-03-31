import numpy as np
from scipy.special import hankel2
from collections   import namedtuple

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
    def __init__(self, loc, pts):
        self.location = loc #bottom-left corner if box is in first quadrant
        self.pts = pts

    def planewaves(self):
       pass 
