import FMM
import numpy as np
from scipy.special import hankel2
import timeit

def timing(func, *args, **kwargs):
    curried = lambda: func(*args, **kwargs)
    return timeit.timeit(curried, number = 5)

def naiive_potential(sources):
    pts = [s.location for s in sources]
    return [hankel2(0, FMM.K_NORM*np.linalg.norm(p2 - p1))
            for p2 in pts for p1 in pts if p2 is not p1]

def fmm_potential(sources, grid_len):
    g = FMM.Grid(grid_len, sources)
    num_boxes = g.boxes_per_row**2
    return [g.compute_box_interaction(i,j) for i in range(num_boxes)
            for j in range(num_boxes) if i != j]

def main():
    sources = FMM.construct_sources(512, 100)

    print(timing(naiive_potential, sources))
    print(timing(fmm_potential, sources, 100))

if __name__ == "__main__":
    main()
