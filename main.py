import FMM
import numpy as np
from scipy.special import hankel2
import timeit

def timing(func, *args, **kwargs):
    curried = lambda: func(*args, **kwargs)
    return timeit.timeit(curried, number = 2)/2

def naiive_potential(sources):
    pts = [s.location for s in sources]
    return [hankel2(0, FMM.K_NORM*np.linalg.norm(p2 - p1))
            for p2 in pts for p1 in pts if p2 is not p1]

def fmm_potential(sources, grid_len):
    g = FMM.Grid(grid_len, sources)
    return [FMM.box_interaction(b1,b2) 
            for b1 in g.boxes for b2 in g.boxes if b1 is not b2]

def naiive_interaction(src_box, obs_box):
    return np.array([np.sum([hankel2(0, FMM.K_NORM*np.linalg.norm(obs_pt -
        src_pt)) for src_pt in src_box.points]) for obs_pt in obs_box.points])

def main():
    for numParticles in 2**np.arange(5,12):
        sources = FMM.construct_sources(numParticles, 100)

        print(numParticles)
        t1 = timing(naiive_potential, sources)
        t2 = timing(fmm_potential, sources, 100)
        g = FMM.Grid(100, sources)
        
        for src_idx, source_box in enumerate(g.boxes):
            for obs_idx, obs_box in enumerate(g.boxes):
                if source_box is not obs_box:
                    fmm_result    = FMM.box_interaction(source_box, obs_box)
                    naiive_result = naiive_interaction(source_box, obs_box)

                    relative_err = np.linalg.norm(fmm_result-naiive_result)
                    absolute_err = relative_err/np.linalg.norm(naiive_result)

                    print("  Error ({}, {}):".format(src_idx, obs_idx),
                            absolute_err)

        np.savetxt("source_box.dat",g.boxes[0].points)
        np.savetxt("observer_box.dat",g.boxes[-1].points)

        print(fmm_result, "\n", naiive_result)
        print("Error:",
        np.linalg.norm(fmm_result-naiive_result)/np.linalg.norm(naiive_result),
              "\n")
        print("Naiive timing: ", t1)
        print("FMM timing: ", t2)

        input()

if __name__ == "__main__":
    main()
