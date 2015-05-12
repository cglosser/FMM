import FMM
import numpy as np
from scipy.special import hankel2
import time

def timing(func):
    def get_results(*args, **kwargs):
        t1 = time.time()
        r = func(*args, **kwargs)
        t2 = time.time()
        return (t2 - t1, r)
    return get_results

@timing
def fmm_interactions(sources, interaction_fn):
    sim_grid = FMM.Grid(sources)
    return np.array([interaction_fn(b1,b2) 
                     for b1 in sim_grid.boxes 
                     for b2 in sim_grid.boxes if b1 is not b2])

def main():
    for numParticles in 2**np.arange(5,12):
        sources = FMM.construct_sources(numParticles, 5)

        sim_grid = FMM.Grid(sources)

        print(numParticles)
        t1, naiive_result = naiive_interaction(sim_grid.boxes[-1], sim_grid.boxes[1])
        t2, fmm_result    = (0, 0) #timing(FMM.box_interaction, sim_grid.boxes[-1], sim_grid.boxes[1])
        
        print(fmm_result, "\n", naiive_result)
        delta = fmm_result - naiive_result
        print("Numerical error in FMM: {}".format(
            np.linalg.norm(delta)/np.linalg.norm(naiive_result)
            )
        )
        print("Naiive timing: {}".format(t1))
        print("FMM timing:    {}".format(t2))

if __name__ == "__main__":
    main()
