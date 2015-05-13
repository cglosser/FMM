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
def box_interactions(grid, interaction_fn):
    return [np.sum([interaction_fn(src_box, obs_box) for (src_id, src_box) 
            in enumerate(grid.boxes) if not grid.nearby(src_id, obs_id)], 0)
            for (obs_id, obs_box) in enumerate(grid.boxes)]

def main():
    for numParticles in 2**np.arange(5,15):
        print(numParticles)
        sources = FMM.construct_sources(numParticles, 100)
        sim_grid = FMM.Grid(sources)

        t1, naiive_result = box_interactions(sim_grid, FMM.naiive_interaction)
        t2, fmm_result    = box_interactions(sim_grid, FMM.fmm_interaction)
        
        naiive_evals = np.hstack(naiive_result)
        fmm_evals    = np.hstack(fmm_result)

        error = np.linalg.norm(fmm_evals - naiive_evals)/np.linalg.norm(naiive_evals)

        print("Naiive timing:  {}".format(t1))
        print("FMM timing:     {}".format(t2))
        print("Num boxes:      {}".format(sim_grid.num_boxes))
        print("Absolute error: {}".format(error))
        print()

if __name__ == "__main__":
    main()
