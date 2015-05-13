import FMM
import numpy as np
import time

def timing(func):
    def get_results(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        stop_time = time.time()
        return (stop_time - start_time, result)
    return get_results

@timing
def box_interactions(grid, interaction_fn):
    return [np.sum([interaction_fn(src_box, obs_box) for (src_id, src_box)
            in enumerate(grid.boxes) if not grid.nearby(src_id, obs_id)], 0)
            for (obs_id, obs_box) in enumerate(grid.boxes)]

def main():
    for num_particles in 2**np.arange(5,15):
        print(num_particles)
        sources = FMM.construct_sources(num_particles, 100)
        sim_grid = FMM.Grid(sources)

        nsq_time, nsq_result = box_interactions(sim_grid, FMM.naiive_interaction)
        fmm_time, fmm_result = box_interactions(sim_grid, FMM.fmm_interaction)

        nsq_evals = np.hstack(nsq_result)
        fmm_evals = np.hstack(fmm_result)

        error = np.linalg.norm(fmm_evals - nsq_evals)/np.linalg.norm(nsq_evals)

        print("Naiive timing:  {}".format(nsq_time))
        print("FMM timing:     {}".format(fmm_time))
        print("Num boxes:      {}".format(sim_grid.num_boxes))
        print("Absolute error: {}".format(error))
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
