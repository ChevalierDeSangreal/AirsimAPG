import numpy as np

# PREDEINE SOME TRAJECTORIES

collected_trajectories = {
    "eight": np.array([
                [0, 0, 0],
                [-2, 2, 1],
                [0, 4, 2],
                [2, 2, 3],
                [0, 0, 4],
                [-2, -2, 3],
                [0, -4, 2],
                [2, -2, 1],
                [0, 0, 0]
            ]) * 2,
    "curve": np.array(
            [
                [-1.5, 0, 2], 
                [-1, 1, 1], 
                [-.5, -1, 2], 
                [0, -3, 3],
                [1, -2, 5],
                [2, -1, 4],
                [3, 1, 3]
            ]
        )*2,
    "flat_eight": np.array(
            [
                [2, -2, 0],
                [-2, 2, 0],
                [0, 4, 0],
                [2, 2, 0],
                [0, 0, 0],
                [-2, -2, 0],
                [0, -4, 0],
                [2, -2, 0],
                [0, 0, 0]
            ]
        ) * 1.5,
    "sinus": np.array(
            [
                [0, 0, 0],
                [0, 2, 1],
                [0, 4, -1],
                [0, 6, 1],
                [0, 8, 0],
            ]
        ) * 4,
}