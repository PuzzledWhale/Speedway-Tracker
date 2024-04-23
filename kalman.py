import numpy as np

def kalman(people, dt):
        dt2 = 0.5*dt**2
        state_transition_matrix = np.array([[   1,   0,  dt,   0, dt2,   0],
                                            [   0,   1,   0,  dt,   0, dt2],
                                            [   0,   0,   1,   0,  dt,   0],
                                            [   0,   0,   0,   1,   0,  dt],
                                            [   0,   0,   0,   0,   1,   0],
                                            [   0,   0,   0,   0,   0,   1]])

