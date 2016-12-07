from __future__ import division
import numpy as np


def llh_fn(curr_value):
    return curr_value


def slice_update(cur_value, llh_fn, bounds , pdf_params =(), initial_bracket_width=2):
    curr_ll = llh_fn(cur_value, *pdf_params)
    jittered_cur_ll = curr_ll + np.log(np.random.rand())
    split_location = np.random.rand()
    x_l = max([bounds[0], (cur_value - split_location * initial_bracket_width)])
    x_r = min([bounds[1], (cur_value + (1-split_location) * initial_bracket_width)])

    # stepping out
    while llh_fn(x_l, *pdf_params) > jittered_cur_ll and x_l > bounds[0]:
        x_l = max([bounds[0], (x_l - initial_bracket_width)])
    while llh_fn(x_r,  *pdf_params) > jittered_cur_ll and x_r < bounds[1]:
        x_r = min([bounds[1], (x_r + initial_bracket_width)])

    new_ll = 0
    new_value = 0
    while True:
        new_value = x_l + (x_r - x_l) * np.random.rand()
        new_ll = llh_fn(new_value, *pdf_params)
        if new_ll > jittered_cur_ll:
            break
        else:
            if new_value > cur_value:
                x_r = new_value
            elif new_value < cur_value:
                x_l = new_value
            else:
                print '*** warning: shrunk to initial value ***\n'
                break
    return new_value, new_ll
