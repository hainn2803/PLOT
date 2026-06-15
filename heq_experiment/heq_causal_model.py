import numpy as np
 
 
def compute_states(rows):
    rows = np.atleast_2d(np.asarray(rows, dtype=np.int64))
    w, x, y, z = rows.T
    wx = (w == x).astype(np.int64)
    yz = (y == z).astype(np.int64)
    o = (wx == yz).astype(np.int64)
    return {"WX": wx, "YZ": yz, "O": o}
 
 
def counterfactual_labels(base_state, source_state):
    return {
        "WX": (source_state["WX"] == base_state["YZ"]).astype(np.int64),
        "YZ": (base_state["WX"] == source_state["YZ"]).astype(np.int64),
    }