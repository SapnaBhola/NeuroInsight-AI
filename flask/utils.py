import numpy as np

def get_calibrated_confidence(pred, T=2.5):
    pred = np.array(pred)

    # Temperature scaling
    calibrated = 1 / (1 + np.exp(-pred / T))

    raw_conf = float(pred[0])
    calibrated_conf = float(calibrated[0])
    uncertainty = 1 - calibrated_conf

    return raw_conf, calibrated_conf, uncertainty