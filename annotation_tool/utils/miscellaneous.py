from scipy.spatial.transform import Rotation
import numpy as np


def avg_poses(poses):
    rots = []
    trans = []
    for pose in poses:
        rots.append(Rotation.from_matrix(pose[:3,:3]).as_rotvec())
        trans.append(pose[:3, -1])

    res = np.eye(4)
    res[:3, :3] = Rotation.from_rotvec(np.stack(rots).mean(0)).as_matrix()
    res[:3, -1] = np.stack(trans).mean(0)
    return res


def next_available_start(keys, cnt):
    if len(keys) == 0:
        return 1
    keys.sort(key=lambda x: x[0])
    if cnt < keys[0][0]:
        return 1
    
    idxs = []
    for key in keys:
        idxs.extend(list(range(key[0], key[1] + 1)))
    
    start = None
    for i, idx in enumerate(idxs[:-1]):
        if idx + cnt < idxs[i + 1]:
            start = i + 1
            break
    if start is None:
        start = idxs[-1] + 1
    return start