from pyeit.eit.interp2d import meshgrid,weight_idw
import numpy as np

def tri2square_own(mesh_obj,n=64):
    pts = mesh_obj["node"]
    tri = mesh_obj["element"]
    xg, yg, mask = meshgrid(pts,n=n)
    im = np.ones_like(mask)
    # mapping from values on xy to values on xyi
    xy = np.mean(pts[tri], axis=1)
    xyi = np.vstack((xg.flatten(), yg.flatten())).T
    w_mat = weight_idw(xy, xyi, k=1)     #xxx 可调节参数
    # w_mat = weight_sigmod(xy, xyi,s=100)
    im = np.dot(w_mat.T, mesh_obj["perm"])
    # im = weight_linear_rbf(xy, xyi, mesh_new['perm'])
    im[mask] = 0.0
    # reshape to grid size
    im = im.reshape(xg.shape)
    return im[::-1,:]

def get_anomaly(margin=0.05,num_inclusion=2):
    """
    Args:
        margin:        inclusion边界距离
        num_inclusion: 数量

    Returns:
         anomaly：
        is_intersect：是否相交
    """
    # circle ,注意圆的xy的设置相同，另外不要贴边，不要内部相交
    if num_inclusion==2:
        x1, x2 = np.random.uniform(-0.55, 0.55, 2)
        y1, y2 = np.random.uniform(-0.55, 0.55, 2)
        r = 0.15 + np.random.rand(2) * 0.1
        assert 0.15 <= r.mean() <= 0.25
        anomaly = [
            {"x": x1, "y": y1, "d": r[0], "perm": 1.5},
            {"x": x2, "y": y2, "d": r[1], "perm": 0.5},
        ]
        is_intersect = (x1 - x2) ** 2 + (y1 - y2) ** 2 < (r.sum() + margin) ** 2

    elif num_inclusion == 3:
        x1, x2, x3 = np.random.uniform(-0.55, 0.55, 3)
        y1, y2, y3 = np.random.uniform(-0.55, 0.55, 3)
        r = 0.15 + np.random.rand(3) * 0.1
        assert 0.15 <= r.mean() <= 0.25
        anomaly = [
            {"x": x1, "y": y1, "d": r[0], "perm": 1.5},
            {"x": x2, "y": y2, "d": r[1], "perm": 0.5},
            {"x": x3, "y": y3, "d": r[2], "perm": 0.1}  # perm随意设的
        ]
        is_intersect = ((x1 - x2) ** 2 + (y1 - y2) ** 2 < (r[0] + r[1] + margin) ** 2) or \
                       ((x1 - x3) ** 2 + (y1 - y3) ** 2 < (r[0] + r[2] + margin) ** 2) or \
                       ((x3 - x2) ** 2 + (y3 - y2) ** 2 < (r[2] + r[1] + margin) ** 2)


    elif num_inclusion == 4:
        x1, x2, x3, x4 = np.random.uniform(-0.55, 0.55, 4)
        y1, y2, y3, y4 = np.random.uniform(-0.55, 0.55, 4)
        r = 0.15 + np.random.rand(4) * 0.1
        assert 0.15 <= r.mean() <= 0.25
        anomaly = [
            {"x": x1, "y": y1, "d": r[0], "perm": 1.5},
            {"x": x2, "y": y2, "d": r[1], "perm": 0.5},
            {"x": x3, "y": y3, "d": r[2], "perm": 0.1},
            {"x": x4, "y": y4, "d": r[3], "perm": 0.01}  # perm随意设的
        ]
        is_intersect = ((x1 - x2) ** 2 + (y1 - y2) ** 2 < (r[0] + r[1] + margin) ** 2) or \
                       ((x1 - x3) ** 2 + (y1 - y3) ** 2 < (r[0] + r[2] + margin) ** 2) or \
                       ((x3 - x2) ** 2 + (y3 - y2) ** 2 < (r[1] + r[2] + margin) ** 2) or \
                       ((x4 - x1) ** 2 + (y4 - y1) ** 2 < (r[3] + r[0] + margin) ** 2) or \
                       ((x4 - x2) ** 2 + (y4 - y2) ** 2 < (r[3] + r[1] + margin) ** 2) or \
                       ((x4 - x3) ** 2 + (y4 - y3) ** 2 < (r[3] + r[2] + margin) ** 2)

    return anomaly, is_intersect