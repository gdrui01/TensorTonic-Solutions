import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    points = np.asarray(points)

    # Single point: shape (3,)
    if points.ndim == 1:
        points_4d = np.hstack([points, 1.0])
        p_transformed = T @ points_4d
        return p_transformed[:3]

    # Multiple points: shape (N, 3)
    else:
        ones = np.ones((points.shape[0], 1))
        points_4d = np.hstack([points, ones])
        p_transformed = (T @ points_4d.T).T
        return p_transformed[:, :3]