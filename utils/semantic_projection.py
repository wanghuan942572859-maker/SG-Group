import torch


def segment_features_for_gaussians(xyz, projection, radii, seg_map, features):
    """Read the segment feature at each visible Gaussian's projected center."""
    height, width = seg_map.shape
    points = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1) @ projection
    w = points[:, 3]
    valid = (radii > 0) & (w > 0) & (points[:, 2] > 0)
    x = torch.round(((points[:, 0] / (w + 1e-8) + 1) * width - 1) / 2).long()
    y = torch.round(((points[:, 1] / (w + 1e-8) + 1) * height - 1) / 2).long()
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    indices = torch.where(valid)[0]
    labels = seg_map[y[indices], x[indices]]
    has_segment = (labels >= 0) & (labels < len(features))
    return indices[has_segment], features[labels[has_segment]]
