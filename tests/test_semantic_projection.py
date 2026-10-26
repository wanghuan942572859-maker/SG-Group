import unittest

import torch

from utils.semantic_projection import segment_features_for_gaussians


class SemanticProjectionTest(unittest.TestCase):
    def test_different_segments_get_different_features(self):
        xyz = torch.tensor([
            [-0.5, 0.0, 0.5], [0.5, 0.0, 0.5],
            [0.0, -0.5, 0.5], [2.0, 0.0, 0.5],
        ])
        projection = torch.eye(4)
        radii = torch.tensor([1.0, 1.0, 0.0, 1.0])
        seg_map = torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ])
        features = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        indices, assigned = segment_features_for_gaussians(
            xyz, projection, radii, seg_map, features
        )

        self.assertEqual(indices.tolist(), [0, 1])
        self.assertTrue(torch.equal(assigned, features))

    def test_unsegmented_pixel_is_skipped(self):
        xyz = torch.tensor([[0.0, 0.0, 0.5]])
        seg_map = torch.full((4, 4), -1)
        indices, assigned = segment_features_for_gaussians(
            xyz, torch.eye(4), torch.ones(1), seg_map, torch.ones((1, 3))
        )
        self.assertEqual(indices.numel(), 0)
        self.assertEqual(assigned.shape, (0, 3))


if __name__ == '__main__':
    unittest.main()
