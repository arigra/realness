"""Boundary and gap checks against the published scene table, independently listed."""

import unittest

from radsimreal.datasets.protocol import scene_id_for_frame, split_for_frame


class PublishedProtocolTests(unittest.TestCase):
    def test_every_published_interval_boundary(self):
        # Independently transcribed expected boundary assignments. Do not derive
        # expected results from SCENE_RANGES, which would conceal copy errors.
        expected = {
            0: 0, 439: 0, 440: 1, 555: 1, 559: 0, 724: 0, 731: 1,
            1548: 1, 1549: 0, 1971: 0, 1972: 1, 2571: 1,
            2572: 2, 3038: 2, 3039: 3, 3437: 3, 3438: 4, 3653: 4,
            3654: 5, 4073: 5, 4074: 6, 4331: 6, 4332: 7, 5018: 7,
            5019: 8, 5622: 8, 5623: 7, 6243: 7, 6244: 8, 6608: 8,
            6609: 9, 8046: 9, 8047: 10, 8634: 10, 8635: 11, 9158: 11,
            9159: 12, 9437: 12, 9438: 13, 9745: 13, 9746: 14,
            10174: 14, 10175: 13, 10292: 13,
        }
        for frame, scene in expected.items():
            with self.subTest(frame=frame):
                self.assertEqual(scene_id_for_frame(frame), scene)

    def test_holes_and_outside_ids_are_unassigned(self):
        for frame in (-1, 556, 557, 558, 725, 726, 727, 728, 729, 730, 10293):
            with self.subTest(frame=frame):
                self.assertIsNone(scene_id_for_frame(frame))
                self.assertIsNone(split_for_frame(frame))

    def test_holdout_boundaries(self):
        expected = {
            6608: "train", 6609: "test", 8046: "test", 8047: "train",
            8634: "train", 8635: "test", 9158: "test", 9159: "train",
        }
        for frame, split in expected.items():
            with self.subTest(frame=frame):
                self.assertEqual(split_for_frame(frame), split)

    def test_only_published_test_intervals_are_held_out(self):
        test_ids = [f for f in range(10293) if split_for_frame(f) == "test"]
        self.assertEqual(test_ids, list(range(6609, 8047)) + list(range(8635, 9159)))
        self.assertEqual(len(test_ids), 1962)

    def test_noninteger_ids_are_not_coerced(self):
        for frame in (True, False, 1.0, "6609", None):
            with self.subTest(frame=frame):
                with self.assertRaises(TypeError):
                    scene_id_for_frame(frame)


if __name__ == "__main__":
    unittest.main()
