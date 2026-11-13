"""Published scene-disjoint RADDet protocol (RadSimReal supplement, Table 4).

Frame numbers refer to original RADDet IDs, not positions in a sorted file list.
The table is incomplete as an available-file manifest: use only observed files.
"""

from numbers import Integral


# Inclusive intervals transcribed from RadSimReal supplementary Table 4.
# There are deliberate uncovered intervals at 556-558 and 725-730.
SCENE_RANGES: dict[int, tuple[tuple[int, int], ...]] = {
    0: ((0, 439), (559, 724), (1549, 1971)),
    1: ((440, 555), (731, 1548), (1972, 2571)),
    2: ((2572, 3038),),
    3: ((3039, 3437),),
    4: ((3438, 3653),),
    5: ((3654, 4073),),
    6: ((4074, 4331),),
    7: ((4332, 5018), (5623, 6243)),
    8: ((5019, 5622), (6244, 6608)),
    9: ((6609, 8046),),
    10: ((8047, 8634),),
    11: ((8635, 9158),),
    12: ((9159, 9437),),
    13: ((9438, 9745), (10175, 10292)),
    14: ((9746, 10174),),
}
TEST_SCENES = frozenset({9, 11})


def scene_id_for_frame(frame: int) -> int | None:
    """Return the published scene ID, or None when the table does not assign it.

    Negative, uncovered, and out-of-range integer IDs return None. Non-integer
    inputs, including bool, raise TypeError rather than silently rounding an ID.
    Availability is not inferred: a mapped ID may still lack a real data file.
    """
    if isinstance(frame, bool) or not isinstance(frame, Integral):
        raise TypeError("frame must be an integer original RADDet frame ID")
    for scene_id, intervals in SCENE_RANGES.items():
        if any(start <= frame <= end for start, end in intervals):
            return scene_id
    return None


def split_for_frame(frame: int) -> str | None:
    """Return 'test' for scenes 9/11, 'train' for other mapped scenes, else None.

    This implements the publication's held-out scenes. It does not create the
    unpublished list of available frames, or an author validation partition.
    """
    scene_id = scene_id_for_frame(frame)
    if scene_id is None:
        return None
    return "test" if scene_id in TEST_SCENES else "train"
