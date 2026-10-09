"""A vector export must not reconnect a gap removed by the privacy pipeline."""
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point

from pinless.render_editorial import _density_segments


def test_density_coloring_does_not_bridge_private_gap_or_move_vertices():
    left = LineString([(-900, 0), (-700, 100), (-500, 0)])
    right = LineString([(500, 0), (700, -100), (900, 0)])
    original = [np.asarray(left.coords).copy(), np.asarray(right.coords).copy()]
    pieces = [(12, left), (12, right)]
    for acc in (np.ones((100, 100), dtype=np.uint16), np.full((100, 100), 80, dtype=np.uint16)):
        segments, _ = _density_segments(pieces, (-1000, 1000, -1000, 1000), acc)
        rendered = MultiLineString(segments.tolist())
        assert rendered.distance(Point(0, 0)) == 500
        assert rendered.equals(MultiLineString([left, right]))
        assert np.array_equal(np.asarray(left.coords), original[0])
        assert np.array_equal(np.asarray(right.coords), original[1])
