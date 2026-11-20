import unittest

from violence_detector import _format_time, _merge_segments, _video_windows


class VideoWindowTests(unittest.TestCase):
    def test_short_video_has_one_window(self):
        self.assertEqual(_video_windows(1.5), [0.0])

    def test_long_video_adds_final_window_to_cover_end(self):
        starts = _video_windows(5.2)
        self.assertEqual(starts[0], 0.0)
        self.assertAlmostEqual(starts[-1], 2.7)

    def test_flags_adjacent_overlapping_windows_as_one_segment(self):
        result = _merge_segments(
            [(0.0, 2.5, 0.7), (1.0, 3.5, 0.8), (2.0, 4.5, 0.2)],
            threshold=0.5,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["start"], 0.0)
        self.assertEqual(result[0]["end"], 3.5)
        self.assertEqual(result[0]["confidence"], 0.8)

    def test_formats_timestamp(self):
        self.assertEqual(_format_time(3661.234), "01:01:01.234")


if __name__ == "__main__":
    unittest.main()
