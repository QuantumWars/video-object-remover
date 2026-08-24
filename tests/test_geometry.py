from video_object_remover.config import Box, compute_window, union_window
from video_object_remover.scenes import plan_chunks


def test_compute_window_is_clamped_and_multiple_of_8():
    w = compute_window(Box(64, 74, 248, 172), 1920, 1080, pad=160, proc_scale=1.0)
    assert (w.x, w.y) == (0, 0)                 # top-left box clamps to origin
    assert w.w % 8 == 0 and w.h % 8 == 0
    assert w.w >= 248 and w.h >= 172            # covers the box
    assert w.x + w.w <= 1920 and w.y + w.h <= 1080


def test_compute_window_downscale_proc_is_multiple_of_8():
    w = compute_window(Box(120, 150, 390, 250), 3840, 2160, pad=160, proc_scale=0.5)
    assert w.proc_w % 8 == 0 and w.proc_h % 8 == 0
    assert w.proc_w < w.w and w.proc_h < w.h


def test_union_window_covers_all_bboxes():
    bboxes = [(100, 100, 200, 200), None, (500, 400, 700, 600)]
    w = union_window(bboxes, 1920, 1080, pad=20, proc_scale=1.0)
    assert w.x <= 80 and w.y <= 80                 # padded past the min corner
    assert w.x + w.w >= 700 and w.y + w.h >= 600   # reaches the far bbox


def test_union_window_roam_falls_back_to_full_frame():
    bboxes = [(10, 10, 30, 30), (1800, 1000, 1900, 1060)]  # spans ~whole frame
    w = union_window(bboxes, 1920, 1080, pad=10, proc_scale=1.0, roam_fraction=0.5)
    assert (w.x, w.y) == (0, 0)
    assert w.w >= 1912 and w.h >= 1080             # ~full frame (rounded /8)


def test_union_window_empty_returns_none():
    assert union_window([None, None], 1920, 1080, pad=10, proc_scale=1.0) is None


def test_plan_chunks_reproduces_a_real_run():
    cuts = [144, 276, 372, 478, 543, 546, 550, 553, 557, 560, 566, 569, 573, 576, 972]
    chunks = plan_chunks(1194, cuts, 500)
    assert sum(b - a + 1 for a, b in chunks) == 1194     # exact cover
    assert all(b - a + 1 <= 500 for a, b in chunks)      # size cap respected
    assert chunks[0][0] == 0 and chunks[-1][1] == 1193   # contiguous ends


def test_plan_chunks_splits_a_long_shot_with_no_cuts():
    chunks = plan_chunks(1300, cuts=[], max_size=500)
    assert sum(b - a + 1 for a, b in chunks) == 1300
    assert all(b - a + 1 <= 500 for a, b in chunks)
