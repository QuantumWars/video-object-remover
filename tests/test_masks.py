import cv2
import numpy as np

from video_object_remover import mask
from video_object_remover.config import Box, compute_window


def test_build_writes_mask_and_alpha(tmp_path):
    box = Box(64, 74, 248, 172)
    win = compute_window(box, 1920, 1080, pad=160, proc_scale=1.0)
    mpath = tmp_path / "mask.png"
    apath = tmp_path / "alpha.png"
    x0, y0, x1, y1 = mask.build(box, win, feather=4, mask_path=str(mpath),
                                alpha_path=str(apath))
    m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
    assert m.shape == (win.proc_h, win.proc_w)
    assert m[y0:y1, x0:x1].min() == 255           # box is filled
    assert m[0, 0] == 0                            # corner outside the box is empty
    a = cv2.imread(str(apath), cv2.IMREAD_GRAYSCALE)
    assert a.max() == 255 and a.min() == 0         # feathered alpha spans full range


def test_crop_sequence_matches_proc_dims(tmp_path):
    win = compute_window(Box(200, 200, 100, 100), 1920, 1080, pad=40, proc_scale=0.5)
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    for n in range(3):
        m = np.zeros((1080, 1920), np.uint8)
        m[220:260, 220:260] = 255                 # a blob inside the window
        cv2.imwrite(str(full_dir / f"f_{n + 1:06d}.png"), m)
    out = mask.crop_sequence(str(full_dir), win, nframes=3,
                             out_dir=str(tmp_path / "win"))
    got = cv2.imread(f"{out}/f_000001.png", cv2.IMREAD_GRAYSCALE)
    assert got.shape == (win.proc_h, win.proc_w)
    assert got.max() == 255                        # blob survived the crop+scale
