import pytest

from video_object_remover.cli import build_parser, main


def test_parser_accepts_box_run():
    a = build_parser().parse_args(
        ["run", "--input", "i.mp4", "--output", "o.mp4",
         "--propainter", "PP", "--box", "1", "2", "3", "4"])
    assert a.mask == "box" and a.box == [1, 2, 3, 4]


def test_parser_accepts_sam_run_with_points():
    a = build_parser().parse_args(
        ["run", "--input", "i.mp4", "--output", "o.mp4", "--propainter", "PP",
         "--mask", "sam", "--sam-checkpoint", "c.pt",
         "--sam-point", "960", "540", "--sam-neg-point", "10", "10"])
    assert a.mask == "sam"
    assert a.sam_point == [[960, 540]] and a.sam_neg_point == [[10, 10]]


def test_run_box_without_box_errors():
    with pytest.raises(SystemExit):
        main(["run", "--input", "i.mp4", "--output", "o.mp4",
              "--propainter", "PP", "--mask", "box"])


def test_run_sam_without_prompt_errors():
    with pytest.raises(SystemExit):
        main(["run", "--input", "i.mp4", "--output", "o.mp4",
              "--propainter", "PP", "--mask", "sam", "--sam-checkpoint", "c.pt"])
