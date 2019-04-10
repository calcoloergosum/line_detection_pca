"""Manual tests, that a human decides"""

import cv2
from pathlib import Path
import numpy as np


def import_module() -> None:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))


def test_detect_lines(im) -> None:
    canvas = np.zeros(im.shape, dtype=np.uint8)
    for pts in detect_lines(im):
        # print(pt1, pt2)
        pt1, pt2 = pts.as_point_pair()
        cv2.line(canvas, pt1, pt2, color=(0, 0, 255), thickness=1)
    for pts in detect_lines(im):
        for pt in pts.pts:
            canvas[pt[1], pt[0], 1] = 255
    cv2.imshow('DetectionResult', canvas)
    cv2.waitKey()


if __name__ == '__main__':
    import_module()
    from line_detection_pca import detect_lines
    im = cv2.imread(str(Path(__file__).parent / 'resource' / 'pentagon.png'))
    test_detect_lines(im)
