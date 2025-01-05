import cv2
import numpy as np

from .utils_math import prepare_draw_frets, prepare_draw_strings

from typing import Union, List
from ..fretboard import Fretboard

def draw_houghline(image: cv2.typing.MatLike, line: Union[np.typing.NDArray, List], **kwargs) -> None:
    length = max(image.shape[0], image.shape[1])

    a = np.cos(line[1])
    b = np.sin(line[1])
    x0 = a * line[0]
    y0 = b * line[0]
    pt1 = (int(x0 + length*(-b)), int(y0 + length*(a)))
    pt2 = (int(x0 - length*(-b)), int(y0 - length*(a)))
    if 'thickness' in kwargs:
        thickness = kwargs['thickness']
    else:
        thickness = 1
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = (0, 255, 0)
    cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)

def draw_houghline_batch(image: cv2.typing.MatLike, lines: np.typing.NDArray, **kwargs) -> None:
    length = max(image.shape[0], image.shape[1])

    a = np.cos(lines[:, 1])
    b = np.sin(lines[:, 1])
    x0 = a * lines[:, 0]
    y0 = b * lines[:, 0]
    if 'thickness' in kwargs:
        thickness = kwargs['thickness']
    else:
        thickness = 1
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = (0, 255, 0)
    for i in range(lines.shape[0]):
        pt1 = (int(x0[i] + length*(-b[i])), int(y0[i] + length*(a[i])))
        pt2 = (int(x0[i] - length*(-b[i])), int(y0[i] - length*(a[i])))
        cv2.line(image, pt1, pt2, color, thickness, cv2.LINE_AA)

def draw_fretboard(image: cv2.typing.MatLike, fretboard: Fretboard) -> None:
    cv2.drawContours(image, [np.intp(cv2.boxPoints(fretboard.oriented_bb))], 0, (0,0,255), 2)

    pnts1, pnts2 = prepare_draw_frets(fretboard.oriented_bb, fretboard.frets)
    for pt1, pt2 in zip(pnts1, pnts2):
        cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=2)
    
    pnts1, pnts2 = prepare_draw_strings(fretboard.oriented_bb, fretboard.strings)
    # if pnts1.shape[0] == 2:
    #     pnts1 = pnts1[0, :] + (np.arange(0, 7)/6).reshape((7, 1)) @ (pnts1[1, :]-pnts1[0, :]).reshape((1, 2))
    #     pnts2 = pnts2[0, :] + (np.arange(0, 7)/6).reshape((7, 1)) @ (pnts2[1, :]-pnts2[0, :]).reshape((1, 2))
    #     pnts1 = pnts1.astype(int)
    #     pnts2 = pnts2.astype(int)
    for pt1, pt2 in zip(pnts1, pnts2):
        cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=2)

def dashed_vertical_line(img, x, y0, y1, color, stroke=5, gap=5):
    stride = stroke + gap
    for y in range(y0, y1):
        if (y % stride) < stroke:
            img[y,x] = color

def draw_houghlineP_batch(cdst, linesP):
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0].astype(int)
            cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,255,0), 1, cv2.LINE_AA)