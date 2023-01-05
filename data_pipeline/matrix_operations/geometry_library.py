# Standard imports
import numpy as np
from scipy.spatial import ConvexHull


def area_of_triangle(coord1, coord2, coord3) -> float:
    x1, y1 = coord1[0], coord1[1]
    x2, y2 = coord2[0], coord2[1]
    x3, y3 = coord3[0], coord3[1]
    area = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return area


def convex_hull(points):
    hull = ConvexHull(points)
    return hull


def main():
    print("this is a library file, do not run as main.")


if __name__ == "__main__":
    main()
