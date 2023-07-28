#!/usr/bin/env python
import math
import random
import numpy as np
from plyfile import PlyData, PlyElement


def distance_point_to_plane(point, plane):
    x, y, z = point
    a, b, c, d = plane
    distance = abs(a*x + b*y + c*z + d) / (a**2 + b**2 + c**2)**0.5
    return distance


def plane_from_points(p1, p2, p3):
    v1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    v2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
    normal = (v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0])
    d = -normal[0]*p1[0] - normal[1]*p1[1] - normal[2]*p1[2]

    a = normal[0]
    b = normal[1]
    c = normal[2]
    
    #normalize the plane
    length = math.sqrt(a**2 + b**2 + c**2)
    a /= length
    b /= length
    c /= length
    d /= length

    return (a, b, c, d)

def seq_ransac_plane_fit(points, n_iters=1000, threshold=8, n_planes=1):

    best_planes = []
    remaining_data = points

    for i in range(n_planes):
        best_plane = ransac_plane_fit(points=remaining_data, n_iters=n_iters, threshold=threshold)
        inlier_indices = []

        print("1")

        for j, (x, y, z) in enumerate(remaining_data):
            if distance_point_to_plane((x, y, z), best_plane) < threshold:
                inlier_indices.append(j)

        # print("2")
        # remaining_data = [remaining_data[j] for j in range(len(remaining_data)) if j not in inlier_indices]
        # remaining_data = np.array(remaining_data)

        # best_planes.append(best_plane)
        # print("3")

    return best_planes

def ransac_plane_fit(points, n_iters=1000, threshold=8):

    best_plane_coeffs = None
    best_inliers = None
    max_inliers = 0

    for i in range(n_iters):
        # Randomly select three points to define a plane
        random_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[random_indices]

        # Compute the normal vector of the plane using the selected points
        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        # Calculate the distance from each point to the plane
        distances = np.abs(np.dot(points - sample_points[0], normal))

        # Count the inliers (points close enough to the plane)
        inliers = points[distances < threshold]
        num_inliers = len(inliers)

        # Update the best plane if necessary
        if num_inliers > max_inliers:
            # Calculate plane coefficients [a, b, c, d]
            d = -np.dot(normal, sample_points[0])
            plane_coeffs = np.append(normal, d)

            best_plane_coeffs = plane_coeffs
            best_inliers = inliers
            max_inliers = num_inliers

    return best_plane_coeffs


def load_points_file(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            if line[0] == "#":
                continue
            x, y, z, _, _, _ = map(float, line.split())
            points.append([x, y, z])

    return points

def main():

    plydata = PlyData.read('./okay_lets.ply')
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']

    points = np.column_stack((x, y, z))

    # normal = np.array([1, 2, 3])
    # res = np.dot(points, normal)

    plane = ransac_plane_fit(points, n_iters=5000, threshold=0.005)

    rgb = []
    for j, (x, y, z) in enumerate(points):
        if distance_point_to_plane((x, y, z), plane) < 0.005:
            rgb.append((255, 0, 0))
        else:
            rgb.append((255, 255, 255))

    rgb_np = np.array(rgb)

    ver = np.hstack((points, rgb_np))

    new_ver = np.core.records.fromarrays(ver.transpose(), 
                                     names='x, y, z, red, green, blue',
                                     formats = 'f4, f4, f4, u1, u1, u1')

    print(new_ver)

    # ver = [(0, 0, 0, 0, 255, 255),
    #        (0, 1, 1, 255, 0, 255),
    #        (1, 0, 1, 255, 255, 0),
    #        (1, 1, 0, 255, 0, 0)]

    # vertex = np.array(ver, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(new_ver, 'vertex')
    PlyData([el], text=True).write('some_ascii.ply')

if __name__ == "__main__":
    main()
