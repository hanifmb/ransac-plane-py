#!/usr/bin/env python
import math
import random
import numpy as np

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

def seq_ransac_plane_fit(points, n_iters=1000, threshold=0.01, n_planes=3):

    best_planes = []
    remaining_data = points

    for i in range(n_planes):
        best_plane = ransac_plane_fit(points=remaining_data, n_iters=n_iters, threshold=threshold)
        inlier_indices = []

        for j, (x, y, z) in enumerate(remaining_data):
            if distance_point_to_plane((x, y, z), best_plane) < threshold:
                inlier_indices.append(j)

        remaining_data = [remaining_data[j] for j in range(len(remaining_data)) if j not in inlier_indices]
        remaining_data = np.array(remaining_data)

        best_planes.append(best_plane)

    return best_planes

def ransac_plane_fit(points, n_iters=1000, threshold=0.01):

    best_plane = None
    best_num_inliers = 0

    for i in range(n_iters):
        # Randomly sample three points from the list of points
        sample_indices = random.sample(range(len(points)), 3)
        sample_points = points[sample_indices]
        
        a, b, c, d = plane_from_points(sample_points[0], sample_points[1], sample_points[2])
        
        normal = (a, b, c)

        # Compute the distance of each point to the plane
        distances = abs(np.dot(points, normal) + np.dot(normal, sample_points[0]))

        # Count the number of inliers (points with distance <= threshold)
        num_inliers = np.sum(distances <= threshold)

        # Update the best plane if we found a better one
        if num_inliers > best_num_inliers:
            best_plane = [a, b, c, d]
            best_num_inliers = num_inliers
    
    print(best_num_inliers)

    return best_plane

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
    points = load_points_file("dual_lidar_planes_only.ply")
    points = np.array(points)

    planes = seq_ransac_plane_fit(points, n_iters=1000, threshold=5, n_planes=3)
    # plane = ransac_plane_fit(points, n_iters=1000, threshold=5)
    print(planes)

if __name__ == "__main__":
    main()
