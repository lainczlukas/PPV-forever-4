import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import laspy

import random
import math


def get_ground_height(las):
    ground_data = las[las.classification == 2] 
    points_data = np.stack([ground_data.Z], axis = 0).transpose((1,0))
    points_sum = 0
    data_min = min(points_data)

    for i in range(len(points_data)):
        points_sum += points_data[i][0] - data_min[0]

    return math.floor(points_sum / len(points_data) + data_min[0])


def get_tree_eps(tree_data, avg_ground_height):
    points_data = np.stack([tree_data.Z], axis = 0).transpose((1,0))
    points_data = sorted(points_data, key=lambda x: x[0], reverse=True)
    points_data = points_data[:100]

    points_sum = 0
    data_min = min(points_data)

    for i in range(len(points_data)):
        points_sum += points_data[i][0] - data_min[0]

    avg_tree_height = math.floor(points_sum / len(points_data) + data_min[0])
    return (avg_tree_height - avg_ground_height) // 4


def paint_vertices(labels):
    random_colors = []
    for label in labels:
        if label == -1:
            random_colors.append([0,0,0])
        else:
            random_colors.append([random.random()*0.8 +0.2,random.random()*0.8 +0.2,random.random()*0.8 +0.2])
    colors = [random_colors[x] for x in labels]
    return colors


def get_cluster_sizes(labels):
    cluster_sizes = []
    for i in set(labels):
        cluster_sizes.append((i, labels.count(i)))
    
    #plt.figure()
    #plt.plot([x for x in range(max(labels)+2)],[x[1] for x in cluster_sizes])
    #plt.show()
    return cluster_sizes


def separate_clusters_by_size(pointcloud, labels):
    clusters_sorted = sorted(get_cluster_sizes(labels), key=lambda x: x[1], reverse=True)
    clusters_sizes_sorted = [x[1] for x in clusters_sorted]
    avg_cluster_size = sum(clusters_sizes_sorted) / len(clusters_sizes_sorted)

    well_clustered_points = []
    well_clustered_labels = []
    bad_clustered_points = []
    bad_clustered_labels = []
    all_points = np.asarray(pointcloud.points)

    for i in range(len(all_points)):
        for cluster in clusters_sorted:
            """
                For each point, find cluster of point. 
                If the cluster is smaller than 3*avereage_cluster_size, than put point to well separated part of dataset. 
                If the cluster is bigger, put the point to bad separated part
            """
            if labels[i] == cluster[0]:
                if cluster[1] > avg_cluster_size * 3:
                    bad_clustered_points.append(all_points[i])
                    bad_clustered_labels.append(labels[i])
                else:
                    well_clustered_points.append(all_points[i])
                    well_clustered_labels.append(labels[i])

    return well_clustered_points, well_clustered_labels, bad_clustered_points, bad_clustered_labels


def cluster_selected(pointcloud, labels):
    well_clustered_points, well_clustered_labels, bad_clustered_points, bad_clustered_labels = separate_clusters_by_size(pointcloud, labels)

    big_cluster = o3d.geometry.PointCloud()
    big_cluster.points = o3d.utility.Vector3dVector(bad_clustered_points)
    o3d.visualization.draw_geometries([big_cluster])

    all_points = well_clustered_points
    all_labels = well_clustered_labels

    """For each bad cluster"""
    for label in set(bad_clustered_labels):
        points_of_one_bad_cluster = []
        """Create list of all points of one big cluster"""
        for i in range(len(bad_clustered_points)):
            if bad_clustered_labels[i] == label:
                points_of_one_bad_cluster.append(bad_clustered_points[i])

        big_cluster.points = o3d.utility.Vector3dVector(points_of_one_bad_cluster)
        """"DBSCAN one big cluster"""
        labels = np.array(big_cluster.cluster_dbscan(tree_eps/3, 8))

        label_baseline = max(well_clustered_labels) + 1

        """Append DBSCANNED big cluster to well separated points/labels"""
        for j in range(len(labels)):
            if labels[j] != -1:
                all_points.append(points_of_one_bad_cluster[j])
                all_labels.append(labels[j] + label_baseline)

    return all_points, all_labels



def get_cluster_dimensions(cluster, scale=1):
    x_axis = [x[0] for x in cluster]
    y_axis = [x[1] for x in cluster]
    z_axis = [x[2] for x in cluster]
    
    x_size = (max(x_axis) - min(x_axis)) * scale 
    y_size = (max(y_axis) - min(y_axis)) * scale
    z_size = (max(z_axis) - min(z_axis)) * scale
    
    centroid = [(max(x_axis) + min(x_axis)) / 2, (max(y_axis) + min(y_axis)) / 2, (max(z_axis) + min(z_axis)) / 2]
    
    return (x_size, y_size, z_size, centroid)



def write_centroinds(points, labels):
    with open('data.csv', 'w') as f:
        f.write('label;x_size;y_size;z_size;centroid;num_points\n')

        for label in set(labels):
            cluster_points = []
            for i in range(len(points)):
                if labels[i] == label:
                    cluster_points.append(points[i])

            x_size, y_size, z_size, centroid = get_cluster_dimensions(cluster_points)
            f.write("{};{};{};{};{};{}\n".format(label, x_size, y_size, z_size, centroid, len(cluster_points)))



if __name__ == "__main__":
    las = laspy.read('data/banska4.las')
    tree_data = las[las.classification == 5]

    avg_ground_height = get_ground_height(las)
    tree_eps = get_tree_eps(tree_data, avg_ground_height)

    points_data = np.stack([tree_data.X,tree_data.Y,tree_data.Z], axis = 0).transpose((1,0))
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points_data)
    pointcloud = pointcloud.remove_radius_outlier(3,1000)[0]

    labels = np.array(pointcloud.cluster_dbscan(tree_eps, 6)).tolist()
    pointcloud.colors = o3d.utility.Vector3dVector(paint_vertices(labels))
    o3d.visualization.draw_geometries([pointcloud])

    all_points = []
    for i in range(3):
        all_points, labels = cluster_selected(pointcloud, labels)
        pointcloud.points = o3d.utility.Vector3dVector(all_points)
        pointcloud.colors = o3d.utility.Vector3dVector(paint_vertices(labels))
        o3d.visualization.draw_geometries([pointcloud])
    
    write_centroinds(all_points, labels)