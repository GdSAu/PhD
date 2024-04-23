import numpy as np
import open3d as o3d

def chamfer_distance(dir_carpeta):
    """This function measure the Chamfer distance between two point clouds,
     ground_truth and accumulated"""
    distance_1_to_2 = 0
    distance_2_to_1 = 0

    points1 = np.asarray(o3d.io.read_point_cloud( dir_carpeta + "/Point_cloud/cloud_gt.pcd").points)
    points2 = np.asarray(o3d.io.read_point_cloud( dir_carpeta + "/Point_cloud/cloud_acc.pcd").points)

    # Compute distance from each point in arr1 to arr2
    for p1 in points1:
        distances = np.sqrt(np.sum((points2 - p1)**2, axis=1))
        min_distance = np.min(distances)
        distance_1_to_2 += min_distance

    # Compute distance from each point in arr2 to arr1
    for p2 in points2:
        distances = np.sqrt(np.sum((points1 - p2)**2, axis=1))
        min_distance = np.min(distances)
        distance_2_to_1 += min_distance

    return (distance_1_to_2 + distance_2_to_1) / (len(points1) + len(points2))


def  Get_surface_coverage(direccion,i,umbral=0.001): # 0.0037 #P_t - P_t-1 < umbral
    
    pt = o3d.io.read_point_cloud(direccion + "/Point_cloud/cloud_{}.pcd".format(i), remove_nan_points=True, remove_infinite_points = True)
    pt_1 = o3d.io.read_point_cloud(direccion + "/Point_cloud/cloud_{}.pcd".format(i-1), remove_nan_points=True, remove_infinite_points = True)
    dist = np.asarray(pt.compute_point_cloud_distance(pt_1))
    ind = np.where(dist < umbral)[0]
    return ind, dist

def stop_condition(direccion,ind,i,umbral= 0.9):
    pt = o3d.io.read_point_cloud(direccion + "/Point_cloud/cloud_{}.pcd".format(i), remove_nan_points=True, remove_infinite_points = True)
    size = len(ind)/np.asarray(pt.points).shape[0]
    if size < umbral:
        condicion = False
    else:
        condicion = True
    return condicion, size