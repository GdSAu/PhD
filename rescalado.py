import numpy as np

def trilinear_interpolation(point, min_vox, max_vox, voxel_values):
    
    differences = (point - min_vox) / (max_vox - min_vox)
    #print(differences.shape)

    c00 = voxel_values[:,0]*(1-differences[:,:,0]) + voxel_values[:,4]*differences[:,:,0]
    c01 = voxel_values[:,1]*(1-differences[:,:,0]) + voxel_values[:,5]*differences[:,:,0]
    c10 = voxel_values[:,2]*(1-differences[:,:,0]) + voxel_values[:,6]*differences[:,:,0]
    c11 = voxel_values[:,3]*(1-differences[:,:,0]) + voxel_values[:,7]*differences[:,:,0]

    c0 = c00*(1-differences[:,:,1]) + c10*differences[:,:,1]
    c1 = c01*(1-differences[:,:,1]) + c11*differences[:,:,1]
    
    c = c0*(1-differences[:,:,2]) + c1*differences[:,:,2]

    return c


def get_vertex(points, bounding_box):
    #TODO: revisar los valores de vertices min y max, parece que conforme avanza el proceso se desplaza, por eso se visualizan esos deplazamientos de voxeles
    
    min, max = bounding_box  
    grid_size = (max-min)
    bottom_left_idx = np.floor(points-min)/grid_size
    voxel_min_vertex = bottom_left_idx*grid_size + min
    voxel_max_vertex = voxel_min_vertex + np.asarray([1.0,1.0,1.0]) # Esto hace que se salga de los limites del grid
    #condicion = ((voxel_max_vertex >= max)) # condicion para limitar los valores maximos dentro del grid
    #voxel_max_vertex[condicion] = voxel_min_vertex[condicion]

    return voxel_min_vertex, voxel_max_vertex

def voxel_resizing(given_point, original_size, desired_size):
    C_1 = original_size/2
    C_2 = desired_size/2

    K = np.min(desired_size/original_size)

    V_1 =C_1 - given_point #Sustituir X por P1 y P1 es la coordenada original 
    V_2 = V_1 * K
    P_2 = C_2 - V_2 # P2 es la coordenada escalada
    return P_2

def neighbor_vertex(max_vertex, obj): 
    
    #Obtenemos los vertices/indices de los voxeles vecinos
    v8 = (max_vertex[0,:] - [0,0,0]).astype(int)
    v7 = (max_vertex[0,:] - [1,0,0]).astype(int)
    v6 = (max_vertex[0,:] - [0,1,0]).astype(int)
    v5 = (max_vertex[0,:] - [1,1,0]).astype(int)
    v4 = (max_vertex[0,:] - [0,0,1]).astype(int)
    v3 = (max_vertex[0,:] - [1,0,1]).astype(int)
    v2 = (max_vertex[0,:] - [0,1,1]).astype(int)
    v1 = (max_vertex[0,:] - [1,1,1]).astype(int)

    #Conseguimos los valores de cada vertices en el grid
    values = np.zeros((29791,8)) # variable para almacenar los valores
    values[:,0] = obj[v1[:,0],v1[:,1],v1[:,2]]
    values[:,1] = obj[v2[:,0],v2[:,1],v2[:,2]]
    values[:,2] = obj[v3[:,0],v3[:,1],v3[:,2]]
    values[:,3] = obj[v4[:,0],v4[:,1],v4[:,2]]
    values[:,4] = obj[v5[:,0],v5[:,1],v5[:,2]]
    values[:,5] = obj[v6[:,0],v6[:,1],v6[:,2]]
    values[:,6] = obj[v7[:,0],v7[:,1],v7[:,2]]
    values[:,7] = obj[v8[:,0],v8[:,1],v8[:,2]]

    return values