import os
import open3d as o3d
import octomap
import torch
import numpy as np
from symlink import symbolic_dir
from utils_o3d import Get_Pointcloud, Get_RGBD, Get_octree, Get_PointcloudGT
from MLP import MLP
from utils import net_position_nbv
from dataset_download import download_collection
from utils_metrics import chamfer_distance, Get_cloud_distance

#pages is the number of objects you want to download
download_collection(owner_name="GoogleResearch", collection_name="Scanned Objects by Google Research", folder="obj", pages=50)



direccion = 'obj'
metricas = {"objeto": [], "chamfer": [], "distancia": [], "nbv":[] ,"vistas": []}
x = os.listdir(direccion)
for carpeta in x:
    #carpeta = input("A que carpeta quieres acceder?: ") #object folder
    direccion_disco = '/mnt/6C24E28478939C77/Saulo/ProyectoPHD'
    dir_carpeta = direccion_disco + "/" + direccion + "/" + carpeta
    if os.path.lexists(dir_carpeta + "/meshes/texture.png") == False:
        symbolic_dir(dir_carpeta)
        RGB = "/RGB"
        Depth = "/Depth"
        Point_cloud = "/Point_cloud"
        Octree = "/Octree"
        os.mkdir(dir_carpeta + RGB)
        os.mkdir(dir_carpeta + Depth)
        os.mkdir(dir_carpeta + Point_cloud)
        os.mkdir(dir_carpeta + Octree)


    img_H = 100
    img_W = 100
    #Cargamos los modelos de predicción de posición
    model= MLP().cuda() 
    path_weights = '/mnt/6C24E28478939C77/Saulo/ProyectoPHD/position/weights_entrenamiento_MLP_xavier_normal_2.pth' ## Modificar direccion de pesos
    model.load_state_dict(torch.load(path_weights))
    device = torch.cuda.current_device()

    #Inicializamos el octomap
    resolution = 0.01 # resolucion del octree
    octree = octomap.OcTree(resolution) # inicializamos el octree

    #Cargamos malla
    mesh = o3d.io.read_triangle_mesh(dir_carpeta + '/meshes/model.obj', True)
    material = o3d.visualization.rendering.MaterialRecord() # Create material
    material.albedo_img = o3d.io.read_image( dir_carpeta + '/meshes/texture.png') # Add texture
    mesh.translate([0.3,0.3,0.3]) # translate to world CF origin
    mesh.scale(1.5, center=mesh.get_center())#center = mesh.get_center()) #Scale mesh
    #Obtenemos pointcloud GT
    Get_PointcloudGT(dir_carpeta, mesh)

    # Raycasting
    mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh1)
    # render for RGBD images
    render = o3d.visualization.rendering.OffscreenRenderer(width=img_W, height=img_H) #Linux only
    render.scene.add_geometry('mesh', mesh, material)
    #Camera vectors setup
    cent = mesh.get_center()
    up = [0, 1, 0]
    poses = np.load("poses.npy")
    eye = poses[0]
    fov = 35


    
    metricas["objeto"].append(carpeta)
    #print("Inicia el proceso de reconstrucción ...")
    #while condicion == False:
    for i in range(0,20):    
        metricas["nbv"].append(eye)
        # RGBD and pointcloud extraction
        Get_Pointcloud(scene, fov, cent, eye, up, img_W, img_H, dir_carpeta, i)
        Get_RGBD(render,  fov, cent, eye, up, dir_carpeta, i)
        #Occupancy grid
        occupancy_probs =  Get_octree(octree, dir_carpeta, i, eye)
        direccion_octree= bytes(dir_carpeta + "/Octree/octree_{}.ot".format(i), encoding='utf8')
        octree.writeBinary(direccion_octree) 
        np.save(dir_carpeta + "/Octree/grid_{}.npy".format(i),occupancy_probs, allow_pickle=True)
        ## Aqui evaluamos si esta completo el modelo en este punto
        CD = chamfer_distance(dir_carpeta)
        condicion, Distance = Get_cloud_distance(dir_carpeta, i)
        #print("Chamfer Distance: {}, Cloud distances: {}, # view: {}".format(CD, Distance, i))
        metricas["chamfer"].append(CD)
        metricas["distancia"].append(Distance)
        if condicion == True:
            break
        ## De no estarlo, se consulta a la NN el NBV 
        else:
            grid = np.reshape(occupancy_probs, (1,1,31,31,31))  
            torch_grid = torch.from_numpy(grid)
            #IA-NBV
            output = net_position_nbv(model, torch_grid, device) 
            eye = output.numpy().reshape(3,).astype("double") 
        #print("nbv:", eye)
    del octree
    del mesh
    del mesh1
    del material
    del scene
    del render
    metricas["vistas"].append(i)

#print(metricas)   
#print("Volví, tonotos!")
#almacena las métricas de error en archivo NPZ
np.savez('metricas.npz', obj=metricas["objeto"] ,chamfer = metricas["chamfer"], distancias = metricas["distancia"], nbv = metricas["nbv"],numero=metricas["vistas"])