import os
import open3d as o3d
import octomap
import torch
import numpy as np
import pandas as pd
from symlink import symbolic_dir
from utils_o3d_v2 import Get_Pointcloud, Get_RGBD, Get_octree, Get_PointcloudGT, scale_and_translate, Get_voxpoints
from MLP import MLP
from utils import net_position_nbv
from dataset_download import download_collection
from utils_metrics import chamfer_distance, Get_cloud_distance, getCobertura
from utils_save import GuardarDS
from params import ExperimentParams
from utils_meshload import SceneLoader


class executeExperiment:

    def __init__(self,file_name):
        self.file_name = file_name
        self.metricas = {"ID": [],"id_objeto": [], "iteracion_objeto":[],"pose_inicial":[], "nube_puntos":[], "rejilla":[], "nbv":[], "id_anterior":[], "id_siguiente":[], "chamfer":[], "ganancia_cobertura":[], "cobertura":[]}
        self.loadParams()
        
    def readParams(self):
        self.params = ExperimentParams(self.file_name)
    
    def loadParams(self):
        self.readParams()
        self.carpeta_iter = self.params.getCarpetaIter()
        self.direccion = self.params.getDireccionCarpeta()
        self.objeto = self.params.getObjetoCarpeta()
        self.direccion = self.direccion + self.objeto + "/"
        self.listado_objetos = os.listdir(self.direccion)
        self.weights_path = self.params.getPesosCarpeta()
        self.csv_name = self.params.getCSVName()
        self.umbral = self.params.getUmbralVariable()
        self.img_H, self.img_W, self.up, self.fov = self.params.getCameraParams()
        self.voxel_resolution = self.params.getVoxelVariable()

    def runExperiment(self):
        I = 0
        for l in range (0, len(self.listado_objetos)):
            #carpeta = input("A que carpeta quieres acceder?: ") #object folder
            dir_carpeta = self.direccion + self.listado_objetos[l] + "/"
            if os.path.lexists( dir_carpeta +"Point_cloud/"+ self.carpeta_iter) == False:
                #os.mkdir(dir_carpeta +"Point_cloud/Voxnet/")
                #os.mkdir(dir_carpeta +"Octree/Voxnet/")
                #os.mkdir(dir_carpeta + "RGB/Voxnet/" )
                #os.mkdir(dir_carpeta + "Depth/Voxnet/")
                os.mkdir(dir_carpeta +"Point_cloud/"+ self.carpeta_iter)
                os.mkdir(dir_carpeta +"Octree/"+ self.carpeta_iter)
                os.mkdir(dir_carpeta + "RGB/" + self.carpeta_iter)
                os.mkdir(dir_carpeta + "Depth/"+ self.carpeta_iter)

            
            #Cargamos los modelos de predicción de posición
            model= MLP().cuda() 
            ## Modificar direccion de pesos
            model.load_state_dict(torch.load(self.weights_path))
            device = torch.cuda.current_device()

            #Inicializamos el octomap
            octree = octomap.OcTree(self.voxel_resolution) # inicializamos el octree

            #Cargamos malla
            miEscena = SceneLoader(dir_carpeta,False)
            render, scene = miEscena.get_scenes(self.img_H,self.img_W)
            #Obtenemos pointcloud GT
            Get_PointcloudGT(dir_carpeta, miEscena.mesh, self.carpeta_iter)

            #Camera vectors setup
            cent = miEscena.mesh.get_center()
            
            poses = np.load("poses.npy")
            eye_init = poses[116]
            eye = eye_init
            
            puntos = Get_voxpoints()

            
            
            #print("Inicia el proceso de reconstrucción ...")
            #while condicion == False:
            for i in range(0,15):    
                # RGBD and pointcloud extraction
                Get_Pointcloud(scene, self.fov, cent, eye, self.up, self.img_W, self.img_H, dir_carpeta, i, self.carpeta_iter, save_acc= True)
                Get_RGBD(render,  self.fov, cent, eye, self.up, dir_carpeta, i, self.carpeta_iter)
                #Occupancy grid
                occupancy_probs =  Get_octree(octree, dir_carpeta, i, self.carpeta_iter, eye, puntos)
                ## Aqui evaluamos si esta completo el modelo en este punto
                CD = chamfer_distance(dir_carpeta, self.carpeta_iter)
                condicion, coverage_gain = Get_cloud_distance(dir_carpeta, i, self.carpeta_iter)
                cov = getCobertura(dir_carpeta, self.carpeta_iter, i, umbral=self.umbral)
                #print("Chamfer Distance: {}, Cloud distances: {}, # view: {}".format(CD, Distance, i))
                if condicion == True:
                    GuardarDS(self.metricas,I, i, self.listado_objetos[l], eye_init, eye, octree, occupancy_probs, dir_carpeta, self.carpeta_iter, CD, coverage_gain, cov)
                    break
                ## De no estarlo, se consulta a la NN el NBV 
                else:
                    grid = np.reshape(occupancy_probs, (1,1,31,31,31))  
                    torch_grid = torch.from_numpy(grid)
                    #IA-NBV
                    output = net_position_nbv(model, torch_grid, device) 
                    eye = output.numpy().reshape(3,).astype("double")
                    GuardarDS(self.metricas,I, i, self.listado_objetos[l], eye_init, eye, octree, occupancy_probs, dir_carpeta, self.carpeta_iter, CD, coverage_gain,cov) 
                #print("nbv:", eye)
                I += 1
            del octree
            del scene
            del render
            del miEscena

        #print(metricas)   
        #print("Volví, tonotos!")
        #almacena las métricas de error en archivo NPZ
        dataframe = pd.DataFrame(self.metricas, index=None)
        dataframe.to_csv(self.csv_name ,index=False)