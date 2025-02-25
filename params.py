import yaml

class ExperimentParams:

    def __init__(self,filename):
        with open(filename, 'r') as file:
            self.params = yaml.safe_load(file)

    def getCarpetaIter(self):
        return self.params["carpetas"]["carpeta_iter"]
    
    def getDireccionCarpeta(self):
        return self.params["carpetas"]["direccion"]
    
    def getObjetoCarpeta(self):
        return self.params["carpetas"]["objeto"]
    
    def getPesosCarpeta(self):
        return self.params["carpetas"]["path_weights"]
    
    def getCSVName(self):
        return self.params["carpetas"]["csv_name"]
    
    def getUmbralVariable(self):
        return self.params["variables"]["umbral"]
    
    def getVoxelVariable(self):
        return self.params["variables"]["voxel_resolution"]
    
    def getVoxelVariable(self):
        return self.params["variables"]["maximum_views"]
    
    def getCameraParams(self):
        return self.params["camera"]["img_H"], self.params["camera"]["img_W"], self.params["camera"]["up"],self.params["camera"]["fov"]
    
    
