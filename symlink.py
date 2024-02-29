import pandas as pd, os

def symbolic(direcciones):

    for i in range(0,len(direcciones)):
        d = direcciones.iat[i,0]
        file_src = d + '/materials/textures/texture.png' #source file
        file_dst = d + '/meshes/texture.png' #destiny file
        os.symlink(file_src,file_dst) #create link

def symbolic_dir(direccion):
    d = direccion
    file_src =  d + '/materials/textures/texture.png' #source file
    file_dst =  d + '/meshes/texture.png' #destiny file
    os.symlink(file_src,file_dst) #create link