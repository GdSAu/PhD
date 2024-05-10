import open3d as o3d
import numpy as np


## Clase que plotea el objeto y las vistas capturadas por el NBV
# Leer objeto
# Leer las camaras
# Crear la pantalla
# Mostrar
from utils_o3d import scale_and_translate

class Visualizar():

    def __init__(self):
        self.mesh
        self.material
        self.__vis = None

    def read_model(self, dir_carpeta):
        self.mesh = o3d.io.read_triangle_mesh(dir_carpeta + '/meshes/model.obj', True)
        self.material = o3d.visualization.rendering.MaterialRecord() # Create material
        self.material.albedo_img = o3d.io.read_image( dir_carpeta + '/meshes/texture.png') # Add texture
        self.mesh = scale_and_translate(self.mesh, scale_factor=0.39)
        self.__vis.add_geometry("mesh", self.mesh, self.material)
        self.__vis.poll_events()
        self.__vis.update_renderer()
    
    def add_cameras(self, dir_carpeta):
        frames = []
        param = o3d.io.read_pinhole_camera_trajectory(dir_carpeta + "/trayectoria_camara.json")
        for i in range (0,len(param.parameters)):
            extrinsic = param.parameters[i].extrinsic
            intrinsic = param.parameters[i].intrinsic.intrinsic_matrix
            width = param.parameters[i].intrinsic.width
            height = param.parameters[i].intrinsic.height
            camera_model = draw_camera(intrinsic, extrinsic, width, height)
            frames.extend(camera_model)
        for i in frames:
            self.__vis.add_geometry(i)
    
    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()
    
    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()



def draw_camera(I,E,w,h,scale=1.0,color=[0.8,0.2,0.8]):
  ''' 
  I: intrinsic matrix
  E: extrinsic matrix
  w: image width
  h: image height
  scale: camera scale
  color: color of the image 
  -> Camera model geometries: axis, plane, pyramid
  '''
  C = I.copy() / scale
  C_inv = np.linalg.inv(C)

  axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.25 * scale)
  axis.transform(E)

  # points in pixel coordinates
  pix_points = [[0,0,0],
    [0,0,1],
    [w,0,1],
    [0,h,1],
    [w,h,1]
  ]

  # pixel to camera
  points = [C_inv @ p for p in pix_points]

  #image plane 
  width_w = abs(points[1][0]) + abs(points[3][0])
  height_w = abs(points[1][1]) + abs(points[3][1])
  plane = o3d.geometry.TriangleMesh.create_box(width_w, height_w, depth = 1e-6)
  plane.paint_uniform_color(color)
  plane.translate([points[1][0], points[1][1], -scale])
  plane.transform(E)

  # view pyramid
  points_w = np.asarray([(E[:3,:3] @ p + E[:3,3:]) for p in points]).reshape(15,3)
  lines = [[0, 1],
  [0, 2],
  [0, 3],
  [0, 4]
  ]
  colors = [color for i in range(len(lines))]
  line_set = o3d.geometry.LineSet(
    points = o3d.utility.Vector3dVector(points_w),
    lines = o3d.utility.Vector2iVector(lines)
  )
  line_set.colors = o3d.utility.Vector3dVector(colors)

  return [axis, plane, line_set]