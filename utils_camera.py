import open3d as o3d
import numpy as np

class camara: 

  def calcular_matriz_extrinsecas(eye, center, up, fov, width, height):
      ## According to : http://ksimek.github.io/2012/08/22/extrinsic/ 
      # https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml
      # Matriz de vista extrínseca
      I = np.eye(4)
      view_matrix = np.stack([I] * len(eye))
      z_axis = (center - eye) / np.linalg.norm(center - eye)
      x_axis = np.cross(z_axis, up) / np.linalg.norm(np.cross(z_axis, up))
      y_axis = np.cross(x_axis, z_axis)
      view_matrix[:,:3,:3] = np.stack((x_axis, y_axis, -z_axis), axis=-1)
      view_matrix[:,:3,3:]= np.stack(np.reshape(eye, (len(view_matrix),3,1)), axis=0)

      return view_matrix

  def calcular_matriz_intrinseca(fov, ancho, alto, s=0):
      
      # Convertir el campo de visión (FOV) de grados a radianes
      fov_rad = np.deg2rad(fov)
      # Calcular la distancia focal utilizando la fórmula de la proyección en perspectiva
      # La distancia focal (f) se calcula como la mitad del tamaño del sensor dividido por la tangente del FOV dividido por 2
      distancia_focal_x = ancho / (2* np.tan(fov_rad / 2 ))
      distancia_focal_y = alto / (2* np.tan(fov_rad / 2))
      
      # Construir la matriz intrínseca
      matriz_intrinseca = np.array([
          [distancia_focal_x, s, ancho / 2],
          [0, distancia_focal_y, alto / 2],
          [0, 0, 1]
      ])
      
      return matriz_intrinseca

def save_camera_trayectory(direccion, eye, cent, up, fov, width, height):

  extrinsic = camara.calcular_matriz_extrinsecas(eye, cent, up, fov, width, height)
  intrinsic = camara.calcular_matriz_intrinseca(fov, width, height)
  tra = []
  camera_intrin = o3d.camera.PinholeCameraIntrinsic(width,height, intrinsic)
  trayectoria_camara = o3d.camera.PinholeCameraTrajectory()
  for pose in extrinsic:
      params = o3d.camera.PinholeCameraParameters()
      params.extrinsic = pose
      params.intrinsic = camera_intrin
      tra.append(params)
  trayectoria_camara.parameters = tra
  # Guardar la trayectoria de cámara en un archivo
  o3d.io.write_pinhole_camera_trajectory(direccion + "/trayectoria_camara.json", trayectoria_camara)
