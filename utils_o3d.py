import open3d as o3d
import numpy as np
import octomap
import trimesh

def Get_RGBD(render, fov, center, eye, up, direccion, i):
   '''render : es el objeto de escena que contiene el objeto
      fov : vertical_field of view
      center: camera center (position)
      eye: camera eye (orientation, where the camera see)
      up: camera up vector ()
      direccion: carpeta donde se almacena 
      i = indice de la imagen
      '''
   render.setup_camera(fov, center, eye, up)
   img = render.render_to_image()
   depth = render.render_to_depth_image()
   o3d.io.write_image(direccion +"/RGB/RGB_{}.png".format(i), img, 9)
   o3d.io.write_image(direccion + "/Depth/D_{}.png".format(i), depth, 9)


def Get_Pointcloud(scene, fov, center, eye, up, width, height, direccion, i):
  # (scene, fov, center, eye, up, width, height)
  rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    fov_deg= fov,
    center= center,
    eye= eye,
    up= up,
    width_px=width,
    height_px=height,
  )
  # We can directly pass the rays tensor to the cast_rays function.
  ans = scene.cast_rays(rays)
  hit = ans['t_hit'].isfinite()
  points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
  pcd = o3d.t.geometry.PointCloud()
  pcd.point["positions"] = o3d.core.Tensor(points.numpy())
  
  if i == 0:
    p_c = pcd
  else:
    p_c.point["positions"] = o3d.core.concatenate((p_c.point["positions"], pcd.point["positions"]), 0)

  o3d.t.io.write_point_cloud(direccion + "/Point_cloud/cloud_{}.pcd".format(i), p_c, write_ascii=True) #almacena nubes de puntos

def Get_octree(octree, direccion, i, origin):
  '''octree -> occupation prob [n,]
    '''
  p_c = o3d.io.read_point_cloud(direccion + "/Point_cloud/cloud_{}.pcd".format(i)) #filtrar Nan
  octree.insertPointCloud(
      pointcloud= p_c,#nube de puntos                         # Filtrar valores nan
      origin= origin,# origen de la foto
      maxrange=1.25, # maximum range for how long individual beams are inserted
      ) # se agrega el pointcloud y el origen del sensor
  
  # Se extrae el minimo y maximo de BBox
  aabb_min = octree.getMetricMin() 
  aabb_max = octree.getMetricMax()
  center = (aabb_min + aabb_max) / 2
  dimension = np.array([31, 31, 31]) # dimension de la voxelizacion
  origin = center - dimension / 2 * resolution

  #nuevos BBox  con la nueva resolución
  aabb_min = origin - resolution / 2
  aabb_max = origin + dimension * resolution + resolution / 2

  grid = np.full(dimension, -1, np.int32)
  transform = trimesh.transformations.scale_and_translate(
      scale=resolution, translate=origin
  ) #transformacion
  # Voxelgrid encoding (create grid) and probability allocation
  points = trimesh.voxel.VoxelGrid(encoding=grid, transform=transform).points # Voxel grid con los puntos de la nube
  labels = octree.getLabels(points) # extrae las etiquetas o valor ocupacional
  # according to octomap-python # -1: unknown, 0: empty, 1: occupied
  #according to our dataset # 0.5: unknown, 0: empty, 1: occupied
  labels = labels.astype(dtype = np.float32)
  np.place(labels, labels <0, [0.5]) # los valores menores a 0 pasan a ser iguales a 0.5
  #grid = np.reshape(labels,(31,31,31))
  return labels 