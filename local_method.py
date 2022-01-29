import numpy as np
import argparse
from PIL import Image

def create_four_points_vertical(point1, point3):
    point2 = [point1[0], point1[1], point3[2]]
    point4 = [point3[0], point3[1], point1[2]]
    return [point1, point2, point3, point4]


def create_four_points_horizontal(point1, point3):
    point2 = [point3[0], point1[1], point3[2]]
    point4 = [point1[0], point3[1], point3[2]]
    return [point1, point2, point3, point4]

def create_square_photo(photo, size):
    picture = Image.open(photo).convert('L')
    picture = picture.resize((size, size), Image.ANTIALIAS)
    return np.array(picture) / 255


class LocalMesh:
    def __init__(self, photos, product_size=200, receiver_dimensions=1, wall_thickness=0.5):
        assert (len(photos) == 3)
        
        self.photos =[]
        for photo in photos: 
            tmp_photo = (1 - photo)
            self.photos.append(tmp_photo)
        
        self.receiver_dimensions = receiver_dimensions
        self.wall_thickness = wall_thickness
        self.sqr_size = self.receiver_dimensions + self.wall_thickness
        self.grid = int(product_size / self.sqr_size)
        self.product_size = self.grid * self.sqr_size + self.wall_thickness
        self.initialize_values()
       
       
    def initialize_values(self):
        self.light_angle = 45
        angle = self.light_angle * (np.pi / 180)
        self.S = np.cos(angle) / np.sin(angle)
        self.u = np.zeros([self.grid, self.grid + 1])
        self.v, self.r = None, None      
        self.chamfer_p = np.zeros([self.grid, self.grid])
        self.chamfer_m = np.zeros([self.grid, self.grid])
        self.verts = [None]
        self.faces = []
        points = [[0, 0, 0], [0, self.product_size, 0], [self.product_size, self.product_size, 0],
                  [self.product_size, 0, 0]]
        n_verts = len(self.verts)
        self.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
        self.verts.extend(points)


    def constrains(self):
        d_sum = 0
        constrain_3 = (-self.photos[0][:self.grid - 1, :] + self.photos[0][1:, :] - self.photos[2][1:, :]) * self.S
        for i in range(self.grid):
            self.u[:, i + 1] = self.u[:, i] + self.S * (self.photos[1][:, i] - self.photos[0][:, i])
        self.u += (self.S * self.photos[0][:, 0])[:, np.newaxis]

        for j in range(self.grid - 1):
            eq3_j_constrain = -self.u[j + 1, :-1] + self.u[j, :-1] + constrain_3[j, :]
            self.u[j + 1, :] += max(np.max(eq3_j_constrain), 0)
        self.r = self.u[:, :self.grid] - self.S * self.photos[0]
        

        for j in range(self.grid):
            for i in range(self.grid - 1):
                if self.r[j, i] > self.r[j, i + 1]:
                    d = min(self.receiver_dimensions - (self.chamfer_p[j, i] / self.S), self.r[j, i] - self.r[j, i + 1], self.u[j, i] - self.r[j, i]
                                )
                    if d > 0:
                        self.d_fix(j, d, i + 1)
                    else:
                        d = 0
                    d_sum += d
                    self.d_fix(j, d, i + 1)
                    self.chamfer_p[j, i + 1] = 0
                    self.chamfer_m[j, i] = d
                elif self.r[j, i] < self.r[j, i + 1]:
                    d = min(self.u[j, i + 2] - self.r[j, i + 1], self.r[j, i + 1] - self.r[j, i])
                    if d > 0:
                        self.u[j, i + 1] -= d
                        self.d_fix(j, -d, i + 1)
                    else:
                        d = 0
                    self.chamfer_m[j, i] = 0
                    self.chamfer_p[j, i + 1] = d
                    d_sum -= d
        self.calc_h()


    def calc_h(self):
        self.v = self.photos[2] * self.S + self.r
        min_h = min(np.min(self.r), np.min(self.v), np.min(self.u))
        self.u -= min_h
        self.r -= min_h
        self.v -= min_h
        self.h = max(np.max(self.r), np.max(self.v), np.max(self.u))

    def d_fix(self, i, fix, begain):
        for j in range(begain, self.grid):
            self.r[i, j] += fix
            self.u[i, j + 1] += fix


def create_wall_mesh(mesh, i, j, param):
    left_wall = create_four_points_vertical([j * mesh.sqr_size, i * mesh.sqr_size, 0],
                                                [(j + 1) * mesh.sqr_size, i * mesh.sqr_size, param])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(left_wall)
    right_wall = create_four_points_vertical([j * mesh.sqr_size, i * mesh.sqr_size + mesh.wall_thickness, 0],
                                                [(j + 1) * mesh.sqr_size, i * mesh.sqr_size + mesh.wall_thickness,
                                                param])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(right_wall)
    up_wall = create_four_points_vertical([(j + 1) * mesh.sqr_size, i * mesh.sqr_size, 0],
                                                [(j + 1) * mesh.sqr_size, i * mesh.sqr_size + mesh.wall_thickness,
                                                    param])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(up_wall)
    down_wall = create_four_points_vertical([j * mesh.sqr_size, i * mesh.sqr_size, 0],
                                                    [j * mesh.sqr_size, i * mesh.sqr_size + mesh.wall_thickness,
                                                    param])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(down_wall)
    topwall = create_four_points_horizontal([j * mesh.sqr_size, i * mesh.sqr_size, param],
                                                    [(j + 1) * mesh.sqr_size, i * mesh.sqr_size + mesh.wall_thickness,
                                                    param])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(topwall)

def create_receiver_mesh(mesh, i, j, param):
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(create_four_points_horizontal([j * mesh.sqr_size, mesh.wall_thickness + i * mesh.sqr_size, param],[(j + 1) * mesh.sqr_size - mesh.wall_thickness,(i + 1) * mesh.sqr_size, param]))

def create_chamfer(mesh, i, j, chamfer_p_param, chamfer_m_param ):
    chamferx_dist_p = chamfer_p_param / mesh.S
    chamferx_dist_m = chamfer_m_param / mesh.S
    points_p = []
    points_m = []

    points_p.append([j * mesh.sqr_size + mesh.wall_thickness, (i + 1) * mesh.sqr_size - chamferx_dist_p, mesh.r[j, i]])
    points_p.append([j * mesh.sqr_size + mesh.wall_thickness, (i + 1) * mesh.sqr_size, mesh.r[j, i] + chamfer_p_param])
    points_p.append([(j + 1) * mesh.sqr_size, (i + 1) * mesh.sqr_size, mesh.r[j, i] + chamfer_p_param])
    points_p.append([(j + 1) * mesh.sqr_size, (i + 1) * mesh.sqr_size - chamferx_dist_p, mesh.r[j, i]])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(points_p)

    points_m.append([j * mesh.sqr_size + mesh.wall_thickness, i * mesh.sqr_size + mesh.wall_thickness + chamferx_dist_m, mesh.r[j, i]])
    points_m.append([j * mesh.sqr_size + mesh.wall_thickness, i * mesh.sqr_size + mesh.wall_thickness, mesh.r[j, i] + chamfer_m_param])
    points_m.append([(j + 1) * mesh.sqr_size, i * mesh.sqr_size + mesh.wall_thickness, mesh.r[j, i] + chamfer_m_param])
    points_m.append([(j + 1) * mesh.sqr_size, i * mesh.sqr_size + mesh.wall_thickness + chamferx_dist_m, mesh.r[j, i]])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(points_m)


def create_vwall_mesh(mesh, i, j, param):
    # creates 5 parts of a wall block
    left_wall = create_four_points_vertical([j * mesh.sqr_size, i * mesh.sqr_size, 0],
                                                [j * mesh.sqr_size + mesh.wall_thickness, i * mesh.sqr_size, param])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(left_wall)

    right_wall = create_four_points_vertical([j * mesh.sqr_size, (i + 1) * mesh.sqr_size, 0],
                                                [j * mesh.sqr_size + mesh.wall_thickness, (i + 1) * mesh.sqr_size,
                                                param])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(right_wall)

    up_wall = create_four_points_vertical([j * mesh.sqr_size + mesh.wall_thickness, i * mesh.sqr_size, 0],
                                                [j * mesh.sqr_size + mesh.wall_thickness, (i + 1) * mesh.sqr_size,
                                                    param])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(up_wall)

    down_wall = create_four_points_vertical([j * mesh.sqr_size, i * mesh.sqr_size, 0],
                                                    [j * mesh.sqr_size, (i + 1) * mesh.sqr_size, param])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(down_wall)

    topwall = create_four_points_horizontal([j * mesh.sqr_size, i * mesh.sqr_size, param],
                                                    [j * mesh.sqr_size + mesh.wall_thickness, (i + 1) * mesh.sqr_size,
                                                    param])
    n_verts = len(mesh.verts)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.verts.extend(topwall)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        default='mesh_local.obj',
                        type=str)
    parser.add_argument('--wall_thickness',
                        default=0.5, type=float)
    parser.add_argument('--pixel_dimensions',
                        default=1, type=float)
    parser.add_argument('--product_size',
                        default=200, type=int)
    parser.add_argument('-p', '--photos', nargs='*',
                        default=["photos/frida.jpeg","photos/salvador_dali.jpeg","photos/adel.jpeg"])
    return parser.parse_args()


if __name__ == '__main__':
    import os
    import sys
    

    args = parse_args()

    resized_photos = []
    for photo in args.photos:
        tmp_photo =  create_square_photo(photo, int(args.product_size / (args.wall_thickness + args.pixel_dimensions)))
        resized_photos.append(tmp_photo)

    local_mesh = LocalMesh(photos=resized_photos,                     
                          product_size=args.product_size,
                          receiver_dimensions=args.pixel_dimensions,
                          wall_thickness=args.wall_thickness)
    local_mesh.constrains()
    
    for i in range(local_mesh.grid + 1):
        for j in range(local_mesh.grid):
            create_wall_mesh(local_mesh, i, j, local_mesh.u[j, i])
            if local_mesh.grid != i:
                create_receiver_mesh(local_mesh,i, j, local_mesh.r[j, i])
                create_chamfer(local_mesh, i, j, local_mesh.chamfer_p[j, i], local_mesh.chamfer_m[j, i])
                create_vwall_mesh(local_mesh, i, j, local_mesh.v[j, i])

    with open(args.output, 'w+') as output_file:
        for v in local_mesh.verts:
            if v is None:
                continue
            output_file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in local_mesh.faces:
            output_file.write(f"f {face[0]} {face[1]} {face[2]}\n")            
    print(f"Mesh saved")
    output_file.close()
