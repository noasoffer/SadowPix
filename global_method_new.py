import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def show_image(image):
    image = np.clip(image, 0, 1)
    if image.shape[0] == 1:
        image.shape = image.shape[1:]
    if len(image.shape) < 3:
        image = np.expand_dims(image, 2)
        image = np.repeat(image, 3, 2)
    plt.imshow(image)
    plt.show(block=True)


def gradient_convolution(photos,kernel_size=3):
    if type(photos) != np.ndarray or len(photos.shape) > 2:
        tmp =[]
        for photo in photos:
           tmp.append(cv2.Sobel(photo, cv2.CV_64F, 1, 0, ksize=kernel_size)) 
        g_x = np.array(tmp)
        tmp =[]
        for photo in photos:
           tmp.append(cv2.Sobel(photo, cv2.CV_64F, 0, 1, ksize=kernel_size)) 
        g_y = np.array(tmp)
    else:
        g_x = cv2.Sobel(photos, cv2.CV_64F, 1, 0, ksize=kernel_size)
        g_y = cv2.Sobel(photos, cv2.CV_64F, 0, 1, ksize=kernel_size)

    return np.sqrt(np.power(g_x, 2) + np.power(g_y, 2))


def L_and_p_convolution(photos,kernel_size=3):
    if type(photos) != np.ndarray or len(photos.shape) > 2:
        tmp = []
        for photo in photos:
            tmp.append(cv2.blur(photo, (kernel_size, kernel_size)))
        return np.array(tmp)
    return cv2.blur(photos, (kernel_size, kernel_size))


def mse(a, b):
    res = ((a - b) ** 2)
    return res

def create_four_points_vertical(point1, point3):
    point2 = [point1[0], point1[1], point3[2]]
    point4 = [point3[0], point3[1], point1[2]]
    return [point1, point2, point3, point4]


def create_four_points_horizontal(point1, point3):
    point2 = [point3[0], point1[1], point3[2]]
    point4 = [point1[0], point3[1], point3[2]]
    return [point1, point2, point3, point4]


def create_square_photo(photo, size):
    picture = Image.open(photo).convert('L')  # opens and converts to grayscale
    if picture.width != picture.height:
        #TODO check if neccessry
        pass
    picture = picture.resize((size, size), Image.ANTIALIAS)
    return np.array(picture) / 255


class GlobalMesh:
    def __init__(self, input_pics, product_size=200, heightfield=1,
                 #TODO change light angle
                 light_angle=60, W_G=1.5, W_S=0.001, radius=10, iterations=1000):

        self.photos = input_pics
        #TODO change this
        self.filter_images = gradient_convolution(L_and_p_convolution(self.photos))
        self.heightfield = heightfield
        self.product_size = product_size
        self.grid_size = int(product_size // self.heightfield)
        self.light_angle = light_angle
        self.radius = radius
        self.W_G = W_G
        self.W_S = W_S
        self.iterations = iterations
        self.calc_initialize_values()
        self.mesh_initialization()

    def calc_initialize_values(self):
        self.height = np.zeros([self.grid_size, self.grid_size])
        self.idx_cost = np.zeros(self.height.size)
        self.S = 1 / np.tan(self.light_angle * (np.pi / 180))
        self.T = 1
        self.alpha = self.T / self.iterations
        self.calculate_l = Calculate_L(self.radius, self.grid_size, len(self.photos)).calculate_next_L
        self.L = Calculate_L( self.radius, self.grid_size, len(self.photos)).calculate_L_total(self.height)
        self.pos_radius = np.arange(1, self.radius + 2)
        self.neg_radius = self.pos_radius[::-1]
        self.objective_value = self.get_objective_value(self.L)

    def mesh_initialization(self):
        self.vert = [None]
        self.faces = []
        points = [[0, 0, 0], [0, self.product_size, 0], [self.product_size, self.product_size, 0],
                  [self.product_size, 0, 0]]
        n_verts = len(self.vert)
        self.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
        self.vert.extend(points)


    def optimize(self):
        fails1 = 0
        fails2 = 0
        success = 0
        success_rand = 0
        convergence_fail = 0
    #TODO this loop is for the printing ---need to see if we want it -- I didn't change
        for i in range(self.iterations):
            if i % 1000 == 0:
                #TODO  change printing format once we understand what it means
                print(
                    f'{i * 100 / self.iterations}% success:{success * 100 / (i + 1)}%,success_rand:{success_rand * 100 / (i + 1)}%, fail1:{fails1 * 100 / (i + 1)}%,fail2:{fails2 * 100 / (i + 1)}% obj_value:{self.objective_value}')
            status, delta_obj = self.iteration()
            if status > 0:
                convergence_fail = 0
                if delta_obj == 1:
                    success_rand += 1
                success += 1
            elif status == -1:
                convergence_fail += 1
                fails1 += 1
            else:
                convergence_fail += 1
                fails2 += 1
            if convergence_fail == 100:
                print(f"optimizing failed after {i} steps, obj value={self.objective_value}")
                break

    def iteration(self):
        delta = 0
        while 0 == delta:
            #TODO ask yariv why hey chose this range
            delta = np.random.randint(-5, 6)
        idx = np.random.choice(self.height.size, 1, p=self.idx_cost)[0]
        row = idx // self.grid_size
        col = idx % self.grid_size
        self.height[row, col] += delta
        if self.legal_iteration(delta,row, col):
            next_l = self.calculate_l(self.height, self.L, row, col)
            new_objective = self.get_objective_value(next_l)
            objective_diff = self.check_objective_diff(new_objective)
            if objective_diff > 0:
                self.L = next_l
                self.T -= self.alpha
                self.objective_value = new_objective
                return new_objective, objective_diff
            else:
                self.height[row, col] -= delta
                return -2, None
        else:
            self.height[row, col] -= delta
            return -1, None


    def check_objective_diff (self, new_objective):
        objective_diff = self.objective_value - new_objective
        if objective_diff > 0:
            return objective_diff
        else:
            if np.random.random() < np.e ** (objective_diff / self.T):
                return 1
            else:
                return -2

    
    def get_objective_value(self, L): 
        g_convolution_p_convolution_l = gradient_convolution(L_and_p_convolution(L))
        h_convolution_g = gradient_convolution(self.height)
        l1 = mse(L_and_p_convolution(L), self.photos)
        l2 = self.W_G * mse(g_convolution_p_convolution_l, self.filter_images)
        l3 = self.W_S * mse(h_convolution_g, np.zeros(h_convolution_g.shape))
        l3 = l3[np.newaxis, :]
        loss = np.concatenate([l1, l2, l3])
        loss = np.sum(loss, axis=0).reshape(self.height.size)
        self.idx_cost = loss / loss.sum()
        return sum([l1.sum(),l2.sum(), l3.sum()])

    def legal_iteration(self, delta, row, col):
        if delta > 0:
            for photo_index in range(len(self.photos)):
                if photo_index == 0 and col > self.radius: 
                    new_value = self.height[row, col - self.radius - 1]
                    potential_shadow = self.height[row, col - self.radius:col + 1] + self.pos_radius
                elif photo_index == 1 and col < self.grid_size - self.radius - 1:
                    new_value = self.height[row, col + self.radius + 1]
                    potential_shadow = self.height[row, col:col + self.radius + 1] + self.neg_radius
                elif photo_index == 2 and row > self.radius:
                    new_value = self.height[row - self.radius - 1, col]
                    potential_shadow = self.height[row - self.radius:row + 1, col] + self.pos_radius
                elif photo_index == 3 and row < self.grid_size - self.radius - 1:
                    new_value = self.height[row + self.radius + 1, col]
                    potential_shadow = self.height[row:row + self.radius + 1, col] + self.neg_radius
                else:
                    continue
                if potential_shadow.max() < new_value:
                    return False
            return True
        else:
            new_value = self.height[row, col]
            for photo_index in range(len(self.photos)):
                if photo_index == 0 and col > self.radius:
                    potential_shadow = self.height[row, col - self.radius - 1:col] + self.neg_radius
                elif photo_index == 1 and col < self.grid_size - self.radius - 1:
                    potential_shadow = self.height[row, col + 1:col + self.radius + 2] + self.pos_radius
                elif photo_index == 2 and row > self.radius:
                    potential_shadow = self.height[row - self.radius - 1:row, col] + self.neg_radius
                elif photo_index == 3 and row < self.grid_size - self.radius - 1:
                    potential_shadow = self.height[row + 1:row + self.radius + 2, col] + self.pos_radius
                else:
                    continue
                if potential_shadow.max() < new_value:
                    return False
            return True

    def create_obj(self):
        floor = self.height.min()
        self.height -= floor
        self.height *= self.S
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.height[i, j] != 0:
                    create_h_mesh(self, i, j, self.height[i, j])

class Calculate_L:
    def __init__(self, radius, grid_size, num_of_photos):
        self.grid_size = grid_size
        self.radius = radius
        self.compare_idx = np.arange(1, radius + 1)
        self.current_matrix = np.arange(0, grid_size).reshape([grid_size, 1]) + self.compare_idx
        self.num_of_angles = num_of_photos

    def calculate_L_total(self, height):
        next_l = np.ones([self.num_of_angles, self.grid_size, self.grid_size])
        for row in range(0, self.grid_size):
            next_l = self.calculate_next_L(height, next_l, row, row)
        return next_l

    def calculate_next_L(self, height, next_l, row, col):
        next_l = next_l.copy()
        current_vector = None
        for direction in range(self.num_of_angles):
            if direction == 0:
                current_vector = height[row, :]
            elif direction == 1:
                current_vector = height[row, ::-1]
            elif direction == 2:
                current_vector = height[:, col]
            elif direction == 3:
                current_vector = height[::-1, col]
            vector_radius = self.add_radius(current_vector)
            tmp_matrix = (vector_radius[self.current_matrix] - self.compare_idx).max(axis=1)
            l_update = np.clip((current_vector - tmp_matrix), 0, 1)
            if direction == 0:
                next_l[direction, row, :] = l_update
            if direction == 1:
                l_update = l_update[::-1]
                next_l[direction, row, :] = l_update
            if direction == 2:
                next_l[direction, :, col] = l_update
            if direction == 3:
                l_update = l_update[::-1]
                next_l[direction, :, col] = l_update                
        return next_l

    def add_radius(self, vector):
        joined = np.ones(self.grid_size + self.radius) * (-2000)
        joined[:vector.shape[0]] = vector
        return  joined

def create_h_mesh(mesh, i, j, param):
    # creates 5 parts of a wall block
    lwall = create_four_points_vertical([i * mesh.heightfield, j * mesh.heightfield, 0],
                                                [i * mesh.heightfield, (j + 1) * mesh.heightfield,
                                                param])
    n_verts = len(mesh.vert)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.vert.extend(lwall)
    rwall = create_four_points_vertical([(i + 1) * mesh.heightfield, j * mesh.heightfield, 0],
                                                [(i + 1) * mesh.heightfield, (j + 1) * mesh.heightfield,
                                                param])
    n_verts = len(mesh.vert)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.vert.extend(rwall)
    upwall =create_four_points_vertical([i * mesh.heightfield, (j + 1) * mesh.heightfield, 0],
                                                [(i + 1) * mesh.heightfield, (j + 1) * mesh.heightfield,
                                                    param])
    n_verts = len(mesh.vert)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.vert.extend(upwall)
    dwnwall = create_four_points_vertical([i * mesh.heightfield, j * mesh.heightfield, 0],
                                                    [(i + 1) * mesh.heightfield, j * mesh.heightfield,
                                                    param])
    n_verts = len(mesh.vert)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.vert.extend(dwnwall)
    topwall = create_four_points_horizontal([i * mesh.heightfield, j * mesh.heightfield, param],
                                                    [(i + 1) * mesh.heightfield,
                                                    (j + 1) * mesh.heightfield,
                                                    param])
    n_verts = len(mesh.vert)
    mesh.faces.extend([[n_verts, n_verts + 1, n_verts + 2], [n_verts, n_verts + 2, n_verts + 3]])
    mesh.vert.extend(topwall)




def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        default='mesh_global.obj',
                        type=str)
    parser.add_argument('--product_size',
                        default=200, type=int)
    #TODO change default
    parser.add_argument("-i", "--iterations",
                        default=2 * 10 ** 6, type=int)
    parser.add_argument("-g", "--w_gradient",
                        default=1.5, type=float )
    parser.add_argument("-s", "--w_smooth",
                        default=0.001, type=float )
    #TODO change default
    parser.add_argument("--heightfield",
                        default=1, type=int)
    #TODO change default
    parser.add_argument("-l", "--light_angle",
                        default=60, type=int )
    #TODO change default pic
    parser.add_argument('-p', '--photos', nargs='*',
                        default=["photos/pic_a.jpg",
                                 "photos/pic_b.jpg",
                                 "photos/pic_c.jpg",
                                 "photos/pic_d.jpg"])
    return parser.parse_args()

if __name__ == '__main__':
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    args = parse_args()

    # Run
    resized_photos = []
    for photo in args.photos:
        tmp_photo =  create_square_photo(photo, int(args.product_size))
        resized_photos.append(tmp_photo)

    global_mesh = GlobalMesh(input_pics=resized_photos,
                            product_size=args.product_size,
                            iterations=args.iterations,
                            heightfield=args.heightfield,
                            light_angle=args.light_angle,
                            W_G=args.w_gradient,
                            W_S=args.w_smooth)
    print("Starting global method")

    global_mesh.optimize()
    global_mesh.create_obj()

    with open(args.output, 'w+') as output_file:
        for v in global_mesh.vert:
            if v is None:
                continue
            output_file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in global_mesh.faces:
            output_file.write(f"f {face[0]} {face[1]} {face[2]}\n")
