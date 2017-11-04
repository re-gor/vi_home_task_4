import numpy as np
from skimage import transform
from skimage.io import imread, imsave
from skimage.util import pad, img_as_float, img_as_uint
from os import listdir, mkdir
from os.path import isfile, join, basename, abspath
from tqdm import tqdm

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
TEMP_DIR_NAME = '__temp'


def bright_points(img, points_vec):
    # img[..., 0 ][points_vec[1].astype(np.int32), points_vec[0].astype(np.int32)] = 1
    img_as_uint(img)
    img[..., :][points_vec[1].astype(np.int32), points_vec[0].astype(np.int32)] = 1


def get_points_list(points_vec: np.ndarray):
    res = []

    for x, y in zip(points_vec[0], points_vec[1]):
        res.append(x)
        res.append(y)

    return res


_FLIP_MAP = [
    (0, 3), (1, 2),  # брови
    (4, 9), (5, 8), (6, 7),  # глаза
    (11, 13)  # губы
]
FLIP_MAP = [[uno for uno, _ in _FLIP_MAP], [duo for _, duo in _FLIP_MAP]]


def flip_img(img, points_vec):
    assert(img.shape[0] == img.shape[1])
    assert(points_vec.shape[0] == 2)

    new_points_vec = points_vec.copy()

    new_points_vec[0] = img.shape[0] - points_vec[0]

    new_points_vec[:, FLIP_MAP[0]], new_points_vec[:, FLIP_MAP[1]] = \
        new_points_vec[:, FLIP_MAP[1]], new_points_vec[:, FLIP_MAP[0]]

    return img[:, ::-1].copy(), new_points_vec


def get_rotation_matrix(angle):
    _angle = np.radians(angle)
    cos = np.cos(_angle)
    sin = np.sin(_angle)

    return np.array([
        [cos, sin],
        [-sin, cos]  # Система координат левосторонняя
    ])


# angles = np.arange(-10, 11, 5)
# rotation_matrixes = [get_rotation_matrix(angle) for angle in angles]

def rotate_img(_img, _points_vec):
    angle = 20 * np.random.random() - 10
    img = transform.rotate(_img, angle=angle)
    points_vec = get_rotation_matrix(angle).dot(_points_vec - _img.shape[0]/2 + 0.5) + _img.shape[0]/2 - 0.5

    return img, points_vec


def rotate_imgs(img, points_vec, count=5):
    assert(img.shape[0] == img.shape[1])
    assert(points_vec.shape[0] == 2)
    angles = 20 * np.random.random(count) - 10

    new_imgs = []
    new_points_vecs = []

    for i, angle in enumerate(angles):
        if angle == 0:
            continue

        new_imgs.append(transform.rotate(img, angle=angle))
        new_points_vecs.append(
            get_rotation_matrix(angle).dot(points_vec - img.shape[0]/2 + 0.5) + img.shape[0]/2 - 0.5
        )

        assert(np.max(new_points_vecs) <= 100)

    return new_imgs, new_points_vecs


def scale_image(img, points_vec):
    assert(img.shape[0] == img.shape[1])
    assert(points_vec.shape[0] == 2)

    shape = img.shape[:2]
    factor = IMAGE_HEIGHT / shape[0]

    new_img = transform.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='symmetric')
    new_points_vec = points_vec * factor

    return new_img, new_points_vec


def pad_image(img, points_vec):
    r = img.shape[0]
    c = img.shape[1]

    row_pad = max(0, c-r)
    col_pad = max(0, r-c)

    if row_pad == col_pad == 0:
        return img, points_vec

    row_pad_left = row_pad // 2
    row_pad_right = row_pad_left + row_pad % 2

    col_pad_top = col_pad // 2
    col_pad_bottom = col_pad_top + col_pad % 2

    padding = [(row_pad_left, row_pad_right), (col_pad_top, col_pad_bottom), (0, 0)]

    if len(img.shape) < 3:
        padding = padding[:2]

    new_img = pad(img, padding, 'constant')
    new_points_vec = points_vec.copy()
    new_points_vec[0] += col_pad_bottom
    new_points_vec[1] += row_pad_left

    return new_img, new_points_vec


def frame_image(img, points_vec):
    min_x, max_x = np.floor(np.min(points_vec[0])), np.ceil(np.max(points_vec[0]))
    min_y, max_y = np.floor(np.min(points_vec[1])), np.ceil(np.max(points_vec[1]))

    low = np.random.randint(0, min_y) if min_y > 0 else 0
    top = np.random.randint(max_y + 1, img.shape[0]) if max_y + 1 < img.shape[0] else img.shape[0]
    left = np.random.randint(0, min_x) if min_x > 0 else 0
    right = np.random.randint(max_x + 1, img.shape[1]) if max_x + 1 < img.shape[0] else img.shape[1]

    new_img = img[low:top, left:right, ...]
    new_points_vec = points_vec.copy()
    new_points_vec[0] -= left
    new_points_vec[1] -= low

    return new_img, new_points_vec


def get_framed_images(img, points_vec):
    imgs, points_vec_list = [], []
    for i in range(5):
        new_img, new_points_vec = frame_image(img, points_vec)
        imgs.append(new_img)
        points_vec_list.append(new_points_vec)

    return imgs, points_vec_list


def create_temp_directory():
    if TEMP_DIR_NAME not in listdir('.'):
        mkdir(TEMP_DIR_NAME)


def save_img(index, img_name, img):
    imsave(join(TEMP_DIR_NAME, '_%d_' % index + img_name), img)
    return '_%d_' % index + img_name


def prepare_images(img_dir, points):
    """

    :type points: dict[str, list[float]]
    :param img_dir:
    :param points:
    :return:
    """
    create_temp_directory()

    new_points = {}

    i = 0
    for img_name, _img_points in tqdm(sorted(points.items()), total=len(points)):  # [('00294.jpg', points['00294.jpg'])]:
        i += 1
        # if i > 3:
        #     break

        index = 0
        img_path = join(img_dir, img_name)
        _img = imread(img_path)
        img_points = np.array([_img_points[::2], _img_points[1::2]])

        # нарезаем кадры
        # for framed_img, framed_points_vec in zip(*get_framed_images(_img, img_points)):
        #     try:
        #         # Оквадрачиваем изображение
        #         img, points_vec = pad_image(framed_img, framed_points_vec)
        #
        #         # Сжимаем изображение
        #         scaled_img, scaled_points = scale_image(img, points_vec)
        #         # s_copy, s_points = scaled_img.copy(), scaled_points.copy()
        #         # bright_points(s_copy, s_points)
        #         s_img_name = save_img(index, img_name, scaled_img)
        #
        #         new_points[s_img_name] = get_points_list(scaled_points)
        #         index += 1
        #
        #     except:
        #         print('Fuck', img_name)

        # Оквадрачиваем изображение
        img, points_vec = pad_image(_img, img_points)

        # Сжимаем изображение
        scaled_img, scaled_points = scale_image(img, points_vec)
        # s_copy, s_points = scaled_img.copy(), scaled_points.copy()
        # bright_points(s_copy, s_points)
        s_img_name = save_img(index, img_name, scaled_img)

        new_points[s_img_name] = get_points_list(scaled_points)
        index += 1

        # try:
        #     # Поворачиваем
        #     rotated_imgs, rotated_points = rotate_img(scaled_img, scaled_points)
        #     for rotated_img, r_points in zip(rotated_imgs, rotated_points):
        #         # bright_points(rotated_img, r_points)
        #         r_img_name = save_img(index, img_name, rotated_img)
        #
        #         new_points[r_img_name] = get_points_list(r_points)
        #         index += 1
        # except:
        #     print('rotate fuck', img_name)

        # Зеркалируем
        flipped_img, flipped_points = flip_img(scaled_img, scaled_points)
        # bright_points(flipped_img, flipped_points)
        f_img_name = save_img(index, img_name, flipped_img)

        new_points[f_img_name] = get_points_list(flipped_points)
        index += 1

    save_csv(new_points, '__temp.csv')


def pad_and_scale(_img, img_name):
    _img_points = csv[img_name]
    _img_points = np.array([_img_points[::2], _img_points[1::2]])

    img, points_vec = pad_image(_img, _img_points)

    return scale_image(img, points_vec)


def prepare_image_test(img_name, csv, img_dir='./public_data/00_input/test/images/'):
    img_path = join(img_dir, img_name)
    _img_points = csv[img_name]
    _img_points = np.array([_img_points[::2], _img_points[1::2]])
    _img = imread(img_path)

    img, points_vec = pad_image(_img, _img_points)
    return scale_image(img, points_vec)


def read_csv(filename):
    """

    :rtype: dict[str, list[float]]
    :param filename:
    :return:
    """
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = np.array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res


def save_csv(facepoints, filename):
    """

    :param facepoints:
    :type facepoints: dict[str, list[int]]
    :param filename:
    :type filename: str
    :return:
    """
    with open(filename, 'w') as fhandle:
        print('filename,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12,x13,y13,x14,y14',
                file=fhandle)
        for filename in sorted(facepoints.keys()):
            points_str = ','.join(map(str, facepoints[filename]))
            print('%s,%s' % (filename, points_str), file=fhandle)

if __name__ == '__main__':
    csv = read_csv('./public_data/00_input/train/gt.csv')

    prepare_images('./public_data/00_input/train/images', csv)
               # {'00046.jpg': [51, 51, 127, 60, 169, 65, 203, 52, 68, 77, 85, 75, 110, 82, 160, 87, 171, 79, 193, 84, 156, 144, 92, 174, 140, 180, 168, 179]})
