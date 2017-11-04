import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import save_model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adadelta, Adam

from os.path import join
from os import listdir
from skimage.color import grey2rgb
from skimage.io import imread
from skimage.util import pad
from skimage import transform

from random import choice


IMAGE_HEIGHT = IMAGE_WIDTH = 100
AS_GREY = False


def scale_image(img, points_vec):
    """

    :type img: np.ndarray
    :type points_vec: np.ndarray
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    assert(img.shape[0] == img.shape[1])
    assert(points_vec.shape[0] == 2)

    shape = img.shape[:2]
    factor = IMAGE_HEIGHT / shape[0]

    new_img = transform.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='symmetric')
    new_points_vec = points_vec * factor

    return new_img, new_points_vec, factor


def pad_image(img, points_vec):
    r = img.shape[0]
    c = img.shape[1]

    row_pad = max(0, c-r)
    col_pad = max(0, r-c)

    if row_pad == col_pad == 0:
        return img, points_vec, (0, 0)

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

    return new_img, new_points_vec, (col_pad_bottom, row_pad_left)


def get_dummy_y(img_dir):
    return {img_name: np.zeros(28) for (i, img_name) in enumerate(listdir(img_dir))}


def pad_and_scale(_img, _img_points):
    img_points = np.array([_img_points[::2], _img_points[1::2]])

    img, points_vec, pad_movement = pad_image(_img, img_points)
    img, points_vec, scale_factor = scale_image(img, points_vec)

    return img, points_vec, (pad_movement, scale_factor)


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


def rotate_img(_img, _points_vec):
    angle = 20 * np.random.random() - 10
    img = transform.rotate(_img, angle=angle)
    points_vec = get_rotation_matrix(angle).dot(_points_vec - _img.shape[0]/2 + 0.5) + _img.shape[0]/2 - 0.5

    return img, points_vec


def frame_image(img, points_vec):
    min_x, max_x = np.floor(np.min(points_vec[0])), np.ceil(np.max(points_vec[0]))
    min_y, max_y = np.floor(np.min(points_vec[1])), np.ceil(np.max(points_vec[1]))

    low = min(20, np.random.randint(0, min_y)) if min_y > 0 else 0
    top = max(img.shape[0] - 20, np.random.randint(max_y + 1, img.shape[0])) \
        if max_y + 1 < img.shape[0] else img.shape[0]
    left = min(20, np.random.randint(0, min_x)) if min_x > 0 else 0
    right = max(img.shape[1] - 20, np.random.randint(max_x + 1, img.shape[1])) \
        if max_x + 1 < img.shape[0] else img.shape[1]

    new_img = img[low:top, left:right]
    new_points_vec = points_vec.copy()
    new_points_vec[0] -= left
    new_points_vec[1] -= low

    return new_img, new_points_vec


def get_points_list(points_vec: np.ndarray):
    res = []

    for x, y in zip(points_vec[0], points_vec[1]):
        res.append(x)
        res.append(y)

    return res


PERMUTATIONS = [
    lambda img, vec: pad_and_scale(img, vec)[:2],
    lambda img, vec: flip_img(*pad_and_scale(img, vec)[:2]),
    lambda img, vec: rotate_img(*pad_and_scale(img, vec)[:2]),
    # lambda img, vec: pad_and_scale(*frame_image(img, vec))[:2]
]


def choice_permutation(img, y):
    perm = choice(PERMUTATIONS)
    return perm(img, y)


def read_generator(y_csv, train_img_dir, batch_size, permutations=False, shuffle=True, mutations={}, grey=True):
    channels = 1 if grey else 3
    batch_features = np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, channels))
    batch_labels = np.zeros((batch_size, 28))
    names = sorted(list(y_csv.keys()))

    while True:
        if shuffle:
            np.random.shuffle(names)

        for ind, name in enumerate(names):
            i = ind % batch_size

            img = imread(join(train_img_dir, name), as_grey=grey)

            if len(img.shape) == 2 and not grey:
                img = grey2rgb(img)

            if permutations:
                img, y = choice_permutation(img, y_csv[name])
            else:
                img, y, _mutations = pad_and_scale(img, y_csv[name])
                mutations[ind] = _mutations

            batch_features[i, ...] = img.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, channels))
            batch_labels[i] = np.ravel(y, order='F')

            if ind % batch_size == batch_size - 1 or ind == len(names) - 1:
                yield batch_features, batch_labels


def train_fast_detector(y_csv, train_img_dir, batch_size=30):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(28))

    model.compile(loss='mean_squared_error', optimizer=Adadelta(lr=0.1))

    model.fit_generator(
        read_generator(y_csv, train_img_dir, batch_size, grey=False),
        epochs=1,
        steps_per_epoch=len(y_csv) // batch_size
    )


def train_hard_detector(y, train_img_dir, fast=False):
    loss_method = 'mean_squared_error'
    as_grey = AS_GREY

    model = Sequential()

    model.add(
        Conv2D(32, (3, 3), input_shape=(100, 100, 1 if as_grey else 3), activation='relu', data_format='channels_last')
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(28))

    model.compile(loss=loss_method, optimizer=Adam())

    checkpoint_callback = ModelCheckpoint(filepath='checkpoint.hdf5', monitor='val_loss', save_best_only=True,
                                          mode='auto')
    early_callback = EarlyStopping(patience=15)
    lr_callback = ReduceLROnPlateau(patience=5)

    h = model.fit_generator(
        read_generator(y, train_img_dir, 30, grey=as_grey),
        steps_per_epoch=len(y) // 30,
        epochs=300 if not fast else 1,
        callbacks=[checkpoint_callback, early_callback, lr_callback],
    )

    save_model(model, 'facepoints_model.hdf5')


def back_mutation(pred, mutations):
    for i, p in enumerate(pred):
        p /= mutations[i][1]  # делим на фактор, на который умножали
        p[::2] -= mutations[i][0][0]  # отнимаем колонки паддинга
        p[1::2] -= mutations[i][0][1]  # отнимаем строки паддинга


def train_detector(y, train_img_dir, fast_train=False):
    """
    :param y:
    :type y: dict[str, list[float]]
    :param train_img_dir:
    :type train_img_dir: str
    :param fast_train:
    :type fast_train: bool
    :return:
    """

    if fast_train:
        _y = y
        return train_fast_detector(_y, train_img_dir)
    else:
        train_hard_detector(y, train_img_dir)


def detect(model, test_img_dir):
    """
    :param model:
    :type model: keras.engine.Model
    :param test_img_dir:
    :type test_img_dir: str
    :return:
    """
    dy = get_dummy_y(test_img_dir)
    mutations = {}

    pred = model.predict_generator(
        read_generator(
            dy,
            test_img_dir,
            30,
            permutations=False,
            shuffle=False,
            mutations=mutations,
            grey=False
        ),
        len(dy) // 30
    )

    back_mutation(pred, mutations)

    return {name: pr for pr, name in zip(pred, dy)}
