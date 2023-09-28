import numpy as np
import cv2
from skimage.draw import line_aa, ellipse_perimeter, line
from math import atan2
from skimage.transform import resize
from time import time
from scripts.filter_2 import filter2
from scripts.filter_4 import filter4
from scripts.filter_5 import filter5
import multiprocessing
import argparse

OUTPUT_FILE = "output.png"
SIDE_LEN = 500
EXPORT_STRENGTH = 0.07
PULL_AMOUNT = 5000
RANDOM_NAILS = None
RADIUS1_MULTIPLIER = 1
RADIUS2_MULTIPLIER = 1
NAILS_SIZE = 240

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def largest_square(image: np.ndarray) -> np.ndarray:
    size = min(image.shape[:2])
    image[:size, :size] = image[:size, :size]
    return image

def create_circle_nail_positions(shape, nail_count=240, r1_multip=1, r2_multip=1):
    height = shape[0]
    width = shape[1]

    centre = (height // 2, width // 2)
    radius = min(height, width) // 2 - 1

    theta = np.linspace(0, 2 * np.pi, nail_count, endpoint=False)
    rr = centre[0] + np.round(radius * r1_multip * np.cos(theta)).astype(int)
    cc = centre[1] + np.round(radius * r2_multip * np.sin(theta)).astype(int)

    nails = np.column_stack((rr, cc))
    # nails = nails.tolist()
    # nails.sort(key=lambda c: atan2(c[0] - centre[0], c[1] - centre[1]))
    # nails = np.asarray(nails)

    return nails


def init_canvas(shape, black=False):
    if black:
        return np.zeros(shape)
    else:
        return np.ones(shape)


def get_aa_line(from_pos, to_pos, str_strength, picture):
    rr, cc, val = line_aa(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
    lin = picture[rr, cc] + str_strength*val
    lin = np.clip(lin, a_min=0, a_max=1)

    return lin, rr, cc



# def find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength):
#     overlayed_lines = []
#     rr_list = []
#     cc_list = []

#     if RANDOM_NAILS is not None:
#         nail_ids = np.random.choice(len(nails), size=RANDOM_NAILS, replace=False)
#         nails_and_ids = nails[nail_ids]
#     else:
#         nails_and_ids = nails

#     for nail_position in nails_and_ids:
#         overlayed_line, rr, cc = get_aa_line(current_position, nail_position, str_strength, str_pic)
#         overlayed_lines.append(overlayed_line)
#         rr_list.append(rr)
#         cc_list.append(cc)

#     overlayed_lines = np.array(overlayed_lines)
#     rr_list = np.concatenate(rr_list)
#     cc_list = np.concatenate(cc_list)

#     before_overlayed_line_diff = np.abs(str_pic[rr_list, cc_list] - orig_pic[rr_list, cc_list]) ** 2
#     after_overlayed_line_diff = np.abs(overlayed_lines - orig_pic[rr_list, cc_list]) ** 2

#     cumulative_improvements = np.sum(before_overlayed_line_diff - after_overlayed_line_diff, axis=1)
#     best_idx = np.argmax(cumulative_improvements)

#     if cumulative_improvements[best_idx] > 0:
#         best_nail_position = nails_and_ids[best_idx]
#         best_nail_idx = best_idx
#     else:
#         best_nail_position = None
#         best_nail_idx = None

#     return best_nail_idx, best_nail_position, cumulative_improvements[best_idx]

def find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength, RANDOM_NAILS=None):
    overlayed_lines = []
    cumulative_improvements = []

    if RANDOM_NAILS is not None:
        nail_ids = np.random.choice(len(nails), size=RANDOM_NAILS, replace=False)
        nails_and_ids = nails[nail_ids]
    else:
        nails_and_ids = nails

    overlayed_lines, rr_list, cc_list = get_aa_lines(current_position, nails_and_ids, str_strength, str_pic)

    before_overlayed_line_diff = np.abs(str_pic[rr_list, cc_list] - orig_pic[rr_list, cc_list]) ** 2

    for overlayed_line in overlayed_lines:
        overlayed_line = overlayed_line.reshape(orig_pic[rr_list, cc_list].shape)
        after_overlayed_line_diff = np.abs(overlayed_line - orig_pic[rr_list, cc_list]) ** 2
        cumulative_improvement = np.sum(before_overlayed_line_diff - after_overlayed_line_diff)
        cumulative_improvements.append(cumulative_improvement)

    cumulative_improvements = np.array(cumulative_improvements)

    best_idx = np.argmax(cumulative_improvements)
    if cumulative_improvements[best_idx] > 0:
        best_nail_position = nails_and_ids[best_idx]
        best_nail_idx = best_idx
    else:
        best_nail_position = None
        best_nail_idx = None

    return best_nail_idx, best_nail_position, cumulative_improvements[best_idx]

def get_aa_lines(current_position, nails_and_ids, str_strength, str_pic):
    overlayed_lines = []
    rr_list = []
    cc_list = []

    for nail_position in nails_and_ids:
        overlayed_line, rr, cc = get_aa_line(current_position, nail_position, str_strength, str_pic)
        overlayed_lines.append(overlayed_line)
        rr_list.append(rr)
        cc_list.append(cc)

    return overlayed_lines, np.concatenate(rr_list), np.concatenate(cc_list)




def create_art(nails, orig_pic, str_pic, str_strength, i_limit=None):

    start = time()
    iter_times = []

    current_position = nails[0]
    pull_order = [0]

    i = 0
    fails = 0
    while True:

        i += 1

        if i % 500 == 0:
            print(f"Iteration {i}")

        # if i_limit is None:
        #     if fails >= 5:
        #         break
        # else:
        #     if i > i_limit:
        #         break
        if fails >= 3:
            break
        if i_limit is not None:
            if i > i_limit:
                break
            
        idx, best_nail_position, best_cumulative_improvement = find_best_nail_position(current_position, nails,
                                                                                       str_pic, orig_pic, str_strength)

        if best_cumulative_improvement <= 0:
            fails += 1
            continue

        pull_order.append(idx)
        best_overlayed_line, rr, cc = get_aa_line(current_position, best_nail_position, str_strength, str_pic)
        str_pic[rr, cc] = best_overlayed_line

        current_position = best_nail_position

    print(f"Time: {time() - start}")
    print(len(pull_order))
    return pull_order


def scale_nails(x_ratio, y_ratio, nails):
    scaled_nails = np.array(nails)  # Преобразуем список nails в массив NumPy
    scaled_nails[:, 0] = np.round(scaled_nails[:, 0] * y_ratio).astype(int)
    scaled_nails[:, 1] = np.round(scaled_nails[:, 1] * x_ratio).astype(int)
    return scaled_nails.tolist()


# def pull_order_to_array_bw(order, canvas, nails, strength):
#     for pull_start, pull_end in zip(order, order[1:]):  # pairwise iteration
#         rr, cc = line(nails[pull_start][0], nails[pull_start][1],
#                               nails[pull_end][0], nails[pull_end][1])
#         canvas[rr, cc] += strength

#     return np.clip(canvas, a_min=0, a_max=1)


def pull_order_to_array_bw(order, canvas, nails, strength):
    pull_start_points = np.array([nails[pull_start] for pull_start in order])
    pull_end_points = np.array([nails[pull_end] for pull_end in order[1:]])

    rr, cc = line(pull_start_points[:, 0].astype(int), pull_start_points[:, 1].astype(int),
                  pull_end_points[:, 0].astype(int), pull_end_points[:, 1].astype(int))
    
    canvas[rr, cc] += strength

    return np.clip(canvas, a_min=0, a_max=1)

def generate(img, output_file):

    LONG_SIDE = 250

    if np.any(img > 100):
        img = img / 255
    # cv2.imshow('frame', img); cv2.waitKey(0)
    if RADIUS1_MULTIPLIER == 1 and RADIUS2_MULTIPLIER == 1:
        img = largest_square(img)
        img = cv2.resize(img, (LONG_SIDE, LONG_SIDE))

    shape = (len(img), len(img[0]))

    nails = create_circle_nail_positions(shape, NAILS_SIZE, RADIUS1_MULTIPLIER, RADIUS2_MULTIPLIER)

    print(f"Nails amount: {len(nails)}")

    orig_pic = rgb2gray(img) * 0.55
    # cv2.imshow('frame', orig_pic); cv2.waitKey(0)

    image_dimens = int(SIDE_LEN * RADIUS1_MULTIPLIER), int(SIDE_LEN * RADIUS2_MULTIPLIER)
    
    str_pic = init_canvas(shape, black=False)
    pull_order = create_art(nails, orig_pic, str_pic, -0.03, i_limit=PULL_AMOUNT)
    blank = init_canvas(image_dimens, black=False)

    scaled_nails = scale_nails(
        image_dimens[1] / shape[1],
        image_dimens[0] / shape[0],
        nails
    )

    result = pull_order_to_array_bw(
        pull_order,
        blank,
        scaled_nails,
        -EXPORT_STRENGTH
    )
    cv2.imwrite(f'{output_file}.png', result * 255)

    with open(f"{output_file}.json", "w") as f:
    # with open("result.json", "w") as f:
        f.write(str(pull_order))
    print("done")

def generate_all(path, filename):

    input_file = path + filename
    img = cv2.imread(input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img1 = generate(img, path + 'result1')
    img2 = generate(filter2(img), path + 'result2')
    img3 = generate(filter4(img), path + 'result3')
    img4 = generate(filter5(img), path + 'result4')
    
    return True

def generate_all_with_processing(path, filename):
    input_file = path + filename
    img = cv2.imread(input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_params = [
        (img, path + 'result1'),
        (filter2(img), path + 'result2'),
        (filter4(img), path + 'result3'),
        (filter5(img), path + 'result4')
    ]

    pool = multiprocessing.Pool(processes=4)
    results = pool.starmap(generate, img_params)
    pool.close()
    pool.join()
    
    return True

if __name__ == '__main__':
    # generate('', '../MyPointArt/generateLocal/scripts/photo10.jpg')
    generate('', '../MyPointArt/generateLocal/scripts/test.jpg')
    # generate('', '../MyPointArt/generateLocal/scripts/photo4.jpg')