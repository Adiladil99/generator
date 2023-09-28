import numpy as np
import cv2
from skimage.draw import line_aa, ellipse_perimeter
from math import atan2
from skimage.transform import resize
from time import time
import argparse

OUTPUT_FILE = "output.png"
SIDE_LEN = 300
EXPORT_STRENGTH = 0.035
PULL_AMOUNT = 4000
RANDOM_NAILS = None
RADIUS1_MULTIPLIER = 1
RADIUS2_MULTIPLIER = 1
NAIL_STEP = 240

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def largest_square(image: np.ndarray) -> np.ndarray:
    short_edge = np.argmin(image.shape[:2])  # 0 = vertical <= horizontal; 1 = otherwise
    short_edge_half = image.shape[short_edge] // 2
    long_edge_center = image.shape[1 - short_edge] // 2
    if short_edge == 0:
        return image[:, long_edge_center - short_edge_half:
                        long_edge_center + short_edge_half]
    if short_edge == 1:
        return image[long_edge_center - short_edge_half:
                     long_edge_center + short_edge_half, :]


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
    line = picture[rr, cc] + str_strength * val
    line = np.clip(line, a_min=0, a_max=1)

    return line, rr, cc


def find_best_nail_position(current_position, nails, str_pic, orig_pic, str_strength):

    best_cumulative_improvement = -99999
    best_nail_position = None
    best_nail_idx = None

    if RANDOM_NAILS is not None:
        nail_ids = np.random.choice(range(len(nails)), size=RANDOM_NAILS, replace=False)
        nails_and_ids = list(zip(nail_ids, nails[nail_ids]))
    else:
        nails_and_ids = enumerate(nails)

    for nail_idx, nail_position in nails_and_ids:

        overlayed_line, rr, cc = get_aa_line(current_position, nail_position, str_strength, str_pic)

        before_overlayed_line_diff = np.abs(str_pic[rr, cc] - orig_pic[rr, cc]) ** 2
        after_overlayed_line_diff = np.abs(overlayed_line - orig_pic[rr, cc]) ** 2

        cumulative_improvement = np.sum(before_overlayed_line_diff - after_overlayed_line_diff)

        if cumulative_improvement >= best_cumulative_improvement:
            best_cumulative_improvement = cumulative_improvement
            best_nail_position = nail_position
            best_nail_idx = nail_idx

    return best_nail_idx, best_nail_position, best_cumulative_improvement


def create_art(nails, orig_pic, str_pic, str_strength, i_limit=None):

    start = time()
    iter_times = []

    current_position = nails[0]
    pull_order = [0]

    i = 0
    fails = 0
    while True:
        start_iter = time()

        i += 1

        if i % 500 == 0:
            print(f"Iteration {i}")

        # if i_limit is None:
        #     if fails >= 5:
        #         break
        # else:
        #     if i > i_limit:
        #         break
        if fails >= 5:
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
        iter_times.append(time() - start_iter)

    print(f"Time: {time() - start}")
    print(f"Avg iteration time: {np.mean(iter_times)}")
    print(len(pull_order))
    return pull_order


def scale_nails(x_ratio, y_ratio, nails):
    return [(int(y_ratio * nail[0]), int(x_ratio * nail[1])) for nail in nails]


def pull_order_to_array_bw(order, canvas, nails, strength):
    # Draw a black and white pull order on the defined resolution

    for pull_start, pull_end in zip(order, order[1:]):  # pairwise iteration
        rr, cc, val = line_aa(nails[pull_start][0], nails[pull_start][1],
                              nails[pull_end][0], nails[pull_end][1])
        canvas[rr, cc] += val * strength

    return np.clip(canvas, a_min=0, a_max=1)

def generate(path, filename):

    LONG_SIDE = 300
    input_file = path + filename
    img = cv2.imread(input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if np.any(img > 100):
        img = img / 255
    # cv2.imshow('frame', img); cv2.waitKey(0)
    if RADIUS1_MULTIPLIER == 1 and RADIUS2_MULTIPLIER == 1:
        img = largest_square(img)
        img = cv2.resize(img, (LONG_SIDE, LONG_SIDE))

    shape = (len(img), len(img[0]))

    nails = create_circle_nail_positions(shape, NAIL_STEP, RADIUS1_MULTIPLIER, RADIUS2_MULTIPLIER)

    print(f"Nails amount: {len(nails)}")

    orig_pic = rgb2gray(img) * 0.9
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
    cv2.imwrite(path + 'result.png', result * 255)

    with open(path + "result.json", "w") as f:
    # with open("result.json", "w") as f:
        f.write(str(pull_order))
    print("done")
    return True

if __name__ == '__main__':
    # main('', '../MyPointArt/generateLocal/scripts/photo10.jpg')
    generate('', '../MyPointArt/generateLocal/scripts/photo7.jpg')
    # main('', '../MyPointArt/generateLocal/scripts/photo4.jpg')