import math

from monocular_camera import MonocularCamera
import time
import cv2
import numpy as np
from utils.datasets import *
import utils.torch_utils as torch_utils
from utils.general import *
from models.experimental import *

weights = 'best.pt'
model_size = (256, 256)

map_points = [[2.600, 2.540],
              [4.340, 1.435],
              [6.080, 2.540],
              [4.340, 3.645]]

camera_position = (5.7, 10.0)

confidence_thresh = 0.5


def letterbox(img, new_shape=(256, 256), color=(114, 114, 114), auto=True, scale_up=True):
    # Resize image to a 32*x rectangle
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new_W / old_w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # only scale down to get better mAP
    if not scale_up:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    # minimum rectangle
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding

    # divide padding into 2 sides
    dw /= 2
    dh /= 2
    # to resize image
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


def detect(input_image, matrix):
    global colors, camera_position, confidence_thresh

    boxed_image = letterbox(input_image, model_size)
    # Stack
    image_data = np.stack(boxed_image, 0)

    # Convert, BGR to RGB, bsx3x416x416
    image_data = image_data[:, :, ::-1].transpose(2, 0, 1)
    image_data = np.ascontiguousarray(image_data)

    image_data = torch.from_numpy(image_data).to(device)
    # u8 to fp16/32
    image_data = image_data.half()
    # from 0~255 to 0.0~1.0
    image_data /= 255.0
    if image_data.ndimension() == 3:
        image_data = image_data.unsqueeze(0)

    # Inference
    t1 = torch_utils.time_sync()

    predict = model(image_data, augment=False)[0]
    # Apply NMS
    predict = non_max_suppression(predict, confidence_thresh, 0.45, classes=0, agnostic=False)

    t2 = torch_utils.time_sync()
    car_map_position = []

    # print("Inference Time:", t2 - t1)
    # Process detections
    for i, det in enumerate(predict):
        s = '%g:' % i
        s += '%gx%g ' % image_data.shape[2:]

        labels_list = []
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(image_data.shape[2:], det[:, :4], input_image.shape).round()

            # Print results
            for c in det[:, -1].detach().unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string
            # Write results
            for *xy, conf, cls in det:
                if True:
                    pst1, pst2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                    center_point = [(pst1[0] + pst2[0]) / 2, (pst1[1] + pst2[1]) / 2]
                    original_position = cv2.perspectiveTransform(np.array([[center_point]], dtype=np.float32), matrix)
                    car_position_array = original_position[0][0]
                    print(car_position_array)

                    # 误差补偿
                    e = error_offset / 100.0
                    yaw_angle = math.atan2(camera_position[1] - car_position_array[1],
                                           camera_position[0] - car_position_array[0])
                    corrected_position = [car_position_array[0] + e * math.cos(yaw_angle),
                                             car_position_array[1] + e * math.sin(yaw_angle)]
                    car_map_position.append(corrected_position)

                    # Add bbox to image
                    label = '(%.2f, %.2f)' % (corrected_position[0], corrected_position[1])
                    color = colors[int(cls)]
                    # draw rect and put text
                    plot_one_box(xy, input_image, label=label, color=color, line_thickness=2)
    return car_map_position


def get_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        temp_points.append([x, y])


def select_points(camera):
    global temp_points

    temp_points = []

    cv2.destroyAllWindows()

    cv2.namedWindow('select_points', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('select_points', get_point)

    while len(temp_points) < 4:
        frame = camera.grab()
        # ret, frame = cam.read()
        for pt in temp_points:
            cv2.circle(frame, pt, 10, (0, 255, 0), cv2.FILLED)
        put_text_at_line(frame, 0, 'Double click to select points.')
        cv2.imshow('select_points', frame)
        cv2.imshow('requirement', draw_map_points_at_map())
        if cv2.waitKey(1) == ord('e'):
            temp_points = []
    cv2.destroyWindow('select_points')
    cv2.destroyWindow('requirement')

    original_filed_matrix = np.array(temp_points, dtype=np.float32)
    filed_array = np.array(map_points, dtype=np.float32)
    cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('offset', 'map', 0, 200, change_offset_callback)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    return cv2.getPerspectiveTransform(original_filed_matrix, filed_array)


def change_offset_callback(x):
    global error_offset
    error_offset = x


def put_text_at_line(img, ln, s):
    sz, baseLine = cv2.getTextSize(s, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
    cv2.putText(img, s, (0, sz[1] + (baseLine + sz[1]) * ln), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 121, 242), 2)


def draw_position_at_map(positions):
    global map_image
    map_temp = map_image.copy()
    resolution = 0.01085
    for p in positions:
        center_pt = (int(p[0] / resolution), int(p[1] / resolution))
        center_pt = flip_coord(center_pt)
        cv2.circle(map_temp, center_pt, int(0.3 / resolution), (0, 255, 0), cv2.FILLED)
    return map_temp


def draw_map_points_at_map():
    global map_image
    map_temp = map_image.copy()
    resolution = 0.01085
    for i, p in enumerate(map_points):
        center_pt = (int(p[0] / resolution), int(p[1] / resolution))
        center_pt = flip_coord(center_pt)
        cv2.circle(map_temp, center_pt, int(0.05 / resolution), (0, 0, 255), cv2.FILLED)
        cv2.putText(map_temp, str(i + 1), center_pt, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    return map_temp


def flip_coord(point):
    return map_image.shape[1] - point[0], point[1]


if __name__ == '__main__':
    device = torch_utils.select_device()
    model = attempt_load(weights, map_location=device)
    image_size = check_img_size(model_size[0], s=model.stride.max())
    model.half()

    map_image = cv2.imread('map.png')

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    cam = MonocularCamera('MV-SUA133GC-Group0.config')
    # cam = cv2.VideoCapture(0)
    st = time.time()

    temp_points = []
    trans = select_points(cam)

    error_offset = 0.0

    while True:
        frame = cam.grab()
        # ret, frame = cam.read()
        if frame is not None:
            car_positions = detect(frame, trans)
            cv2.imshow('map', draw_position_at_map(car_positions))

            et = time.time()
            put_text_at_line(frame, 0, f"FPS: {1 / (et - st):.2f}")
            put_text_at_line(frame, 1, 'Press Q to quit')
            put_text_at_line(frame, 2, 'Press E to reselect the points')
            for pt in temp_points:
                cv2.circle(frame, pt, 10, (0, 0, 255), cv2.FILLED)
            st = et

            cv2.imshow("result", frame)

        ch = cv2.waitKey(1)
        if ch == ord('q'):
            break
        if ch == ord('e'):
            trans = select_points(cam)

    cam.close()
