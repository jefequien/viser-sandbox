"""Camera Visualizer

Connect to a RealSense camera, then visualize RGB-D readings as a point clouds. Requires
pyrealsense2.
"""

import cv2
import numpy as np
import viser
from PIL import Image
from tqdm import tqdm

from viser_sandbox.util.projection import backproject_depth


def main():
    # Start visualization server.
    viser_server = viser.ViserServer()
    # define a video capture object
    vid = cv2.VideoCapture(0)
    size = (640, 480)
    w, h = size

    # model_path = "../Depth-Anything-ONNX/weights/depth_anything_vits14.onnx"
    # session = ort.InferenceSession(
    #     model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    # )

    for _ in tqdm(range(10000000)):
        ret, frame = vid.read()
        if not ret:
            break

        image = Image.fromarray(frame[:, :, ::-1]).resize(size)
        image = np.array(image)

        # image_inp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        # image_inp = transform({"image": image_inp})["image"]  # C, H, W
        # image_inp = image_inp[None]  # B, C, H, W
        # disp = session.run(None, {"image": image_inp})[0]
        # disp = cv2.resize(disp[0, 0], size) + 1e-6
        # depth = 10 * (1 / disp) + 4
        depth = np.ones((h, w)) * 10.0

        K = np.array([[0.5, 0.0, 0.5], [0.0, 0.667, 0.5], [0.0, 0.0, 1.0]])
        points = backproject_depth(depth, K)
        colors = image.reshape((-1, 3))

        # Place point cloud.
        viser_server.add_point_cloud(
            "/points_main",
            points=points,
            colors=colors,
            point_size=0.1,
        )

        # Place the frustum.
        fov = 2 * np.arctan2(h / 2, K[1, 1] * h)
        aspect = w / h
        viser_server.add_camera_frustum(
            "/frames/t0/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            image=image,
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.zeros(3),
        )

        # Add some axes.
        viser_server.add_frame(
            "/frames/t0/frustum/axes",
            axes_length=0.05,
            axes_radius=0.005,
        )


if __name__ == "__main__":
    main()
