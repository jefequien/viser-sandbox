"""RealSense visualizer

Connect to a RealSense camera, then visualize RGB-D readings as a point clouds. Requires
pyrealsense2.
"""
import time

import numpy as np
import viser

from viser_sandbox.util.projection import backproject_depth

# def point_cloud_arrays_from_frames(
#     depth_frame, color_frame
# ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
#     """Maps realsense frames to two arrays.

#     Returns:
#     - A point position array: (N, 3) float32.
#     - A point color array: (N, 3) uint8.
#     """
#     # Processing blocks. Could be tuned.
#     point_cloud = rs.pointcloud()  # type: ignore
#     decimate = rs.decimation_filter()  # type: ignore
#     decimate.set_option(rs.option.filter_magnitude, 3)  # type: ignore

#     # Downsample depth frame.
#     depth_frame = decimate.process(depth_frame)

#     # Map texture and calculate points from frames. Uses frame intrinsics.
#     point_cloud.map_to(color_frame)
#     points = point_cloud.calculate(depth_frame)

#     # Get color coordinates.
#     texture_uv = (
#         np.asanyarray(points.get_texture_coordinates())
#         .view(np.float32)
#         .reshape((-1, 2))
#     )
#     color_image = np.asanyarray(color_frame.get_data())
#     color_h, color_w, _ = color_image.shape

#     # Note: for points that aren't in the view of our RGB camera, we currently clamp to
#     # the closes available RGB pixel. We could also just remove these points.
#     texture_uv = texture_uv.clip(0.0, 1.0)

#     # Get positions and colors.
#     positions = np.asanyarray(points.get_vertices()).view(np.float32)
#     positions = positions.reshape((-1, 3))
#     colors = color_image[
#         (texture_uv[:, 1] * (color_h - 1.0)).astype(np.int32),
#         (texture_uv[:, 0] * (color_w - 1.0)).astype(np.int32),
#         :,
#     ]
#     N = positions.shape[0]

#     assert positions.shape == (N, 3)
#     assert positions.dtype == np.float32
#     assert colors.shape == (N, 3)
#     assert colors.dtype == np.uint8

#     return positions, colors


def main():
    # Start visualization server.
    viser_server = viser.ViserServer()

    while True:
        # Wait for a coherent pair of frames: depth and color
        # frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()
        K = np.array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])

        color = (np.random.rand(60, 120, 3) * 255).astype(np.uint8)
        depth = np.ones((60, 120), dtype=np.float32) * 10.0
        points = backproject_depth(depth, K)
        colors = color.reshape((-1, 3))

        # Place point cloud.
        viser_server.add_point_cloud(
            "/points_main",
            points=points,
            colors=colors,
            point_size=0.1,
        )

        # Place the frustum.
        h, w = depth.shape[:2]
        fov = 2 * np.arctan2(h / 2, K[1, 1] * h)
        aspect = w / h
        viser_server.add_camera_frustum(
            "/frames/t0/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            image=color,
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.zeros(3),
        )

        # Add some axes.
        viser_server.add_frame(
            "/frames/t0/frustum/axes",
            axes_length=0.05,
            axes_radius=0.005,
        )

        time.sleep(0.2)


if __name__ == "__main__":
    main()
