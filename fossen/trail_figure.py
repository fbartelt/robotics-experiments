# %%
import os
import cv2
import numpy as np


def create_motion_trail(
    video_path,
    output_path,
    decay=0.9,
    interval=5,
    start_frame=0,
    final_frame=None,
    save_frames=False,
    frames_dir=None,
):
    """
    Creates a composite image where the newest frame has high opacity
    and older frames fade out (onion skinning effect).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Prepare frame-dump directory
    if save_frames:
        if frames_dir is None:
            frames_dir = os.path.dirname(output_path) or "."
        os.makedirs(frames_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(output_path))[0]
        print(f"Saving captured frames to {frames_dir}/{base}_frame_XXX.png")

    composite = first_frame.astype(np.float32)
    frame_count = start_frame
    saved_idx = 0

    if save_frames:
        fname = os.path.join(frames_dir, f"{base}_frame_{saved_idx:03d}.png")
        cv2.imwrite(fname, first_frame)
        saved_idx += 1

    while True:
        if final_frame is not None and frame_count >= final_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % interval != 0:
            continue

        # THE FIX: Swap decay and (1 - decay)
        # Now, 'frame' gets the 'decay' weight (e.g., 0.9), making it strong.
        # The 'composite' (old frames) gets the (1 - decay) weight (e.g., 0.1), making them fade.
        composite = cv2.addWeighted(
            composite, 1 - decay, frame.astype(np.float32), decay, 0
        )

        if save_frames:
            fname = os.path.join(frames_dir, f"{base}_frame_{saved_idx:03d}.png")
            cv2.imwrite(fname, frame)
            saved_idx += 1

    final_image = np.clip(composite, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, final_image)
    print(f"Motion trail image saved to {output_path}")

    if save_frames:
        print(f"Total captured frames saved: {saved_idx}")

    cap.release()


# --- Usage ---
fps = 30
decay = 0.21
n_snapshots = 8 # 6 - ellipse

final_time = 105 # 126 - ellipse 189 - toroidal
init_time = 0
interval = int(fps * (final_time - init_time) / n_snapshots)
base_path = "./fossen/media"

# ellipse | vertical_coil | toroidal_coil
curve_name = "vertical_coil"
frame_dir = os.path.join(base_path, "frames")

create_motion_trail(
    os.path.join(base_path, f"{curve_name}.mp4"),
    os.path.join(base_path, f"{curve_name}.png"),
    decay=decay,
    interval=interval,
    start_frame=fps * init_time,
    final_frame=fps * final_time,
    save_frames=True,
    frames_dir=frame_dir,
)

# Name is manually changed to "trail_{init_time}-{final_time}.png",
# where XZ: [0, 60], YZ: [60, 120], XY: [120, 180]

# 1 - 0.15 ** (1/8)
