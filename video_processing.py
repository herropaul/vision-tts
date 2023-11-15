import cv2
import base64


def read_video_frames(video_path):
    """
    Reads video frames from a specified path, encodes them in JPEG format, 
    and converts them to Base64 encoded strings.

    Args:
    video_path (str): Path to the video file.

    Returns:
    list: A list of Base64 encoded strings representing each frame of the video.
    """
    # Initialize a video capture object with the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Unable to open video file.")

    base64_frames = []  # List to store Base64 encoded frames

    # Read each frame of the video
    while True:
        success, frame = video.read()
        if not success:
            break  # Exit loop if no more frames are available

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode(".jpg", frame)
        # Convert the encoded frame to a Base64 string and append it to the list
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    # Release the video capture object
    video.release()

    print(len(base64_frames), "frames read.")
    return base64_frames
