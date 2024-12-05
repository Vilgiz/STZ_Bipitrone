import cv2
from image import Image  # Ensure this module is implemented correctly
from Camera_std import Camera  # Ensure this module is implemented correctly

if __name__ == '__main__':
    # Uncomment the following line if using a custom camera implementation
    # camera = Camera()

    # Open the video file
    video_path = "Video_20241202174919002.mp4"
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
        exit(1)

    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()
        if not ret:  # Break the loop if no frame is returned (end of video)
            print("End of video or cannot read frame.")
            break

        image = Image(frame)
        frame = image.transform_zone(frame)
        # frame = image.transform_chees(frame)
        frame = image.image_correction(frame)
        image.prepare_frames(frame)
        frame_painted, __, __ = image.detect_contours(frame)
        frame_painted = image.draw_contours(frame_painted)

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Exit on pressing 'q'
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
    # camera.end()  # Uncomment if using a custom camera class
