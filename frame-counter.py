import cv2
import numpy as np
import sys
import argparse
import glob
import datetime
import math
np.set_printoptions(threshold=sys.maxsize)


def count_frames(video_file, start_frame, end_frame, load_threshold, blank_threshold):
    capture = cv2.VideoCapture(video_file)
    fps = capture.get(5)
    total_frames = end_frame - start_frame
    capture.set(1, start_frame)
    frame_num = start_frame
    load_start_frame = 0
    loading_frames = 0
    frames_to_remove = 0
    load_found = False
    loading = False
    load_images = [cv2.imread(ref_file, cv2.IMREAD_GRAYSCALE)
                   for ref_file in glob.glob("ref_images/*.png") if ref_file != "ref_images\\blank.png"]
    blank_image = cv2.imread("ref_images/blank.png", cv2.IMREAD_GRAYSCALE)
    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            print(f"Failed to read frame: {frame_num}")
            continue

        if frame_num > end_frame:
            break

        if loading:
            ref_images = load_images
            if check_for_load(frame, ref_images, load_threshold):
                loading_frames += 1
                load_found = True
            else:
                ref_images = [blank_image]
                if check_for_load(frame, ref_images, blank_threshold):
                    loading_frames += 1
                else:
                    if load_found:
                        frames_to_remove += loading_frames - math.ceil(39 / (60 / fps))
                        print(f" Load found {load_start_frame} - {frame_num - 1}")
                        load_found = False
                    loading_frames = 0
                    loading = False

        else:
            ref_images = [blank_image]
            if check_for_load(frame, ref_images, blank_threshold):
                loading_frames += 1
                loading = True
                load_start_frame = frame_num

        frame_num += 1
        print(f"\rFrame: {frame_num}", end="", flush=True)


    print("")
    capture.release()
    cv2.destroyAllWindows()
    adjusted_frames = total_frames - frames_to_remove
    time_with_loads = str(datetime.timedelta(seconds=total_frames / fps))
    time_without_loads = str(datetime.timedelta(seconds=adjusted_frames / fps))
    return frames_to_remove, total_frames, adjusted_frames, time_with_loads, time_without_loads


def check_for_load(frame, ref_images, threshold=1000):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for ref_image in ref_images:
        result = cv2.matchTemplate(frame_gray, ref_image, cv2.TM_SQDIFF)

        min_match = np.min(result) / 100000
        # print(f" min: {min_match}")
        # cv2.imshow("frame", frame_gray)
        # cv2.imshow("ref", ref_image)
        # cv2.waitKey(30000)
        if min_match <= threshold:
            return True

    return False


def save_ref_image(video_file, frame_num, output_filename):
    capture = cv2.VideoCapture(video_file)
    capture.set(1, frame_num)
    success, frame = capture.read()
    if success:
        cv2.imwrite(output_filename, frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input video file to be counted")
    parser.add_argument("-s", "--start", required=True, type=int, help="Starting frame number (timer start frame)")
    parser.add_argument("-e", "--end", type=int, help="Ending frame number (timer end frame)")
    parser.add_argument("-lt", "--loadthreshold", type=int,
                        required=False, default=500, help="Threshold for detecting load messages")
    parser.add_argument("-bt", "--blankthreshold", type=int,
                        required=False, default=1000, help="Threshold for detecting blank load screens")
    parser.add_argument("-sf", "--save", action="store_true", help="Save the frame at -s into file -o")
    parser.add_argument("-o", "--output", help="Output filename")
    args = parser.parse_args()
    if args.save:
        save_ref_image(args.input, args.start, args.output)
    else:
        assert args.end > args.start, "End frame must be after start frame"
        loading_frames, total_frames, adjusted_frames, time_with_loads, time_without_loads \
            = count_frames(args.input, args.start, args.end, args.loadthreshold, args.blankthreshold)
        print(f"Total frames: {total_frames}")
        print(f"Frames to remove: {loading_frames}")
        print(f"Adjusted frames: {adjusted_frames}")
        print(f"Time with full loads: {time_with_loads}")
        print(f"Time with adjusted loads: {time_without_loads}")
