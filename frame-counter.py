import cv2
import numpy as np
import sys
import io
import math
import argparse
import glob
import datetime
import time
import multiprocessing as mp

np.set_printoptions(threshold=sys.maxsize)


def count_frames(video_file, start_frame, end_frame, load_threshold, blank_threshold, blank_frame, threads):
    capture = cv2.VideoCapture(video_file)
    capture.set(1, start_frame)
    fps = capture.get(5)
    total_frames = end_frame - start_frame

    success, frame = capture.read()
    roi = None
    if success:
        roi = cv2.selectROI("Select game capture area, then press enter", frame, showCrosshair=True)
        cv2.destroyAllWindows()
    else:
        print("Failed to load capture")
        sys.exit(1)

    blank_image = cv2.imread("ref_images/blank.png", cv2.IMREAD_GRAYSCALE)
    # blank_image = cv2.cvtColor(blank_image, cv2.COLOR_GRAY2BGR)
    scale = (roi[2] / blank_image.shape[1], roi[3] / blank_image.shape[0])

    orig_ar = blank_image.shape[1] / blank_image.shape[0]
    new_ar = roi[2] / roi[3]
    adjustment = 1 - ((1 - (orig_ar / new_ar)) / 2)
    capture.set(1, blank_frame)
    success, brightness_frame = capture.read()
    brightness_frame = cv2.cvtColor(brightness_frame, cv2.COLOR_BGR2GRAY)
    brightness_frame = brightness_frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    brightness = np.mean(brightness_frame[5, 5] - blank_image[0, 0])
    base_blank_match = check_for_load(brightness_frame, blank_image)
    blank_threshold = math.floor(base_blank_match * 1000) / 1000.0 - 0.001

    if 1.33 <= new_ar:
        print(f"Adjusting scale by: {adjustment}")
        scale = (scale[0] * adjustment, scale[1])

    # load_images = [cv2.cvtColor(cv2.imread(ref_file, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    load_images = [cv2.imread(ref_file, cv2.IMREAD_GRAYSCALE)
                   for ref_file in glob.glob("ref_images/*.png") if ref_file != "ref_images\\blank.png"]

    print(f"Adjusting brightness by: {brightness}")
    blank_image = cv2.convertScaleAbs(blank_image, beta=brightness)
    load_images = [cv2.convertScaleAbs(img, alpha=2.0, beta=brightness) for img in load_images]

    load_images = [cv2.resize(img, (int(img.shape[1] * scale[0]), int(img.shape[0] * scale[1])), interpolation=cv2.INTER_AREA)
                   for img in load_images]
    blank_image = cv2.resize(blank_image, (roi[2], roi[3]), interpolation=cv2.INTER_AREA)

    frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    base_load_match = check_for_load(frame, blank_image)
    load_threshold_adjustment = max((base_load_match - 0.80) / 2, 0)
    print(f"Adjusting load image threshold by: {load_threshold_adjustment}")
    load_threshold -= load_threshold_adjustment

    capture.set(1, start_frame)
    frame_num = start_frame
    load_arr = mp.Array('i', total_frames)
    offset = 0
    finished = False
    frames_per_load = 1000
    proc_frames = int(frames_per_load / threads)
    while not finished:
        procs = []
        # Number of frames to load at a time
        for i in range(threads):
            frames = []
            start = frame_num - start_frame
            for j in range(proc_frames):
                success, frame = capture.read()
                if not success:
                    print(f"Failed to read frame: {frame_num}")
                    continue
                frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                frame_num += 1
                if frame_num >= end_frame:
                    finished = True
                    break

            procs.append(mp.Process(target=counter_process,
                                    args=(blank_image, load_images, load_arr, start, frames, load_threshold, blank_threshold)))
            procs[i].start()
            if finished:
                break

        for proc in procs:
            proc.join()
            offset += proc_frames
            print(f"\rProcessed {offset} frames", end="", flush=True)

    capture.release()
    cv2.destroyAllWindows()

    print("")
    frame_num = start_frame
    skipped = 0
    max_skips = 2
    load_start_frame = 0
    loading_frames = 0
    frames_to_remove = 0
    load_found = False
    loading = False

    for load in load_arr:
        if load == 0:
            if loading:
                if skipped < max_skips:
                    skipped += 1
                    frame_num += 1
                    continue
                loading = False
                if load_found:
                    frames_to_remove += loading_frames - int(39 / (60 / fps))
                    print(f"Load found {load_start_frame} - {load_start_frame +  loading_frames - 1}")
                    load_found = False

                loading_frames = 0
            else:
                skipped = 0
        else:
            if loading and skipped > 0:
                # we missed an intermediate frame
                loading_frames += skipped
                skipped = 0

            loading_frames += 1
            if not loading:
                load_start_frame = frame_num

            loading = True
            if load == 2:
                load_found = True
        frame_num += 1

    adjusted_frames = total_frames - frames_to_remove
    time_with_loads = str(datetime.timedelta(seconds=total_frames / fps))
    time_without_loads = str(datetime.timedelta(seconds=adjusted_frames / fps))
    return frames_to_remove, total_frames, adjusted_frames, time_with_loads, time_without_loads


def counter_process(blank_image, load_images, load_arr, start, frames, load_threshold, blank_threshold):
    frame_num = int(start)
    loading = True  # assume we are loading first
    load_found = False
    skipped = 0
    max_skips = 2

    for frame in frames:
        # print(f"\rFrame: {frame_num + 3720}", end="", flush=True)
        try:
            load_arr[frame_num] = 0
        except IndexError:
            print(f"Invalid index: {frame_num}")
            return

        if loading:
            half_frame = frame[int(frame.shape[0] / 2):, :]
            for img in load_images:
                load_match = check_for_load(half_frame, img)
                if load_match >= load_threshold:
                    load_arr[frame_num] = 2
                    load_found = True
                    break
            if not load_found:
                blank_match = check_for_load(frame, blank_image)
                if blank_match >= blank_threshold:
                    load_arr[frame_num] = 1
                elif skipped < max_skips:
                    skipped += 1
                else:
                    skipped = 0
                    loading = False
            load_found = False
        else:
            blank_match = check_for_load(frame, blank_image)
            if blank_match >= blank_threshold:
                loading = True
                load_arr[frame_num] = 1

        frame_num += 1


def check_for_load(frame, ref_image, method=cv2.TM_SQDIFF):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(frame, ref_image, method)
    # result = cv2.norm(frame, ref_image, cv2.NORM_L2)

    max_error = ref_image.size * 255 * 255
    # max_error = (ref_image.size ** 0.5) * 255
    min_match, max_match, min_loc, max_loc = cv2.minMaxLoc(result)

    if method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED:
        loc = min_loc
        match = 1 - (min_match / max_error)
    else:
        loc = max_loc
        match = (max_match / max_error)

    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.rectangle(frame_gray, loc, (loc[0] + ref_image.shape[1], loc[1] + ref_image.shape[0]), 255,
    #               2)
    # cv2.imshow("frame", frame_gray)
    # cv2.imshow("ref", ref_image)
    # cv2.waitKey(30000)
    # print(f" min: {match}")
    return match


def adjust_gamma(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(img, table)


def adjust_brightness(img, brightness=0):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y, x] = np.clip(img[y, x] + brightness, 0, 255)

    return img


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
    parser.add_argument("-lt", "--loadthreshold", type=float,
                        required=False, default=0.99, help="Threshold for detecting load messages")
    parser.add_argument("-bt", "--blankthreshold", type=float,
                        required=False, default=0.999, help="Threshold for detecting blank load screens")
    parser.add_argument("-t", "--threads", type=int,
                        required=False, default=4, help="Number of threads to run")
    parser.add_argument("-b", "--blank", type=int,
                        required=False, default=0, help="Frame with blank image")
    parser.add_argument("-sf", "--save", action="store_true", help="Save the frame at -s into file -o")
    parser.add_argument("-o", "--output", help="Output filename")
    args = parser.parse_args()
    if args.save:
        save_ref_image(args.input, args.start, args.output)
    else:
        assert args.end > args.start, "End frame must be after start frame"
        loading_frames, total_frames, adjusted_frames, time_with_loads, time_without_loads \
            = count_frames(args.input, args.start, args.end, args.loadthreshold, args.blankthreshold, args.blank, args.threads)
        print(f"Total frames: {total_frames}")
        print(f"Frames to remove: {loading_frames}")
        print(f"Adjusted frames: {adjusted_frames}")
        print(f"Time with full loads: {time_with_loads}")
        print(f"Time with adjusted loads: {time_without_loads}")
        print(f"Processing time: {str(datetime.timedelta(seconds=time.perf_counter()))}")
