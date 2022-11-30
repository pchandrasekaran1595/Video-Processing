import os
import cv2
import sys
import numpy as np

from typing import Union


BASE_PATH: str   = "\\".join(os.path.dirname(os.path.abspath(__file__)).split("\\")[:-1])
INPUT_PATH: str  = os.path.join(BASE_PATH, "input")
OUTPUT_PATH: str = os.path.join(BASE_PATH, "output")


def combine_and_alpha_blend() -> None:

    args_1_0: tuple = ("--filename-1", "-f1")
    args_1_1: tuple = ("--filename-2", "-f2")
    args_2: tuple   = ("--alpha", "-a")
    args_3: tuple   = ("--combinevid", "-cv")
    args_4: tuple   = ("--combineimg", "-ci")
    args_5: tuple   = ("--alphavid", "-av")
    args_6: tuple   = ("--alphaimg", "-ai")
    args_7: tuple   = ("--save", "-s")
    args_8: tuple   = ("--width", "-w")
    args_9: tuple   = ("--height", "-h")
    args_10: tuple  = ("--downscale", "-ds")
    args_11: tuple  = ("--vertical", "-v")

    do_combine_vid: bool = False
    do_combine_img: bool = False
    do_alpha_vid: bool = False
    do_alpha_img: bool = False
    alpha: float = 0.1
    filename_1: str = "Video_1.mp4"
    filename_2: str = "Video_2.mp4"
    save: bool = False
    vertical: bool = False

    if args_1_0[0] in sys.argv: filename_1 = sys.argv[sys.argv.index(args_1_0[0]) + 1]
    if args_1_0[1] in sys.argv: filename_1 = sys.argv[sys.argv.index(args_1_0[1]) + 1]

    if args_1_1[0] in sys.argv: filename_2 = sys.argv[sys.argv.index(args_1_1[0]) + 1]
    if args_1_1[1] in sys.argv: filename_2 = sys.argv[sys.argv.index(args_1_1[1]) + 1]

    assert filename_1 in os.listdir(INPUT_PATH), f"File 1 ({filename_1}) not found in input directory"
    assert filename_2 in os.listdir(INPUT_PATH), f"File 2 ({filename_2}) not found in input directory"

    if args_2[0] in sys.argv: alpha = float(sys.argv[sys.argv.index(args_2[0]) + 1])
    if args_2[1] in sys.argv: alpha = float(sys.argv[sys.argv.index(args_2[1]) + 1])

    if args_3[0] in sys.argv or args_3[1] in sys.argv: do_combine_vid = True
    if args_4[0] in sys.argv or args_4[1] in sys.argv: do_combine_img = True
    if args_5[0] in sys.argv or args_5[1] in sys.argv: do_alpha_vid = True
    if args_6[0] in sys.argv or args_6[1] in sys.argv: do_alpha_img = True
    if args_7[0] in sys.argv or args_7[1] in sys.argv: save = True

    cap_1 = cv2.VideoCapture(os.path.join(INPUT_PATH, filename_1))
    
    if do_combine_vid or do_alpha_vid: cap_2 = cv2.VideoCapture(os.path.join(INPUT_PATH, filename_2))
    if do_combine_img or do_alpha_img: image = cv2.imread(os.path.join(INPUT_PATH, filename_2))
    
    width: int  = int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    factor: Union[float, None] = None

    if args_8[0] in sys.argv: width = int(sys.argv[sys.argv.index(args_8[0]) + 1])
    if args_8[1] in sys.argv: width = int(sys.argv[sys.argv.index(args_8[1]) + 1])

    if args_9[0] in sys.argv: height = int(sys.argv[sys.argv.index(args_9[0]) + 1])
    if args_9[1] in sys.argv: height = int(sys.argv[sys.argv.index(args_9[1]) + 1])

    if args_10[0] in sys.argv: factor = float(sys.argv[sys.argv.index(args_10[0]) + 1])
    if args_10[1] in sys.argv: factor = float(sys.argv[sys.argv.index(args_10[1]) + 1])

    if args_11[0] in sys.argv or args_11[1] in sys.argv: vertical = True

    if save:
        if do_combine_vid or do_combine_img:
            if vertical:
                out = cv2.VideoWriter(
                    os.path.join(OUTPUT_PATH, "V-Stack Combined.mp4"), 
                    cv2.VideoWriter_fourcc(*"mp4v"), 
                    30, 
                    (width, 2*height)
                )
            else:
                out = cv2.VideoWriter(
                    os.path.join(OUTPUT_PATH, "H-Stack Combined.mp4"), 
                    cv2.VideoWriter_fourcc(*"mp4v"), 
                    30, 
                    (2*width, height)
                )
        else:
            out = cv2.VideoWriter(
                os.path.join(OUTPUT_PATH, "Alpha Combined.mp4"), 
                cv2.VideoWriter_fourcc(*"mp4v"), 
                30, 
                (width, height)
            )

    if do_combine_vid:

        while cap_1.isOpened() and cap_2.isOpened():
            ret_1, frame_1 = cap_1.read()
            ret_2, frame_2 = cap_2.read()

            if ret_1 and ret_2:
                frame_1 = cv2.resize(src=frame_1, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
                frame_2 = cv2.resize(src=frame_2, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

                if factor:
                    frame_1 = cv2.resize(src=frame_1, dsize=(int(width / factor), int(height / factor)), interpolation=cv2.INTER_CUBIC)
                    frame_2 = cv2.resize(src=frame_2, dsize=(int(width / factor), int(height / factor)), interpolation=cv2.INTER_CUBIC)

                if vertical:
                    frame = np.vstack((frame_1, frame_2))
                else:
                    frame = np.hstack((frame_1, frame_2))

                if save:
                    out.write(frame)
                else:
                    cv2.imshow("Stacked", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"): 
                    break
            else:
                if save:
                    break
                else:
                    cap_1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cap_2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cap_1.release()
        cap_2.release()

        if save:
            out.release()
    

    if do_combine_img:
        image = cv2.resize(src=image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        if factor: image = cv2.resize(src=image, dsize=(int(width / factor), int(height / factor)), interpolation=cv2.INTER_CUBIC)

        while cap_1.isOpened():
            ret_1, frame_1 = cap_1.read()

            if ret_1:
                frame_1 = cv2.resize(src=frame_1, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
                if factor: frame_1 = cv2.resize(src=frame_1, dsize=(int(width / factor), int(height / factor)), interpolation=cv2.INTER_CUBIC)

                if vertical:
                    frame = np.vstack((frame_1, image))
                else:
                    frame = np.hstack((frame_1, image))

                if save:
                    out.write(frame)
                else:
                    cv2.imshow("Stacked", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            else:
                if save:
                    break
                else:
                    cap_1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cap_1.release()

        if save:
            out.release()
    

    if do_alpha_vid:

        while cap_1.isOpened() and cap_2.isOpened():
            ret_1, frame_1 = cap_1.read()
            ret_2, frame_2 = cap_2.read()

            if ret_1 and ret_2:
                frame_1 = cv2.resize(src=frame_1, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
                frame_2 = cv2.resize(src=frame_2, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

                if factor:
                    frame_1 = cv2.resize(src=frame_1, dsize=(int(width / factor), int(height / factor)), interpolation=cv2.INTER_CUBIC)
                    frame_2 = cv2.resize(src=frame_2, dsize=(int(width / factor), int(height / factor)), interpolation=cv2.INTER_CUBIC)
                
                frame = cv2.addWeighted(frame_1, 1-alpha, frame_2, alpha, 0)

                if save:
                    out.write(frame)
                else:
                    cv2.imshow("Alpha Blended", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            else:
                if save:
                    break
                else:
                    cap_1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cap_2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cap_1.release()
        cap_2.release()
        
        if save:
            out.release()
    

    if do_alpha_img:
        image = cv2.resize(src=image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
        if factor: image = cv2.resize(src=image, dsize=(int(width / factor), int(height / factor)), interpolation=cv2.INTER_CUBIC)

        while cap_1.isOpened():
            ret_1, frame_1 = cap_1.read()

            if ret_1:
                frame_1 = cv2.resize(src=frame_1, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
                if factor: frame_1 = cv2.resize(src=frame_1, dsize=(int(width / factor), int(height / factor)), interpolation=cv2.INTER_CUBIC)

                frame = cv2.addWeighted(frame_1, 1-alpha, image, alpha, 0)

                if save:
                    out.write(frame)
                else:
                    cv2.imshow("Alpha Blended", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            else:
                if save:
                    break
                else:
                    cap_1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        cap_1.release()

        if save:
            out.release()
