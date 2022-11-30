import os
import cv2
import sys
import numpy as np

from typing import Union


BASE_PATH: str   = "\\".join(os.path.dirname(os.path.abspath(__file__)).split("\\")[:-1])
INPUT_PATH: str  = os.path.join(BASE_PATH, "input")
OUTPUT_PATH: str = os.path.join(BASE_PATH, "output")


def new_color(pixel: int, num_colors: int) -> int:
    colors = [(1/num_colors)*i for i in range(num_colors)]
    distances = [abs(pixel-colors[i]) for i in range(len(colors))]
    index = distances.index(min(distances))
    return colors[index]


def find_closest_color(pixel: int, num_colors: int) -> int:
    colors = [i*(1/num_colors) for i in range(num_colors+1)]
    distances = [abs(colors[i]-pixel) for i in range(len(colors))]
    index = distances.index(min(distances))
    return colors[index]


def process() -> None:
    args_1: tuple  = ("--file", "-f")
    args_2: tuple  = ("--gauss-blur", "-gb")
    args_3: tuple  = ("--avg-blur", "-ab")
    args_4: tuple  = ("--median-blur", "-mb")
    args_5: tuple  = ("--gamma", "-g")
    args_6: tuple  = ("--linear", "-l")
    args_7: tuple  = ("--clahe", "-ae")
    args_8: tuple  = ("--hist-equ", "-he")
    args_9: tuple  = ("--hue")
    args_10: tuple = ("--saturation", "-sat")
    args_11: tuple = ("--vibrance", "-v")
    args_12: tuple = ("--sharpen", "-sh")
    args_13: tuple = ("--width", "-w")
    args_14: tuple = ("--height", "-h")
    args_15: tuple = ("--downscale", "-ds")
    args_16: tuple = ("--posterize", "-post")
    args_17: tuple = ("--dither", "-dith")
    args_18: tuple = ("--save", "-s")

    filename: str = "Video_1.mp4"
    do_gauss_blur: bool = False
    do_average_blur: bool = False
    do_median_blur: bool = False
    do_gamma: bool = False
    do_linear: bool = False
    do_clahe: bool = False
    do_histogram_equalization: bool = False
    do_hue: bool = False
    do_saturation: bool = False
    do_vibrance: bool = False
    width: Union[int, None] = None
    height: Union[int, None] = None
    factor: Union[float, None] = None
    do_sharpen: bool = False
    do_posterize: bool = False
    do_dither: bool = False
    save: bool = False

    if args_1[0] in sys.argv: filename = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: filename = sys.argv[sys.argv.index(args_1[1]) + 1]


    if args_2[0] in sys.argv: 
        do_gauss_blur = True
        setup = sys.argv[sys.argv.index(args_2[0]) + 1] + ","
        
        gaussian_blur_kernel_size: str = setup.split(",")[0]
        gaussian_blur_sigmaX: str = setup.split(",")[1]

        if int(gaussian_blur_kernel_size) == 1: gaussian_blur_kernel_size = 3
        elif int(gaussian_blur_kernel_size) % 2 == 0: int(gaussian_blur_kernel_size) + 1
        else: gaussian_blur_kernel_size = int(gaussian_blur_kernel_size)

        if gaussian_blur_sigmaX != "": gaussian_blur_sigmaX = float(gaussian_blur_sigmaX)
        else: gaussian_blur_sigmaX = 0

    if args_2[1] in sys.argv: 
        do_gauss_blur = True
        setup = sys.argv[sys.argv.index(args_2[1]) + 1] + ","
        
        gaussian_blur_kernel_size: str = setup.split(",")[0]
        gaussian_blur_sigmaX: str = setup.split(",")[1]

        if int(gaussian_blur_kernel_size) == 1: gaussian_blur_kernel_size = 3
        if int(gaussian_blur_kernel_size) % 2 == 0: int(gaussian_blur_kernel_size) + 1
        else: gaussian_blur_kernel_size = int(gaussian_blur_kernel_size)

        if gaussian_blur_sigmaX != "": gaussian_blur_sigmaX = float(gaussian_blur_sigmaX)
        else: gaussian_blur_sigmaX = 0


    if args_3[0] in sys.argv: 
        do_average_blur = True
        average_blur_kernel_size: int = int(sys.argv[sys.argv.index(args_3[0]) + 1])

    if args_3[1] in sys.argv: 
        do_average_blur = True
        average_blur_kernel_size: int = int(sys.argv[sys.argv.index(args_3[1]) + 1])
    

    if args_4[0] in sys.argv: 
        do_median_blur = True
        median_blur_kernel_size = int(sys.argv[sys.argv.index(args_4[0]) + 1])

        if median_blur_kernel_size == 1: median_blur_kernel_size = 3
        if median_blur_kernel_size % 2 == 0: median_blur_kernel_size + 1

    if args_4[1] in sys.argv: 
        do_median_blur = True
        median_blur_kernel_size = int(sys.argv[sys.argv.index(args_4[1]) + 1])

        if median_blur_kernel_size == 1: median_blur_kernel_size = 3
        if median_blur_kernel_size % 2 == 0: median_blur_kernel_size + 1
    

    if args_5[0] in sys.argv: 
        do_gamma = True
        gamma = float(sys.argv[sys.argv.index(args_5[0]) + 1])

    if args_5[1] in sys.argv: 
        do_gamma = True
        gamma = float(sys.argv[sys.argv.index(args_5[1]) + 1])
    

    if args_6[0] in sys.argv: 
        do_linear = True
        linear = float(sys.argv[sys.argv.index(args_6[0]) + 1])

    if args_6[1] in sys.argv: 
        do_linear= True
        linear = float(sys.argv[sys.argv.index(args_6[1]) + 1])
    

    if args_7[0] in sys.argv: 
        do_clahe = True
        setup = sys.argv[sys.argv.index(args_7[0]) + 1] + ","

        clipLimit = float(setup.split(",")[0])
        tileGridSize = setup.split(",")[1]

        if tileGridSize != "": tileGridSize = int(tileGridSize)
        else: tileGridSize = 2

    if args_7[1] in sys.argv: 
        do_clahe = True
        setup = sys.argv[sys.argv.index(args_7[1]) + 1] + ","

        clipLimit = float(setup.split(",")[0])
        tileGridSize = setup.split(",")[1]

        if tileGridSize != "": tileGridSize = int(tileGridSize)
        else: tileGridSize = 2
    

    if args_8[0] in sys.argv: do_histogram_equalization = True
    if args_8[1] in sys.argv: do_histogram_equalization = True
    

    if args_9 in sys.argv:
        do_hue = True
        hue = float(sys.argv[sys.argv.index(args_9) + 1])
    

    if args_10[0] in sys.argv:
        do_saturation = True
        saturation = float(sys.argv[sys.argv.index(args_10[0]) + 1])

    if args_10[1] in sys.argv:
        do_saturation = True
        saturation = float(sys.argv[sys.argv.index(args_10[1]) + 1])
    

    if args_11[0] in sys.argv:
        do_vibrance = True
        vibrance = float(sys.argv[sys.argv.index(args_11[0]) + 1])
    
    if args_11[1] in sys.argv:
        do_vibrance = True
        vibrance = float(sys.argv[sys.argv.index(args_11[1]) + 1])
    

    if args_12[0] in sys.argv:
        do_sharpen = True
        sharpen_kernel_size = int(sys.argv[sys.argv.index(args_12[0]) + 1])
        if sharpen_kernel_size % 2 == 0: sharpen_kernel_size + 1

    if args_12[1] in sys.argv:
        do_sharpen = True
        sharpen_kernel_size = int(sys.argv[sys.argv.index(args_12[1]) + 1])
        if sharpen_kernel_size % 2 == 0: sharpen_kernel_size + 1
    

    if args_13[0] in sys.argv: width = int(sys.argv[sys.argv.index(args_13[0]) + 1])
    if args_13[1] in sys.argv: width = int(sys.argv[sys.argv.index(args_13[1]) + 1])
    

    if args_14[0] in sys.argv: height = int(sys.argv[sys.argv.index(args_14[0]) + 1])
    if args_14[1] in sys.argv: height = int(sys.argv[sys.argv.index(args_14[1]) + 1])


    if args_15[0] in sys.argv: factor = float(sys.argv[sys.argv.index(args_15[0]) + 1])
    if args_15[1] in sys.argv: factor = float(sys.argv[sys.argv.index(args_15[1]) + 1])
    
    
    if args_16[0] in sys.argv:
        do_posterize = True
        num_colors = int(sys.argv[sys.argv.index(args_16[0]) + 1])
        
    if args_16[1] in sys.argv:
        do_posterize = True
        num_colors = int(sys.argv[sys.argv.index(args_16[1]) + 1])
    
    
    if args_17[0] in sys.argv:
        do_dither = True
        num_colors = int(sys.argv[sys.argv.index(args_17[0]) + 1])
        
    if args_17[1] in sys.argv:
        do_dither = True
        num_colors = int(sys.argv[sys.argv.index(args_17[1]) + 1])
  

    if args_18[0] in sys.argv or args_18[1] in sys.argv: save = True

    assert filename in os.listdir(INPUT_PATH), f"{filename} not found in input directory"

    cap = cv2.VideoCapture(os.path.join(INPUT_PATH, filename))
    orig_width: int  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    if save: 
        if isinstance(width, int):
            if width != orig_width:
                out = cv2.VideoWriter(
                    os.path.join(OUTPUT_PATH, "Processed - {}.mp4".format(filename[:-4])), 
                    cv2.VideoWriter_fourcc(*"mp4v"), 
                    30, 
                    (width, orig_height)
                )
            
        elif isinstance(height, int):
            if height != orig_height:
                out = cv2.VideoWriter(
                    os.path.join(OUTPUT_PATH, "Processed - {}.mp4".format(filename[:-4])), 
                    cv2.VideoWriter_fourcc(*"mp4v"), 
                    30, 
                    (orig_width, height)
                )
        
        elif isinstance(width, int) and isinstance(height, int):
            if width != orig_width or height != orig_height:
                out = cv2.VideoWriter(
                    os.path.join(OUTPUT_PATH, "Processed - {}.mp4".format(filename[:-4])), 
                    cv2.VideoWriter_fourcc(*"mp4v"), 
                    30, 
                    (width, height)
                )
        
        else:
            out = cv2.VideoWriter(
                os.path.join(OUTPUT_PATH, "Processed - {}.mp4".format(filename[:-4])), 
                cv2.VideoWriter_fourcc(*"mp4v"), 
                30, 
                (orig_width, orig_height)
            )
    

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if isinstance(width, int):
                if width != orig_width:
                    frame = cv2.resize(src=frame, dsize=(width, orig_height), interpolation=cv2.INTER_CUBIC)
                
            if isinstance(height, int):
                if height != orig_height:
                    frame = cv2.resize(src=frame, dsize=(orig_width, height), interpolation=cv2.INTER_CUBIC)
            
            if isinstance(width, int) and isinstance(height, int):
                if width != orig_width or height != orig_height:
                    frame = cv2.resize(src=frame, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

            if factor:
                frame = cv2.resize(src=frame, dsize=(int(frame.shape[1] / factor), int(frame.shape[0] / factor)), interpolation=cv2.INTER_CUBIC)

            if do_gauss_blur:
                frame = cv2.GaussianBlur(
                    src=frame, 
                    ksize=(gaussian_blur_kernel_size, gaussian_blur_kernel_size), 
                    sigmaX=gaussian_blur_sigmaX,
                )
            
            if do_average_blur: 
                frame = cv2.blur(
                    src=frame, 
                    ksize=(average_blur_kernel_size, 
                        average_blur_kernel_size),
                )

            if do_median_blur: 
                frame = cv2.medianBlur(
                    src=frame,
                    ksize=median_blur_kernel_size
                )
            
            if do_gamma: 
                frame = frame / 255
                frame = np.clip(((frame ** gamma) * 255), 0, 255).astype("uint8")
    
            if do_linear: 
                frame = np.clip((frame + (linear % 255)), 0, 255).astype("uint8")
            
            if do_clahe:
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
                for i in range(3): frame[:, :, i] = clahe.apply(frame[:, :, i])
            
            if do_histogram_equalization: 
                for i in range(3): frame[:, :, i] = cv2.equalizeHist(frame[:, :, i])
            
            if do_hue:
                frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2HSV)
                feature = frame[:, :, 0]
                feature = np.clip((hue * feature), 0, 179).astype("uint8")
                frame[:, :, 0] = feature
                frame = cv2.cvtColor(src=frame, code=cv2.COLOR_HSV2BGR)
    
            if do_saturation:
                frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2HSV)
                feature = frame[:, :, 1]
                feature = np.clip((saturation * feature), 0, 255).astype("uint8")
                frame[:, :, 1] = feature
                frame = cv2.cvtColor(src=frame, code=cv2.COLOR_HSV2BGR)
    
            if do_vibrance: 
                frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2HSV)
                feature = frame[:, :, 2]
                feature = np.clip((vibrance * feature), 0, 255).astype("uint8")
                frame[:, :, 2] = feature
                frame = cv2.cvtColor(src=frame, code=cv2.COLOR_HSV2BGR)
            
            if do_sharpen: 
                kernel = cv2.getStructuringElement(
                    shape=cv2.MORPH_CROSS, 
                    ksize=(sharpen_kernel_size, sharpen_kernel_size)
                ) * -1
                kernel[int(sharpen_kernel_size / 2), int(sharpen_kernel_size / 2)] = ((sharpen_kernel_size - 1) * 2) + 1

                frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
                frame = np.clip(frame, 0, 255).astype("uint8")
            
            if do_posterize: 
                h, w, c = frame.shape
                frame = frame / 255
                for c in range(c):
                    for i in range(h):
                        for j in range(w):
                            frame[i][j][c] = new_color(frame[i][j][c], num_colors)
                frame = np.clip((frame*255), 0, 255).astype("uint8")
            
            if do_dither: 
                frame = frame / 255
                h, w, c = frame.shape
                for c in range(c):
                    for i in range(h-1):
                        for j in range(1, w-1):
                            old_pixel = frame[i][j][c]
                            new_pixel = find_closest_color(old_pixel, num_colors)
                            frame[i][j][c] = new_pixel
                            quant_error = old_pixel - new_pixel

                            frame[i][j+1][c]   = frame[i][j+1][c] + (quant_error * 7/16)
                            frame[i+1][j+1][c] = frame[i+1][j+1][c] + (quant_error * 1/16)
                            frame[i+1][j][c]   = frame[i+1][j][c] + (quant_error * 5/16)
                            frame[i+1][j-1][c] = frame[i+1][j-1][c] + (quant_error * 3/16)
                frame = np.clip((frame*255), 0, 255).astype("uint8")
    

            if save:
                out.write(frame)
            else:
                cv2.imshow("Processed", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
        else:
            if save:
                break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    cap.release()

    if save:
        out.release()
