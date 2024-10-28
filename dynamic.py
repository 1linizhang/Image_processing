import cv2 as cv
import numpy as np

inputf = '7e7ca4ac00ddf608798e4ede18839bc8(1).mp4'
inputb = 'f34a84dfaef8377ff47751f4fb917361.mp4'
output = 'output.mp4'

f_cap = cv.VideoCapture(inputf)
b_cap = cv.VideoCapture(inputb)

fwidth = int(f_cap.get(cv.CAP_PROP_FRAME_WIDTH))
fheight = int(f_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
ffps = round(f_cap.get(cv.CAP_PROP_FPS))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
capWrite = cv.VideoWriter(output, fourcc, ffps, (fwidth, fheight))

retb, background = b_cap.read()
background = cv.resize(background, (fwidth, fheight))

bg_model = cv.createBackgroundSubtractorMOG2()

while f_cap.isOpened():
    retf, framef = f_cap.read()
    if not retf:
        break
    fg_mask = bg_model.apply(framef)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)
    fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel)

    contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(fg_mask)

    if contours:
        for contour in contours:
            if cv.contourArea(contour) > 500:
                cv.drawContours(mask, [contour], -1, (255), thickness=cv.FILLED)


    mask = cv.dilate(mask, kernel, iterations=1)

    fore = cv.bitwise_and(framef, framef, mask=mask)

    back = cv.bitwise_and(background, background, mask=cv.bitwise_not(mask))

    combined_frame = cv.add(fore, back)
    capWrite.write(combined_frame)

f_cap.release()
b_cap.release()
capWrite.release()
cv.destroyAllWindows()