import cv2
import cv2 as cv
import numpy as np

inputf='7e7ca4ac00ddf608798e4ede18839bc8(1).mp4'
inputb='f34a84dfaef8377ff47751f4fb917361.mp4'
output='output1.mp4'

f_cap=cv.VideoCapture(inputf)
b_cap=cv.VideoCapture(inputb)

fwidth=int(f_cap.get(cv.CAP_PROP_FRAME_WIDTH))
fheight=int(f_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
ffps=round(f_cap.get(cv.CAP_PROP_FPS))


fourcc = cv.VideoWriter_fourcc(*'mp4v')
capWrite = cv.VideoWriter(output, fourcc, ffps, (fwidth, fheight))

bfps=round(b_cap.get(cv.CAP_PROP_FPS))
bframeCount=int(b_cap.get(cv.CAP_PROP_FRAME_COUNT))

def edge1(binary):
    img = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, fheight - 1):
        for j in range(1, fwidth - 1):
            if binary[i, j] == 0:
                img[i, j] = 255
            else:
                sum = (binary[i - 1, j] + binary[i + 1, j] +
                                 binary[i, j + 1] + binary[i, j - 1] +
                                 binary[i - 1, j + 1] + binary[i - 1, j - 1] +
                                 binary[i + 1, j - 1] + binary[i + 1, j + 1])
                if sum == 0:
                    img[i, j] = 0
    return img

while f_cap.isOpened():
    retf,framef=f_cap.read()
    if not retf:
        break

    cindex=int(f_cap.get(cv.CAP_PROP_POS_FRAMES))-1
    bindex=int((cindex*bfps/ffps)%bframeCount)
    b_cap.set(cv.CAP_PROP_POS_FRAMES,bindex)

    retb,frameb=b_cap.read()
    if not retb:
        b_cap.set(cv.CAP_PROP_POS_FRAMES,0)
        retb,frameb=b_cap.read()

    fgray=cv.cvtColor(framef,cv.COLOR_BGR2GRAY)
    _,binary_fframe=cv.threshold(fgray, 128 ,255,cv.THRESH_BINARY)
    edge2=edge1(binary_fframe)


    if frameb.shape[:2] != framef.shape[:2]:
        frameb = cv.resize(frameb, (framef.shape[1], framef.shape[0]))

    fore=cv.bitwise_and(framef,framef,mask=edge2)
    back2=cv.bitwise_not(edge2)
    back=cv.bitwise_and(frameb,frameb,mask=back2)
    if fore.shape != back.shape:
        back = cv.resize(back, (fore.shape[1], fore.shape[0]))
    combined_frame=cv.add(fore,back)

    capWrite.write(combined_frame)


f_cap.release()
b_cap.release()
capWrite.release()
cv.destroyAllWindows()




