#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from scipy import fftpack, signal

sampling = "4:2:0"
interpolation = True 

mat = np.array([[0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312]])

quantization_y_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]])

quantization_cbcr_matrix = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                        [18, 21, 26, 66, 99, 99, 99, 99],
                        [24, 26, 56, 99, 99, 99, 99, 99],
                        [47, 66, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99]])

#3.2 colormap
def newCmap(keyColors=[(0,0,0),(1,1,1)], name = "gray", N=256):
    cm = clr.LinearSegmentedColormap.from_list(name, keyColors, N)
    return cm

#3.2 criar colormap
cm_red = newCmap([(0 ,0 ,0), (1, 0, 0)], "cm_red", 256)
cm_green = newCmap([(0 ,0 ,0), (0, 1, 0)], "cm_green", 256)
cm_blue = newCmap([(0 ,0 ,0), (0, 0, 1)], "cm_blue", 256)
cm_gray = newCmap([(0 ,0 ,0), (1, 1, 1)], "cm_gray", 256)

def encoder(img, fname):
    #print('ENCODER')
    R, G, B = splitRGB(img)
    #3.6
    """ showImg(R, 'imagem vermelha', fname, cm_red)
    showImg(G, 'imagem verde', fname, cm_green)
    showImg(B, 'imagem azul', fname, cm_blue) """
    #4.1
    new_img= pad(img, 32)
    #showImg(new_img, 'imagem com pad', fname)
    #5.3
    YCbCr = convertYCbCr(img)
    Y, Cb, Cr = splitRGB(YCbCr)
    
    """ showImg(Y, 'Y', fname, cm_gray)
    showImg(Cb, 'Cb', fname, cm_gray)
    showImg(Cr, 'Cr', fname, cm_gray) """

    #6.3
    y_d, cb_d, cr_d = imageDownsampling(YCbCr)
    cb_dh, cr_dh = imageUpsampling(y_d, cb_d, cr_d, Cb, Cr)

    #7.1.3
    Y_dct = Dct(y_d)
    Cb_dct = Dct(cb_d)
    Cr_dct = Dct(cr_d)

    showImg(np.log(np.abs(Y_dct) + 0.0001), 'Y_dct', fname, cm_gray)
    showImg(np.log(np.abs(Cb_dct) + 0.0001), 'Cb_dct', fname, cm_gray)
    showImg(np.log(np.abs(Cr_dct) + 0.0001), 'Cr_dct', fname, cm_gray)

    #7.2.3
    Y_dct8 = dctBlocos(y_d, 8)
    Cb_dct8 = dctBlocos(cb_d, 8)
    Cr_dct8 = dctBlocos(cr_d, 8)

    showImg(np.log(np.abs(Y_dct8) + 0.0001), 'Y_dct8', fname, cm_gray)
    showImg(np.log(np.abs(Cb_dct8) + 0.0001), 'Cb_dct8', fname, cm_gray)
    showImg(np.log(np.abs(Cr_dct8) + 0.0001), 'Cr_dct8', fname, cm_gray)

    #7.3
    Y_dct64 = dctBlocos(y_d, 64)
    Cb_dct64 = dctBlocos(cb_d, 64)
    Cr_dct64 = dctBlocos(cr_d, 64)

    showImg(np.log(np.abs(Y_dct64) + 0.0001), 'Y_dct64', fname, cm_gray)
    showImg(np.log(np.abs(Cb_dct64) + 0.0001), 'Cb_dct64', fname, cm_gray)
    showImg(np.log(np.abs(Cr_dct64) + 0.0001), 'Cr_dct64', fname, cm_gray)

    #8.3
    y_quant = quant(Y_dct, 75, quantization_y_matrix)
    showImg(np.log(np.abs(y_quant) + 0.0001), 'Y_quant', fname, cm_gray)
        
    
    return YCbCr, new_img, Y_dct, Cb_dct, Cr_dct, Y_dct8, Cb_dct8, Cr_dct8, Y_dct64, Cb_dct64, Cr_dct64

def decoder(pad_img, YCbCr, Y_dct, Cb_dct, Cr_dct, Y_dct8, Cb_dct8, Cr_dct8, Y_dct64, Cb_dct64, Cr_dct64, fname):
    #print('DECODER')
    #imgRec = joinRGB(R, G, B)
    #4.2 tirar pad
    shape = YCbCr.shape
    no_pad = unpad(pad_img, shape[0], shape[1])
    #showImg(no_pad, 'imagem sem pad', fname)
    
    #5.4
    RGB = convertRGB(YCbCr)
    R, G, B = splitRGB(RGB)
    """ showImg(R, 'imagem vermelha reconstruida', fname, cm_red)
    showImg(G, 'imagem verde reconstruida', fname, cm_green)
    showImg(B, 'imagem azul reconstruida', fname, cm_blue)
    
    showImg(RGB, 'imagem reconstruida', fname) """

    #7.1.4
    y_d = Idct(Y_dct)
    cb_d = Idct(Cb_dct)
    cr_d = Idct(Cr_dct)

    #7.2.4
    Y_d8 = IdctBlocos(Y_dct8, 8)
    Cb_d8 = IdctBlocos(Cb_dct8, 8)
    Cr_d8 = IdctBlocos(Cr_dct8, 8)

    #7.3
    Y_d64 = IdctBlocos(Y_dct64, 64)
    Cb_d64 = IdctBlocos(Cb_dct64, 64)
    Cr_d64 = IdctBlocos(Cr_dct64, 64)

#3.3 visualizaçao de imagem com colormap
def showImg(img, caption='', fname='', cmap=None):
    plt.figure()
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(caption + ': ' + fname)
    
#3.4 separar RGB
def splitRGB(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    
    return R, G, B

#3.5 juntar RGB
def joinRGB(R, G, B):
    nl, nc = R.shape
    imgRec = np.zeros((nl, nc, 3), dtype=np.uint8)
    imgRec[:,:,0] = R
    imgRec[:,:,1] = G
    imgRec[:,:,2] = B
    
    return imgRec

#4.1 adicionar pad
def pad(img, pad):
    nLine, nCol, _ = img.shape

    padLine = pad - (nLine % pad)  
    padCol = pad - (nCol % pad)  
    
    arrLine = np.repeat(img[-1:,:,:], padLine, axis=0)
    img_padded = np.vstack((img, arrLine))

    arrCol = np.repeat(img_padded[:,-1:,:], padCol, axis=1)
    img_padded = np.hstack((img_padded, arrCol))

    return img_padded

#4.2 unpad
def unpad(img, nlines, ncols):
    return img[:nlines, :ncols, :]

#5.1 converter RGB em YCbCr
def convertYCbCr(img):
    R, G, B = splitRGB(img)
    width, height, _ = img.shape
    mat = np.array([[0.299, 0.587, 0.114],
                    [-0.168736, -0.331264, 0.5],
                    [0.5, -0.418688, -0.081312]])

    YCbCr = np.zeros((width, height, 3))

    for i in range(3):
        YCbCr[:, :, i] = mat[i][0] * R + mat[i][1] * G + mat[i][2] * B

    YCbCr[:,:,1:] += 128
            
    return YCbCr

#5.2 converter YCbCr em RGB
def convertRGB(YCbCr):
    shape = YCbCr.shape
    RGB = np.zeros(shape)

    matInv = np.linalg.inv(mat)
    YCbCr[:, :, 1:] -= 128

    for i in range(3):
        RGB[:, :, i] = matInv[i][0] * YCbCr[:, :, 0] + matInv[i][1] * YCbCr[:, :, 1] + matInv[i][2] * YCbCr[:, :, 2]

    RGB = np.clip(RGB, 0, 255).astype(np.uint8)

    return RGB

# Ex 6.1
def imageDownsampling(ycbcr):
    imageComponents = []

    # Appending arrays to list
    for i in range(3):
        imageComponents.append(ycbcr[:, :, i])

    y_d = imageComponents[0]

    if sampling == "4:2:2":
        cb_d = np.delete(imageComponents[1], np.s_[1::2], 1)
        cr_d = np.delete(imageComponents[2], np.s_[1::2], 1)
    elif sampling == "4:2:0":
        cb_d = np.delete(imageComponents[1], np.s_[1::2], 1)
        cb_d = np.delete(cb_d, np.s_[1::2], 0)
        cr_d = np.delete(imageComponents[2], np.s_[1::2], 1)
        cr_d = np.delete(cr_d, np.s_[1::2], 0)
    else:
        print("Invalid downsampling variant!")
        return -1
    return y_d, cb_d, cr_d
    
# Ex 6.2
def imageUpsampling(y_d, cb_d, cr_d, cb, cr):
    if sampling == "4:2:2":
        if interpolation:
            cb_ch = signal.resample(cb_d, len(cb_d[0]) * 2, axis=1)
            cr_ch = signal.resample(cr_d, len(cr_d[0]) * 2, axis=1)
        else:
            cb_ch = np.repeat(cb_d, 2, axis=1)
            cr_ch = np.repeat(cr_d, 2, axis=1)
    elif sampling == "4:2:0":
        if interpolation:
            cb_ch = signal.resample(cb_d, len(cb_d) * 2, axis=0)
            cb_ch = signal.resample(cb_ch, len(cb[0]) * 2, axis=1)
            cr_ch = signal.resample(cr_d, len(cr_d) * 2, axis=0)
            cr_ch = signal.resample(cr_ch, len(cr[0]) * 2, axis=1)
        else:
            cb_ch = np.repeat(cb_d, 2, axis=0)
            cb_ch = np.repeat(cb_ch, 2, axis=1)
            cr_ch = np.repeat(cr_d, 2, axis=0)
            cr_ch = np.repeat(cr_ch, 2, axis=1)
    else:
        print("Invalid downsampling variant!")
        return -1
    
    return cb_ch, cr_ch

#7.1.1
def Dct(canal):
    canal_dct = fftpack.dct(fftpack.dct(canal, norm="ortho").T, norm="ortho").T

    return canal_dct

#7.1.2
def Idct(canal_dct):
    canal = fftpack.idct(fftpack.idct(canal_dct, norm="ortho").T, norm="ortho").T

    return canal

# Ex 7.2.1
def dctBlocos(canal, blocos):
    nl, nc = np.shape(canal)
    dct = np.zeros((nl, nc))

    for i in range(0, int(nl / blocos)):
        for j in range(0, int(nc / blocos)):
            temp = fftpack.dct(fftpack.dct(canal[i * blocos:(i + 1) * blocos, j * blocos:(j + 1) * blocos], norm="ortho").T, norm="ortho").T
            dct[i * blocos:(i + 1) * blocos, j * blocos:(j + 1) * blocos] = temp

    return dct

# Ex 7.2.2
def IdctBlocos(canal_dct, blocos):
    nl, nc = np.shape(canal_dct)
    canal = np.zeros((nl, nc))

    for i in range(0, int(nl / blocos)):
        for j in range(0, int(nc / blocos)):
            temp = fftpack.idct(
                fftpack.idct(canal_dct[i * blocos:(i + 1) * blocos, j * blocos:(j + 1) * blocos], norm="ortho").T, norm="ortho").T
            canal[i * blocos:(i + 1) * blocos, j * blocos:(j + 1) * blocos] = temp

    return canal

# 8.1 Quantização
def quant(channel, quality, matrix):
    dim = np.shape(channel)
    quant = np.zeros((dim[0], dim[1]))

    if quality < 50:
        sf = 50 / quality
    else:
        sf = (100 - quality) / 50

    for i in range(0, int(dim[0] / 8)):
        for j in range(0, int(dim[1] / 8)):
            if sf == 0:
                img = np.around(channel[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8])
            else:
                qsf = np.round(matrix * sf)
                qsf[qsf > 255] = 255
                qsf[qsf < 1] = 1
                img = np.divide(channel[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8], qsf)
                img = np.around(img)
            quant[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = img

    return quant

            
def main():
    #3.1 leitura da img
    fname = 'airport.bmp'
    img = plt.imread("../imagens/" + fname)
    
    #3.3 visualizar a imagem original
    #showImg(img, 'imagem orginal', fname)
    
    YCbCr, pad_img, Y_dct, Cb_dct, Cr_dct, Y_dct8, Cb_dct8, Cr_dct8, Y_dct64, Cb_dct64, Cr_dct64 = encoder(img, fname)
    
    decoder(pad_img, YCbCr, Y_dct, Cb_dct, Cr_dct, Y_dct8, Cb_dct8, Cr_dct8, Y_dct64, Cb_dct64, Cr_dct64, fname)
    
if __name__ == "__main__":
    main()

# %%
