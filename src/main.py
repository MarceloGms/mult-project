#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from scipy import fftpack
import cv2

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
    R, G, B = splitRGB(img)
    #3.6
    """ showImg(R, 'imagem vermelha', fname, cm_red)
    showImg(G, 'imagem verde', fname, cm_green)
    showImg(B, 'imagem azul', fname, cm_blue) """
    #4.1
    new_img= pad(img, 32)
    # showImg(new_img, 'imagem com pad', fname)
    #5.3
    YCbCr = convertYCbCr(new_img)
    Y, Cb, Cr = splitRGB(YCbCr)
    
    """ showImg(Y, 'Y', fname, cm_gray)
    showImg(Cb, 'Cb', fname, cm_gray)
    showImg(Cr, 'Cr', fname, cm_gray) """

    #6.3
    y_d, cb_d, cr_d = downsampling(Y, Cb, Cr, 4,2,2)

    #7.1.3
    Y_dct = Dct(y_d)
    Cb_dct = Dct(cb_d)
    Cr_dct = Dct(cr_d)
    
    """ showImg(np.log(np.abs(Y_dct) + 0.0001), 'Y_dct', fname, cm_gray)
    showImg(np.log(np.abs(Cb_dct) + 0.0001), 'Cb_dct', fname, cm_gray)
    showImg(np.log(np.abs(Cr_dct) + 0.0001), 'Cr_dct', fname, cm_gray) """

    #7.2.3
    Y_dct8 = dctBlocos(y_d, 8)
    Cb_dct8 = dctBlocos(cb_d, 8)
    Cr_dct8 = dctBlocos(cr_d, 8)

    """ showImg(np.log(np.abs(Y_dct8) + 0.0001), 'Y_dct8', fname, cm_gray)
    showImg(np.log(np.abs(Cb_dct8) + 0.0001), 'Cb_dct8', fname, cm_gray)
    showImg(np.log(np.abs(Cr_dct8) + 0.0001), 'Cr_dct8', fname, cm_gray) """

    #7.3
    """ Y_dct64 = dctBlocos(y_d, 64)
    Cb_dct64 = dctBlocos(cb_d, 64)
    Cr_dct64 = dctBlocos(cr_d, 64)

    showImg(np.log(np.abs(Y_dct64) + 0.0001), 'Y_dct64', fname, cm_gray)
    showImg(np.log(np.abs(Cb_dct64) + 0.0001), 'Cb_dct64', fname, cm_gray)
    showImg(np.log(np.abs(Cr_dct64) + 0.0001), 'Cr_dct64', fname, cm_gray) """

    #8.3
    Y_q = quant(Y_dct8, 75, quantization_y_matrix)
    showImg(np.log(np.abs(Y_q) + 0.0001), f'Y_q (QF {75})', fname, cm_gray)
    Cb_q = quant(Cb_dct8, 75, quantization_cbcr_matrix)
    showImg(np.log(np.abs(Cb_q) + 0.0001), f'Cb_q (QF {75})', fname, cm_gray)
    Cr_q = quant(Cr_dct8, 75, quantization_cbcr_matrix)
    showImg(np.log(np.abs(Cr_q) + 0.0001), f'Cr_q (QF {75})', fname, cm_gray)
        
    return Y_q, Cb_q, Cr_q

def decoder(Y_q, Cb_q, Cr_q, img_original, fname):
    #imgRec = joinRGB(R, G, B)
    #8.4
    Y_dct8 = iquant(Y_q, 75, quantization_y_matrix)
    Cb_dct8 = iquant(Cb_q, 75, quantization_cbcr_matrix)
    Cr_dct8 = iquant(Cr_q, 75, quantization_cbcr_matrix)

    """ showImg(np.log(np.abs(Y_dct8) + 0.0001), 'Y_dct8-rec', fname, cm_gray)
    showImg(np.log(np.abs(Cb_dct8) + 0.0001), 'Cb_dct8-rec', fname, cm_gray)
    showImg(np.log(np.abs(Cr_dct8) + 0.0001), 'Cr_dct8-rec', fname, cm_gray) """

    #7.2.4
    Y_d = IdctBlocos(Y_dct8, 8)
    Cb_d = IdctBlocos(Cb_dct8, 8)
    Cr_d = IdctBlocos(Cr_dct8, 8)
    
    #6.3
    cb_u, cr_u = upsampling(Y_d, Cb_d, Cr_d, 4,2,2)
    
    #5.4
    RGB = convertRGB(Y_d, cb_u, cr_u)
    
    #4.2 tirar pad
    nl, nc, _ = img_original.shape
    no_pad = unpad(RGB, nl, nc)
    showImg(no_pad, 'imagem reconstruida', fname)

#3.3 visualizacao de imagem com colormap
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

#4,2 unpad
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
def convertRGB(Y,Cb,Cr):
    shape = Y.shape
    RGB = np.zeros((shape[0], shape[1], 3))

    matInv = np.linalg.inv(mat)
    Cb -= 128
    Cr -= 128

    for i in range(3):
        RGB[:, :, i] = matInv[i][0] * Y + matInv[i][1] * Cb + matInv[i][2] * Cr

    RGB = np.clip(RGB, 0, 255).astype(np.uint8)

    return RGB

# Ex 6.1
def downsampling (Y, Cb, Cr, x, y, z):  
    if(z!=0):
        Y_d = cv2.resize(Y, dsize= None, fx=1, fy=1) 
        Cb_d = cv2.resize(Cb, dsize= None, fx=y/x, fy=1)
        Cr_d = cv2.resize(Cr, dsize= None, fx=y/x, fy=1)
    else:        
        Y_d = cv2.resize(Y, dsize= None, fx=1, fy=1) 
        Cb_d = cv2.resize(Cb, dsize= None, fx=y/x, fy=y/x)
        Cr_d = cv2.resize(Cr, dsize= None, fx=y/x, fy=y/x)

    """ showImg(Y_d, cmap= cm_gray, fname="Y_d")
    showImg(Cb_d, cmap= cm_gray, fname="Cb_d")
    showImg(Cr_d, cmap= cm_gray, fname="Cr_d") """
    
    return  Y_d, Cb_d, Cr_d
    
# Ex 6.2
def upsampling (Y_d, Cb_d, Cr_d, x, y, z):
    
    if(z!=0):
        Y_u = cv2.resize(Y_d, dsize= None, fx=1, fy=1) 
        Cb_u = cv2.resize(Cb_d, dsize= None, fx=x/y, fy=1)
        Cr_u = cv2.resize(Cr_d, dsize= None, fx=x/y, fy=1)
    else:        
        Y_u = cv2.resize(Y_d, dsize= None, fx=1, fy=1) 
        Cb_u = cv2.resize(Cb_d, dsize= None, fx=x/y, fy=x/y)
        Cr_u = cv2.resize(Cr_d, dsize= None, fx=x/y, fy=x/y)
    
    """ showImg(Y_u, cmap= cm_gray, fname="Y_u")
    showImg(Cb_u, cmap= cm_gray, fname="Cb_u")
    showImg(Cr_u, cmap= cm_gray, fname="Cr_u") """
    
    return Cb_u, Cr_u

#7.1.1
def Dct(canal):
    canal_dct = fftpack.dct(fftpack.dct(canal, norm="ortho").T, norm="ortho").T

    return canal_dct

#7.1.2
def invDct(canal_dct):
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
def quant(canal, qual, matriz):
    nl, nc = np.shape(canal)
    quant = np.zeros((nl, nc))

    if qual < 50:
        sf = 50 / qual
    else:
        sf = (100 - qual) / 50

    for i in range(0, int(nl / 8)):
        for j in range(0, int(nc / 8)):
            if sf == 0:
                temp = np.around(canal[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8])
            else:
                qsf = np.round(matriz * sf)
                qsf = np.clip(qsf, 1, 255)
                temp = np.divide(canal[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8], qsf)
                temp = np.around(temp)
            quant[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = temp

    return quant

# Ex 8.2
def iquant(canal_q, qual, matriz):
    nl, nc = np.shape(canal_q)
    canal = np.zeros((nl, nc))

    if qual < 50:
        sf = 50 / qual
    else:
        sf = (100 - qual) / 50

    for i in range(0, int(nl / 8)):
        for j in range(0, int(nc / 8)):
            if sf == 0:
                temp = np.around(canal_q[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8])
            else:
                qsf = np.round(matriz * sf)
                qsf = np.clip(qsf, 1, 255)
                temp = np.multiply(canal_q[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8], qsf)
                temp = np.around(temp)
            canal[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = temp

    return canal
            
def main():
    fname = 'airport.bmp'
    img = plt.imread("../imagens/" + fname)
    
    #3.3 visualizar a imagem original
    #showImg(img, 'imagem orginal', fname)
    
    Y, Cb, Cr = encoder(img, fname)
    
    decoder(Y, Cb, Cr, img, fname)
    
if __name__ == "__main__":
    main()
# %%
