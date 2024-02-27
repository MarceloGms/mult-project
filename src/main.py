import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

mat = np.array([[0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312]])

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
    showImg(R, 'imagem vermelha', fname, cm_red)
    showImg(G, 'imagem verde', fname, cm_green)
    showImg(B, 'imagem azul', fname, cm_blue)
    #4.1
    new_img= pad(img, 32)
    showImg(new_img, 'imagem com pad', fname)
    #5.3
    YCbCr = convertYCbCr(img)
    Y, Cb, Cr = splitRGB(YCbCr)
    
    showImg(Y, 'Y', fname, cm_gray)
    showImg(Cb, 'Cb', fname, cm_gray)
    showImg(Cr, 'Cr', fname, cm_gray)
    
    return YCbCr, new_img

def decoder(pad_img, YCbCr, fname):
    #imgRec = joinRGB(R, G, B)
    #4.2 tirar pad
    while np.all(pad_img[-1]==pad_img[-2]):
        pad_img=pad_img[:-1]
    while np.all(pad_img[:,-1]==pad_img[:,-2]):
        pad_img=pad_img[:,:-1]
        
    showImg(pad_img, 'imagem sem pad', fname)
    
    #5.4
    RGB = convertRGB(YCbCr)
    R, G, B = splitRGB(RGB)
    showImg(R, 'imagem vermelha reconstruida', fname, cm_red)
    showImg(G, 'imagem verde reconstruida', fname, cm_green)
    showImg(B, 'imagem azul reconstruida', fname, cm_blue)
    
    showImg(RGB, 'imagem reconstruida', fname)

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

#5.1 converter RGB em YCbCr
def convertYCbCr(img):
    R, G, B = splitRGB(img)
    width, height, _ = img.shape
    mat = np.array([[0.299, 0.587, 0.114],
                    [-0.168736, -0.331264, 0.5],
                    [0.5, -0.418688, -0.081312]])

    rgb_values = np.stack([R, G, B], axis=-1)  # Stack R, G, B along the last axis

    YCbCr = np.dot(rgb_values, mat.T)
    YCbCr[:,:,1:] += 128
            
    return YCbCr

#5.2 converter YCbCr em RGB
def convertRGB(YCbCr):
    width, height, _ = YCbCr.shape

    matInv = np.linalg.inv(mat)
    YCbCr[:, :, 1:] -= 128

    RGB = np.dot(YCbCr.reshape((width * height, 3)), matInv.T).reshape((width, height, 3))
    RGB = np.clip(RGB, 0, 255).astype(np.uint8)

    return RGB
            
def main():
    #3.1 leitura da img
    fname = 'airport.bmp'
    img = plt.imread("../imagens/" + fname)
    
    #3.3 visualizar a imagem original
    showImg(img, 'imagem orginal', fname)
    
    YCbCr, pad_img = encoder(img, fname)
    
    decoder(pad_img, YCbCr, fname)
    
if __name__ == "__main__":
    main()