#imports
import numpy as np
import imageio
import matplotlib.pyplot as plt
import math
import cv2 as cv


def euclidean_distance(x,y):
    """ Function to perform euclidean distance """
    return math.sqrt(x**2 + y**2)

def gaussian_kernel(x, sigma):
    """Function to compute gaussian kernel"""
    gaussiank = (1 / (2*math.pi* (sigma**2))) * np.exp(-1 * (x**2) / (2 * (sigma**2)))  
    
    return gaussiank

def gaussian_filter(n, sigma):
    """Function to compute the gaussian filter"""
    gs = np.zeros([n,n])
    a = int((n-1) / 2)
    for i in range(n):
        for j in range(n):
            gs[i,j] = gaussian_kernel(euclidean_distance(i-a,j-a),sigma)
    
    return gs

def Dog(input_img, n, k, sigma, gamma):
    """Function to perform the difference of gaussians technique
        input_img => grayscale image
        n => gaussian filter dimension
        k => scale parameter 
        sigma => spacial scale 
        gamma => sensitivity of the edge detector       
    """
    
    gf1 = gaussian_filter(n, sigma)
    # k is 1.6 for a good trade-off between accurate approximation and adequate sensitivity
    gf2 = gaussian_filter(n, k*sigma) 
    
    a = int((n-1) / 2)
    
    dog = gf1 - (gamma * gf2) # gamma = 0.98
    
    f = np.pad(input_img, (a,a), 'constant').astype(np.int32)
    

    N,M = f.shape
    g = np.array(f, copy=True)
    g = np.zeros(input_img.shape)
    for x in range(a,N-a):
        
           for y in range(a,M-a):
                sub_f = f[ x-a : x+a+1 , y-a:y+a+1 ]
                g[x-a,y-a] = np.sum(np.multiply(sub_f, dog))
                
    return g

def soft_threshold(img, epsilon, phi):
    """Function to compute the soft threshold
       img => the grayscale image
       epsilon => ir controls the level above which the luminance adjustmentes will become white
       phi => it controls the slope of the falloff
    """
    out = np.array(img, copy=True)
    N,M = img.shape
    for i in range(N):
        for j in range(M):
            if img[i,j] < epsilon:
                out[i,j] = 1
            else:
                out[i,j] = 1 + np.tanh(phi*img[i,j])
    return out

def mean_threshold(img):
    """Function to compute the mean threshold"""
    mean = np.mean(img)
    
    out = np.array(img, copy=True)
    N,M = img.shape
    for i in range(N):
        for j in range(M):
            if img[i,j] <= mean:
                out[i,j] = 1
            else:
                out[i,j] = 0
    return out


def combine_color_edges(img, edge):
    """Function to combine the colored image with the edges"""
    R = img[:,:,0]
    G = img[:,:,1]                                                
    B = img[:,:,2] 
    
    N,M = edge.shape
    
    for i in range(N):
        for j in range(M):
            if int(edge[i,j]) == 1:
                R[i,j] = 0
                G[i,j] = 0
                B[i,j] = 0
             
    img_out =  np.array(img, copy=True).astype(np.uint32)
    img_out[:,:,0] = np.clip(R, 0, 255)
    img_out[:,:,1] = np.clip(G, 0, 255)
    img_out[:,:,2] = np.clip(B, 0, 255)
    
    return img_out


def color_slot(pixel):
    """Function to compute new value of the pixel in color quantization"""
    
    # each RGB chanel has 6 options now, which means 6*6*6 = 216 colors
    if pixel >= 0 and pixel < 8:
        return 0
    if pixel >= 8 and pixel < 16:
        return 8
    if pixel >= 16 and pixel < 32:
        return 16
    if pixel >= 32 and pixel < 64:
        return 32
    if pixel >= 64 and pixel < 128:
        return 64
    if pixel >= 128 and pixel < 256:
        return 255


def color_quantization(img):
    """Function to reduce the color range of the image"""
    R = img[:,:,0]
    G = img[:,:,1]                                                
    B = img[:,:,2] 
    
    N,M = R.shape
    
    # performin the color thresold in each RGB channel
    for i in range(N):
        for j in range(M):
            R[i,j] = color_slot(R[i,j])
            G[i,j] = color_slot(G[i,j])
            B[i,j] = color_slot(B[i,j])
    
    # put the channels RGB together again
    img_out =  np.array(img, copy=True).astype(np.uint32)
    img_out[:,:,0] = np.clip(R, 0, 255)
    img_out[:,:,1] = np.clip(G, 0, 255)
    img_out[:,:,2] = np.clip(B, 0, 255)
    
    return img_out 

def display(filename, n, sigma, epsilon, phi):
    img = cv.imread(filename)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    img_dog = Dog(img_gray, n, 1.6,  sigma, 0.98)
    
    img_color_smooth = cv.bilateralFilter(img_rgb, d=9, sigmaColor=150, sigmaSpace=150)
    img_color_qt = color_quantization(img_rgb)
    
    dog_t = soft_threshold(img_dog, epsilon, phi)
    dog_tt = mean_threshold(dog_t)
    img_out1 = combine_color_edges(img_color_smooth, dog_tt)
    img_out2 = combine_color_edges(img_color_qt, dog_tt) 
    
    cv.imwrite('output_img1_bilateral.png', img_out1)
    cv.imwrite('output_img2_colorqt.png', img_out2)


print("Input image path: ")
filename = str(input())
print("Parameters: \n")
print("n (gaussian filter size): ")
n = int(input())
print("gaussian sigma: ")
sigma = float(input())
print("soft threshold epsilon: ")
epsilon = float(input())
print("soft threshold phi: ")
phi = float(input())
display(filename, n, sigma, epsilon, phi)