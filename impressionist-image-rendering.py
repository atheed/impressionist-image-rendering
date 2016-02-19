'''
Some boilerplate code contributed by: 
Michael Guerzhoy, University of Toronto
'''

import os
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
import matplotlib.image as mpimg
import scipy.misc
try:
    import Image
except ImportError:
    print 'PIL not found. You cannot view the image'
import os 
from scipy import *
from scipy.ndimage import *
from scipy.signal import convolve2d as conv

np.set_printoptions(threshold = np.nan)  

def canny(im, sigma, thresHigh = 50,thresLow = 10):
    '''
        Takes an input image in the range [0, 1] and generate a gradient image
        with edges marked by 1 pixels.
    '''
    imin = im.copy() * 255.0

    # Create the gauss kernel for blurring the input image
    # It will be convolved with the image
    # wsize should be an odd number
    wsize = 5
    gausskernel = gaussFilter(sigma, window = wsize)
    # fx is the filter for vertical gradient
    # fy is the filter for horizontal gradient
    # Please note the vertical direction is positive X

    fx = createFilter([0,  1, 0,
                       0,  0, 0,
                       0, -1, 0])
    fy = createFilter([ 0, 0, 0,
                       -1, 0, 1,
                        0, 0, 0])

    imout = conv(imin, gausskernel, 'valid')
    # print "imout:", imout.shape
    gradxx = conv(imout, fx, 'valid')
    gradyy = conv(imout, fy, 'valid')

    gradx = np.zeros(im.shape)
    grady = np.zeros(im.shape)
    padx = (imin.shape[0] - gradxx.shape[0]) / 2.0
    pady = (imin.shape[1] - gradxx.shape[1]) / 2.0
    gradx[padx:-padx, pady:-pady] = gradxx
    grady[padx:-padx, pady:-pady] = gradyy
    
    # Net gradient is the square root of sum of square of the horizontal
    # and vertical gradients

    grad = hypot(gradx, grady)
    theta = arctan2(grady, gradx)
    theta = 180 + (180 / pi) * theta
    # Only significant magnitudes are considered. All others are removed
    xx, yy = where(grad < 10)
    theta[xx, yy] = 0
    grad[xx, yy] = 0

    # The angles are quantized. This is the first step in non-maximum
    # supression. Since, any pixel will have only 4 approach directions.
    x0,y0 = where(((theta<22.5)+(theta>157.5)*(theta<202.5)
                   +(theta>337.5)) == True)
    x45,y45 = where( ((theta>22.5)*(theta<67.5)
                      +(theta>202.5)*(theta<247.5)) == True)
    x90,y90 = where( ((theta>67.5)*(theta<112.5)
                      +(theta>247.5)*(theta<292.5)) == True)
    x135,y135 = where( ((theta>112.5)*(theta<157.5)
                        +(theta>292.5)*(theta<337.5)) == True)

    theta = theta
    Image.fromarray(theta).convert('L').save('Angle map.jpg')
    theta[x0,y0] = 0
    theta[x45,y45] = 45
    theta[x90,y90] = 90
    theta[x135,y135] = 135
    x,y = theta.shape       
    temp = Image.new('RGB',(y,x),(255,255,255))
    for i in range(x):
        for j in range(y):
            if theta[i,j] == 0:
                temp.putpixel((j,i),(0,0,255))
            elif theta[i,j] == 45:
                temp.putpixel((j,i),(255,0,0))
            elif theta[i,j] == 90:
                temp.putpixel((j,i),(255,255,0))
            elif theta[i,j] == 45:
                temp.putpixel((j,i),(0,255,0))
    retgrad = grad.copy()
    x,y = retgrad.shape

    for i in range(x):
        for j in range(y):
            if theta[i,j] == 0:
                test = nms_check(grad,i,j,1,0,-1,0)
                if not test:
                    retgrad[i,j] = 0

            elif theta[i,j] == 45:
                test = nms_check(grad,i,j,1,-1,-1,1)
                if not test:
                    retgrad[i,j] = 0

            elif theta[i,j] == 90:
                test = nms_check(grad,i,j,0,1,0,-1)
                if not test:
                    retgrad[i,j] = 0
            elif theta[i,j] == 135:
                test = nms_check(grad,i,j,1,1,-1,-1)
                if not test:
                    retgrad[i,j] = 0

    init_point = stop(retgrad, thresHigh)
    # Hysteresis tracking. Since we know that significant edges are
    # continuous contours, we will exploit the same.
    # thresHigh is used to track the starting point of edges and
    # thresLow is used to track the whole edge till end of the edge.

    while (init_point != -1):
        #Image.fromarray(retgrad).show()
        # print 'next segment at',init_point
        retgrad[init_point[0],init_point[1]] = -1
        p2 = init_point
        p1 = init_point
        p0 = init_point
        p0 = nextNbd(retgrad,p0,p1,p2,thresLow)

        while (p0 != -1):
            #print p0
            p2 = p1
            p1 = p0
            retgrad[p0[0],p0[1]] = -1
            p0 = nextNbd(retgrad,p0,p1,p2,thresLow)

        init_point = stop(retgrad,thresHigh)

    # Finally, convert the image into a binary image
    x,y = where(retgrad == -1)
    retgrad[:,:] = 0
    retgrad[x,y] = 1.0
    return retgrad
    

def createFilter(rawfilter):
    '''
        This method is used to create an NxN matrix to be used as a filter,
        given a N*N list
    '''
    order = pow(len(rawfilter), 0.5)
    order = int(order)
    filt_array = array(rawfilter)
    outfilter = filt_array.reshape((order,order))
    return outfilter
    

def gaussFilter(sigma, window = 3):
    '''
        This method is used to create a gaussian kernel to be used
        for the blurring purpose. inputs are sigma and the window size
    '''
    kernel = zeros((window,window))
    c0 = window // 2

    for x in range(window):
        for y in range(window):
            r = hypot((x-c0),(y-c0))
            val = (1.0/2*pi*sigma*sigma)*exp(-(r*r)/(2*sigma*sigma))
            kernel[x,y] = val
    return kernel / kernel.sum()
    
 
def nms_check(grad, i, j, x1, y1, x2, y2):
    '''
        Method for non maximum supression check. A gradient point is an
        edge only if the gradient magnitude and the slope agree

        for example, consider a horizontal edge. if the angle of gradient
        is 0 degress, it is an edge point only if the value of gradient
        at that point is greater than its top and bottom neighbours.
    '''
    try:
        if (grad[i,j] > grad[i+x1,j+y1]) and (grad[i,j] > grad[i+x2,j+y2]):
            return 1
        else:
            return 0
    except IndexError:
        return -1
        
     
def stop(im, thres):
    '''
        This method is used to find the starting point of an edge.
    '''
    X,Y = where(im > thres)
    try:
        y = Y.min()
    except:
        return -1
    X = X.tolist()
    Y = Y.tolist()
    index = Y.index(y)
    x = X[index]
    return [x,y]
    
   
def nextNbd(im, p0, p1, p2, thres):
    '''
        This method is used to return the next point on the edge.
    '''
    kit = [-1,0,1]
    X,Y = im.shape
    for i in kit:
        for j in kit:
            if (i+j) == 0:
                continue
            x = p0[0]+i
            y = p0[1]+j

            if (x<0) or (y<0) or (x>=X) or (y>=Y):
                continue
            if ([x,y] == p1) or ([x,y] == p2):
                continue
            if (im[x,y] > thres): #and (im[i,j] < 256):
                return [x,y]
    return -1
    

def colorImSave(filename, array):
    imArray = scipy.misc.imresize(array, 3., 'nearest')
    if (len(imArray.shape) == 2):
        scipy.misc.imsave(filename, cm.jet(imArray))
    else:
        scipy.misc.imsave(filename, imArray)
        

def markStroke(mrkd, p0, p1, rad, val):
    # Mark the pixels that will be painted by
    # a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1).
    # These pixels are set to val in the ny x nx double array mrkd.
    # The paintbrush is circular with radius rad>0
    
    sizeIm = mrkd.shape
    sizeIm = sizeIm[0:2];
    nx = sizeIm[1]
    ny = sizeIm[0]
    p0 = p0.flatten('F')
    p1 = p1.flatten('F')
    rad = max(rad,1)
    # Bounding box
    concat = np.vstack([p0,p1])
    bb0 = np.floor(np.amin(concat, axis=0))-rad
    bb1 = np.ceil(np.amax(concat, axis=0))+rad
    # Check for intersection of bounding box with image.
    intersect = 1
    if ((bb0[0] > nx) or (bb0[1] > ny) or (bb1[0] < 1) or (bb1[1] < 1)):
        intersect = 0
    if intersect:
        # Crop bounding box.
        bb0 = np.amax(np.vstack([np.array([bb0[0], 1]), np.array([bb0[1],1])]), axis=1)
        bb0 = np.amin(np.vstack([np.array([bb0[0], nx]), np.array([bb0[1],ny])]), axis=1)
        bb1 = np.amax(np.vstack([np.array([bb1[0], 1]), np.array([bb1[1],1])]), axis=1)
        bb1 = np.amin(np.vstack([np.array([bb1[0], nx]), np.array([bb1[1],ny])]), axis=1)
        # Compute distance d(j,i) to segment in bounding box
        tmp = bb1 - bb0 + 1
        szBB = [tmp[1], tmp[0]]
        q0 = p0 - bb0 + 1
        q1 = p1 - bb0 + 1
        t = q1 - q0
        nrmt = np.linalg.norm(t)
        [x,y] = np.meshgrid(np.array([i+1 for i in range(int(szBB[1]))]), np.array([i+1 for i in range(int(szBB[0]))]))
        d = np.zeros(szBB)
        d.fill(float("inf"))
        
        if nrmt == 0:
            # Use distance to point q0
            d = np.sqrt( (x - q0[0])**2 +(y - q0[1])**2)
            idx = (d <= rad)
        else:
            # Use distance to segment q0, q1
            t = t/nrmt
            n = [t[1], -t[0]]
            tmp = t[0] * (x - q0[0]) + t[1] * (y - q0[1])
            idx = (tmp >= 0) & (tmp <= nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = abs(n[0] * (x[np.where(idx)] - q0[0]) + n[1] * (y[np.where(idx)] - q0[1]))
            idx = (tmp < 0)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q0[0])**2 +(y[np.where(idx)] - q0[1])**2)
            idx = (tmp > nrmt)
            if np.any(idx.flatten('F')):
                d[np.where(idx)] = np.sqrt( (x[np.where(idx)] - q1[0])**2 +(y[np.where(idx)] - q1[1])**2)

            #Pixels within crop box to paint have distance <= rad
            idx = (d <= rad)
        #Mark the pixels
        if np.any(idx.flatten('F')):
            xy = (bb0[1]-1+y[np.where(idx)] + sizeIm[0] * (bb0[0]+x[np.where(idx)]-2)).astype(int)
            sz = mrkd.shape
            m = mrkd.flatten('F')
            m[xy-1] = val
            mrkd = m.reshape(mrkd.shape[0], mrkd.shape[1], order = 'F')

            '''
            row = 0
            col = 0
            for i in range(len(m)):
                col = i//sz[0]
                mrkd[row][col] = m[i]
                row += 1
                if row >= sz[0]:
                    row = 0
            '''
            
            
            
    return mrkd
    

def paintStroke(canvas, x, y, p0, p1, colour, rad):
    # Paint a stroke from pixel p0 = (x0, y0) to pixel p1 = (x1, y1)
    # on the canvas (ny x nx x 3 double array).
    # The stroke has rgb values given by colour (a 3 x 1 vector, with
    # values in [0, 1].  The paintbrush is circular with radius rad>0
    sizeIm = canvas.shape
    sizeIm = sizeIm[0:2]
    idx = markStroke(np.zeros(sizeIm), p0, p1, rad, 1) > 0
    # Paint
    if np.any(idx.flatten('F')):
        canvas = np.reshape(canvas, (np.prod(sizeIm),3), "F")
        xy = y[idx] + sizeIm[0] * (x[idx]-1)
        canvas[xy-1,:] = np.tile(np.transpose(colour[:]), (len(xy), 1))
        canvas = np.reshape(canvas, sizeIm + (3,), "F")
    return canvas


def gradientCalculator(im, sigma):
    '''
    Helper function that calculates the array of gradient directions for
    an input image. This will be used in Part 5 to make the paint strokes
    orthogonal to edge directions.
    '''
    imin = im.copy() * 255.0
    # Create the gauss kernel for blurring the input image
    # It will be convolved with the image
    # wsize should be an odd number
    wsize = 5
    gausskernel = gaussFilter(sigma, window = wsize)
    # fx is the filter for vertical gradient
    # fy is the filter for horizontal gradient
    # Please note the vertical direction is positive X

    fx = createFilter([0,  1, 0,
                       0,  0, 0,
                       0, -1, 0])
    fy = createFilter([ 0, 0, 0,
                       -1, 0, 1,
                        0, 0, 0])

    imout = conv(imin, gausskernel, 'valid')
    # print "imout:", imout.shape
    gradxx = conv(imout, fx, 'valid')
    gradyy = conv(imout, fy, 'valid')

    gradx = np.zeros(im.shape)
    grady = np.zeros(im.shape)
    padx = (imin.shape[0] - gradxx.shape[0]) / 2.0
    pady = (imin.shape[1] - gradxx.shape[1]) / 2.0
    gradx[padx:-padx, pady:-pady] = gradxx
    grady[padx:-padx, pady:-pady] = gradyy
    
    # Net gradient is the square root of sum of square of the horizontal
    # and vertical gradients

    grad = hypot(gradx, grady)
    theta = arctan2(grady, gradx)
    
    # clear gradients below threshold 5 (for Part5)
    xx, yy = where(grad < 5)
    theta[xx, yy] = 0
    grad[xx, yy] = 0

    return theta

    
def main_execute_final(name_of_file):
    '''
    Function to produce final output (i.e. Part 6)
    '''
    # Read image and convert it to double, and scale each R,G,B
    # channel to range [0,1].
    imRGB = array(Image.open(name_of_file))
    imRGB = double(imRGB) / 255.0
    plt.clf()
    plt.axis('off')

    # make image into monochrome using appropriate intensities
    r, g, b = imRGB[:,:,0], imRGB[:,:,1], imRGB[:,:,2]
    monochrome = 0.30*r + 0.59*g + 0.11*b
    
    # create Canny edgels map using sigma=4.0 and custom-set thresholds
    canny_map = canny(monochrome, 4.0, 18, 5)

    sizeIm = imRGB.shape
    sizeIm = sizeIm[0:2]
    # Set radius of paint brush and half length of drawn lines
    rad = 1
    
    # Set up x, y coordinate images, and canvas.
    [x, y] = np.meshgrid(np.array([i+1 for i in range(int(sizeIm[1]))]), np.array([i+1 for i in range(int(sizeIm[0]))]))
    canvas = np.zeros((sizeIm[0],sizeIm[1], 3))
    canvas.fill(-1) ## Initially mark the canvas with a value out of range.
    # Negative values will be used to denote pixels which are unpainted.
    
    # Random number seed
    np.random.seed(29645)
    
    # Orientation of paint brush strokes
    theta = 2 * pi * np.random.rand(1,1)[0][0]
    # Set vector from center to one end of the stroke.
    #delta = np.array([cos(theta), sin(theta)])
    
    thetaArray = gradientCalculator(monochrome, 4.0)
    
    time.time()
    time.clock()
    
    p = 0
    while len(np.where(canvas==-1)[1])>0:
        # half length of drawn lines, in left and right directions
        left, right = 5, 5
    
        # find all unfilled pixels
        unfilled = np.where(canvas==-1)
        
        # choose random pixel from the unfilled pixels, and make it the
        # stroke centre
        random_pixel = np.random.randint(0, len(unfilled[0]))
        cntr = array([unfilled[1][random_pixel], unfilled[0][random_pixel]])
        
        # get theta from thetaArray, and add 90 degrees (for orthogonal)
        theta = thetaArray[cntr[1], cntr[0]] + 90
        delta = np.array([cos(theta), sin(theta)])
        
        # check if the cntr pixel is an edge in the canny edgel map.
        # if so, set half lengths to 0 (for brush stroke of length = 0)
        if canny_map[cntr[1], cntr[0]] == 1:
            left = 0
            right = 0
        else:
             # loop to find the endpoint for painting in the right direction
            for right_distance in range(0, right):
                # moving in the direction of delta
                right_endpoint = cntr + right_distance * delta 
                
                if (canny_map[right_endpoint[1], right_endpoint[0]] == 1 
                        or right_endpoint[0] <= 0 
                        or right_endpoint[1] >= canny_map.shape[0] - 1):
                    right = right_distance
                    
                    # must break the loop when the endpoint is found
                    break
            
            # loop to find the endpoint for painting in the left direction
            for left_distance in range(1, left):
                # moving in the opposite direction to delta
                left_endpoint = cntr - left_distance * delta
                
                # walks until the nearest edgel or until the edge of the image.
                # if either of these is found, that is the left endpoint, so
                # left half length = the distance walked
                if (canny_map[left_endpoint[1], left_endpoint[0]] == 1 
                        or left_endpoint[0] >= canny_map.shape[1] - 1 
                        or left_endpoint[1] <= 0):
                    left = left_distance
                    
                    # must break the loop when the endpoint is found
                    break
        theta = thetaArray[cntr[1], cntr[0]]
        
        # Grab colour from image at center position of the stroke.
        colour = np.reshape(imRGB[cntr[1]-1, cntr[0]-1, :],(3,1))        

        # Add the stroke to the canvas
        nx, ny = (sizeIm[1], sizeIm[0])
        length1, length2 = (right, left)
        
        # randomly perturb intensity, colour and angle for Part6, using the 
        # values mentioned in Litwinowicz's paper
        colour *= random.uniform(0.85, 1.15)
        colour += random.uniform(-15/255.0, 15/255.0)
        theta += random.uniform(-15, 15)
        
        delta = np.array([cos(theta), sin(theta)])
        
        if (abs(delta[0]) > abs(delta[1])):
            new_del = delta / abs(delta[0])
            canvas = paintStroke(canvas, x, y, cntr+1 - np.round(new_del * length2), cntr+1 + np.round(new_del * length1), colour, rad)
        else:
            new_del = delta / abs(delta[1])
            canvas = paintStroke(canvas, x, y, cntr+1 - np.round(new_del * length2), cntr+1 + np.round(new_del * length1), colour, rad)
        print 'stroke', p
        p += 1
        
    print name_of_file + " final rendering done!"
    time.time()
    
    canvas[canvas < 0] = 0.0
    return canvas


if __name__ == "__main__":
    # call appropriate function and produce painterly output for both images 
    canvas_1_final = main_execute_final('orchid.jpg')
    colorImSave('orchid_final_output.png', canvas_1_final)
    
    canvas_2_final = main_execute_final('RV.jpg')
    colorImSave('RV_final_output.png', canvas_2_final)