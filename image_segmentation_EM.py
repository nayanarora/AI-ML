## Initiializations and inputs

import matplotlib.pyplot as plt 
import os
from os.path import join 
import numpy as np
from PIL import Image
import matplotlib.image as mpimg 
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.color import label2rgb 
from skimage.filters import gaussian 
from sklearn.cluster import KMeans

plt.close('all')
clear = lambda: os.system('clear') 
clear()
np.random.seed(110)
colors = [[1,0,0],[0,1,0],[0,0,1],[0,0.5,0.5],[0.5,0,0.5]]
imgNames = ['water_coins','jump','tiger']#{'balloons', 'mountains', 'nature', 'ocean', 'polarlights'};
segmentCounts = [2,3,4,5]


## define functions required for processing
def normalize(img):
    """ min-max normalization """
    h = img.shape[0]
    w = img.shape[1]
    nc = img.shape[2]
    new_img = np.zeros((h,w,nc),dtype='float') 
    for cc in range(nc):
        new_img[:,cc] = (img[:,cc] - img[:,cc].min()) / (img[:,cc].max() - img[:,cc].min())
    return(new_img)

def im2double(im): 
    try:
        info = np.iinfo(im.dtype) # Get the data type of the input image
        return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype
    
    except ValueError:
        print('Image is of type Double-- No conversion required') 
        return im.astype(np.float)
    
from posix import setpgrp
for imgName in imgNames:
    for SegCount in segmentCounts:
        # Load the image using the Matplotlib image library.
        # Images are in the 'Input' directory. Use 'join' command to include folder
        # path to access image file

        
        img = mpimg.imread(''.join(['/Users/nayanarora/Desktop/AIT/Assignment1/AIT_Ass1_Qn_&_Data/Input/',imgName,'.png']))
        """ Read Image using mplib library-- 2 points """

        print('Using Matplotlib Image Library: Image is of datatype ',img.dtype,'and size ',img.shape) # Image is of type float
        # Load image using the Pillow or the Python Imaging Library

        #print(img.shape[0])
        img = Image.open(''.join(['/Users/nayanarora/Desktop/AIT/Assignment1/AIT_Ass1_Qn_&_Data/Input/',imgName,'.png'])) 
        """ Read Image using PILLOW-- 3 points"""
        
        print('Using Pillow (Python Image Library): Image is of datatype ',img.mode,'and size ',img.size) # Image is of type uint8
        #%% %Define Parameters
        nSegments = SegCount # of color clusters in image
        #width*height = img.shape
        #print(img.height)
        nPixels = img.width * img.height
        """ Compute number of image pixels from image dimensions-- 2 points"""; #Image can be represented by a matrix of size nPixels*nColors
        
        maxIterations = 20; #maximum number of iterations allowed for EM algorithm.
        nColors = 3;
        #%% Determine the output path for writing images to files
        outputPath = join(''.join(['Output/',str(SegCount), '_segments/', imgName , '/']));
        
       
        if not(os.path.exists(outputPath)): 
            os.makedirs(outputPath)
            #mpimg.imsave('outputPath/0.png',img)
            #mpimg.imsave(''.join([outputPath,str(iteration+1),'.png']),seglabelsRGB)
            mpimg.imsave(''.join([outputPath,str(iteration),'.png']), img)
        """ save input image as *0.png* under outputPath using imsave command in mpimg library-- 3 points""" #save using Matplotlib image library
       
        #%% Vectorizing image for easier loops- done as im(:) in Matlab
        pixels = img
        pixels = np.array(img)

        #img_ar = np.array(img)
        pixels = pixels.reshape(nPixels, nColors, 1)
        pixels = pixels.astype(float)
        """ Reshape pixels as a nPixels X nColors X 1 matrix-- 5 points"""
        #%%
        """ Initialize pi (mixture proportion) vector and mu matrix
            (containing means of each distribution)
            Vector of probabilities for segments... 1 value for each segment.
            The idea behind image generation goes like this...
            When the image was generated, color was determined for each pixel
            by selecting a value from one of "n" normal distributions
            corresponding to the "n" color segments. Each value
            in this vector corresponds to the probability that a
            particular segment or normal distribution was chosen."""
        
        """ Initial guess for pi's is 1/nSegments. Small amount of noise
            added to slightly perturb GMM coefficients from the initial guess"""

        pi = 1/nSegments*(np.ones((nSegments, 1),dtype='float')) 
        increment = np.random.normal(0,.0001,1)
        for seg_ctr in range(len(pi)):
            if(seg_ctr%2==1):
                pi[seg_ctr] = pi[seg_ctr] + increment
            else:
                pi[seg_ctr] = pi[seg_ctr] - increment
        #%%
        """Similarly, the initial guess for the segment color means would be a 
           perturbed version of [mu_R, mu_G, mu_B],
           where mu_R, mu_G, mu_B respectively denote the means of the
           R,G,B color channels in the image.
           mu is a nSegments X nColors matrix,(seglabels*255).np.asarray
           (in int format) where each matrix row denotes mean RGB color
           for a particular segment"""

        mu = 1/nSegments*(np.ones((nSegments, nColors), dtype='float'))
        #noise = np.random.normal(0, 0.0001, size = (nSegments, nColors))
        #mu = mu + noise
        """Initialize mu to 1/nSegments*['ones' matrix
                (whose elements are all 1) of size nSegments X nColors] -- 5 points""" 
                #for even start #add noise to the initialization (but keep it unit)
        
        for seg_ctr in range(nSegments): 
            if(seg_ctr%2==1):
                increment = np.random.normal(0,.0001,1) 
            for col_ctr in range(nColors):
                if(seg_ctr%2==1):
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) + increment
                else:
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) - increment;

        #%% EM-iterations begin here. Start with the initial (pi, mu) guesses
        mu_last_iter = mu;
        pi_last_iter = pi;
        for iteration in range(maxIterations): 
            """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ----------------- E-step -----estimating likelihoods and membership weights (Ws)
              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
            print(''.join(['Image: ',imgName,' nSegments: ',str(nSegments),'iteration: ',str(iteration+1), ' E-step']))
            # Weights that describe the likelihood that pixel denoted by "pix_import scipy.miscctr" belongs to a color cluster "seg_ctr"
            Ws = np.ones((nPixels,nSegments),dtype='float') # temporarily reinitialize all weights to 1, before they are recomputed
            
            """ logarithmic form of the E step."""

            for pix_ctr in range(nPixels): 
                # Calculate Ajs
                logAjVec = np.zeros((nSegments,1),dtype='float') 
                for seg_ctr in range(nSegments):
                    x_minus_mu_T  = np.transpose(pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T)
                    x_minus_mu    = ((pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T))
                    logAjVec[seg_ctr] = np.log(pi[seg_ctr]) - .5*(np.dot(x_minus_mu_T,x_minus_mu))
                # Note the max
                logAmax = max(logAjVec.tolist())
                # Calculate the third term from the final eqn in the above link
                thirdTerm = 0;
                for seg_ctr in range(nSegments):
                    thirdTerm = thirdTerm + np.exp(logAjVec[seg_ctr]-logAmax)

                # Here Ws are the relative membership weights(p_i/sum(p_i)),␣ ,→but computed in a round-about way
                
                for seg_ctr in range(nSegments):
                    logY = logAjVec[seg_ctr] - logAmax - np.log(thirdTerm) 
                    Ws[pix_ctr][seg_ctr] = np.exp(logY)
            """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % -----------------   M-step  --------------------
              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
            

            print(''.join(['Image: ',imgName,' nSegments: ',str(nSegments),'iteration: ',str(iteration+1), ' M-step: Mixture coefficients']))
            #%% temporarily reinitialize mu and pi to 0, before they are recomputed
            mu = np.zeros((nSegments,nColors),dtype='float') # mean color for each segment
            pi = np.zeros((nSegments,1),dtype='float') #mixture coefficients 
            
            for seg_ctr in range(nSegments):
                denominatorSum = 0;
                for pix_ctr in range(nPixels):
                  mu[seg_ctr,:]=mu[seg_ctr,:]+np.dot(pixels[pix_ctr,:].flatten(),Ws[pix_ctr, seg_ctr])
                  """Update RGB color vector of mu[seg_ctr] as current mu[seg_ctr] + pixels[pix_ctr,:] times Ws[pix_ctr,seg_ctr] -- 5 points"""
                  denominatorSum = denominatorSum + Ws[pix_ctr][seg_ctr]

                """Compute mu[seg_ctr] and denominatorSum directly without the 'for loop'-- 10 points.
                    If you find the replacement instruction, comment out the for loop with your solution"
                    Hint: Use functions squeeze, tile and reshape along with sum"""
                ## Update mu
                mu[seg_ctr,:] = mu[seg_ctr,:]/ denominatorSum;

                ## Update pi
                #pi[seg_ctr] = np.sum([Ws[:,seg_ctr]]) / nPixels;
                pi[seg_ctr] = denominatorSum / nPixels; #sum of weights (each weight is a probability) for given segment/total num of pixels 
            
            #NumPy function can be used to calculate the weighted sum of pixel values along with the sum of the membership weights for each color class instead of using the for loop. This helps increase efficiency.
            #denominatorSum = np.sum(Ws[:, seg_ctr])
            #mu[seg_ctr, :] = np.sum(np.tile(Ws[:, seg_ctr][:, np.newaxis], (1, nColors)) * pixels[:,:,0], axis=0) / denominatorSum
            #mu[seg_ctr, :] = np.sum(np.tile(np.reshape(Ws[:, seg_ctr], (-1, 1)), (1, nColors)) * pixels, axis=0) / denominatorSum
            
            print(np.transpose(pi))
            muDiffSq = np.sum(np.multiply((mu - mu_last_iter),(mu - mu_last_iter)))
            piDiffSq = np.sum(np.multiply((pi - pi_last_iter),(pi - pi_last_iter)))

            if (muDiffSq < .0000001 and piDiffSq < .0000001): #sign of convergence
                print('Convergence Criteria Met at Iteration: ',iteration, 'Exiting code')
                break;
        
            mu_last_iter = mu;
            pi_last_iter = pi;

            ##Draw the segmented image using the mean of the color cluster as the
            ## RGB value for all pixels in that cluster.
        
            segpixels = np.array(pixels)
            cluster = 0
            for pix_ctr in range(nPixels):
                cluster = np.where(Ws[pix_ctr,:] == max(Ws[pix_ctr,:]))
                vec = np.squeeze(np.transpose(mu[cluster,:]))
                segpixels[pix_ctr,:] = vec.reshape(vec.shape[0],1)
                
                """ Save segmented image at each iteration. For displaying consistent image clusters, it would be useful to blur/smoothen the segpixels image using a Gaussian filter.
                    Prior to smoothing, convert segpixels to a Grayscale image, and convert the grayscale image into clusters based on pixel intensities""" 
                
            #segpixels = np.reshape(segpixels,(img.shape[0],img.shape[1],nColors)) ##reshape segpixels to obtain R,G, B image
                
            segpixels = np.reshape(segpixels,(img.height, img.width, nColors))

            greyPixels = rgb2gray(segpixels) #convert to grayscale
            #greyPixels = (greyPixels*255).astype(np.uint8)
            #greyPixels = greyPixels.astype(np.float) / 255.0 #normalise image to 0 - 1 
            greyPixels = np.uint8(segpixels)
            greyPixels = img_as_ubyte(greyPixels) #convert image to uint8
            #print(greyPixels)
            #print(segpixels)
            # segpixels = np.reshape(segpixels,(img.height, img.width, nColors))

            #segpixels = rgb2gray(segpixels) #convert to grayscale
            #segpixels = (segpixels*255).astype(np.uint8)
            #segpixels = np.uint8(segpixels)
            #segpixels = rgb2gray(segpixels)
            #segpixels = img_as_ubyte(segpixels)
            #segpixels = segpixels.astype(float)/255.0
            #print(segpixels)
            """convert segpixels to uint8 gray scale image and convert to grayscale-- 5 points""" #convert to grayscale
            #segpixels = np.uint8(np.mean(segpixels, axis=2))
            
            #onedPixelValues = greyPixels.flatten() #normalize to 1d array
            segpixels = segpixels.flatten()
            onedPixelValues = greyPixels.reshape(-1, 1) #reshape 1d array and transform it into a column vector                
            kmeans = KMeans(n_clusters=nSegments, n_init = 4).fit(onedPixelValues)
            """ Use kmeans from sci-kit learn library to cluster pixels in gray scale segpixels image to *nSegments* clusters-- 10 points"""
            labels = kmeans.labels_
            seglabels = labels.reshape(greyPixels.shape) 
            #print(seglabels)
            """ reshape kmeans.labels_ output of kmeans to have the same size as segpixels -- 5 points"""

            
            seglabels =np.clip(seglabels, 0, 1).astype(float)
              
            seglabelSmooth = gaussian(seglabels, sigma = 2).astype(float) #.astype(float), sigma = 2)

                #seglabelSmooth = np.clip(seglabelSmooth, 0 ,1)
                
            seglabelsRGB = label2rgb(seglabelSmooth, bg_label=0, bg_color=(0,0,0))
            #print('1234567890')
            #print(seglabelsRGB)
            seglabelsRGB = seglabelsRGB.astype(float)/255.0 #(np.float32)/255.0
            # print(seglabelsRGB)
            "Use np.clip, Gaussian smoothing with sigma =2 and label2rgb functions to smoothen the seglabels image, and output a float RGB image with pixel values between [0--1]-- 20 points"""

            mpimg.imsave(''.join([outputPath,str(iteration+1),'.png']),seglabelsRGB) #save the segmented output
            #print('heloooo8')
            """ Display the 20th iteration (or final output in case of convergence) segmentation 
                    images with nSegments = 2,3,4,5 for the three images--
                    this will be a 3 row X 4 column image matrix-- 15 points"""

            """ Comment on the results obtained, and discuss your understanding of the
                    Image Segmentation problem in general-- 15 points """
# %%
