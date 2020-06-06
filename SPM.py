from __future__ import division

from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from numpy.lib.stride_tricks import as_strided

def load_names(mypath, folders):
    image_files = {}
    for name in folders:
        image_files[name] = [f for f in listdir(mypath + name + '/')]

    return image_files

def pool2d(A, kernel_size, stride, padding):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    return A_w.mean(axis=(1,2)).reshape(output_shape)

#input IMAGE1, IMAGE2, DEGREE = 1 or 2 or 4
def pyr_match(image_1, image_2, degree):
    #images to arrays

    OG_image_1_array = np.array(image_1)
    OG_image_2_array = np.array(image_2)
    #insures correct dimensions
    if len(OG_image_1_array.shape) >= 3:
        OG_image_1_array = OG_image_1_array[:,:,0]
    if len(OG_image_2_array.shape) >= 3:
        OG_image_2_array = OG_image_2_array[:,:,0]
    #applies SPM using modified pooling layer
    pym_vals = []
    total = 1
    count = 1
    the_max = []
    while count <= degree: 
        image_1_array = pool2d(OG_image_1_array, total, total, 0)
        image_2_array = pool2d(OG_image_2_array, total, total, 0)
        the_max.append(total ** 2)
        total = 2 ** count
        count += 1
        #Where More than Half the Squares are 0
        condition_1_1 = (image_1_array < 127)
        condition_1_2 = (image_2_array < 127)
        #Where Half or More squares are 255
        condition_2_1 = (image_1_array > 127)
        condition_2_2 = (image_2_array > 127)
        part1 = np.where(condition_1_1 & condition_1_2)
        part2 = np.where(condition_2_1 & condition_2_2)
        pym_vals.append(len(part1[0]) + len(part2[0]))
    return pym_vals, the_max

def pyr_all(folders1, folders2, mypath):
    for folder1 in folders2:
        if folder1 in folders2:
            folders2.remove(folder1)
        for folder2 in folders2:
            print ''            
            print folder1, folder2
            pics1 = [f for f in listdir(mypath + folder1 + '/')]
            pics2 = [f for f in listdir(mypath + folder2 + '/')]
            for pic in pics1:
                if pic in pics2:
                    pic1 = Image.open('/home/tseibel/Desktop/SPM/labels/' + folder1 + '/' + pic)
                    pic2 = Image.open('/home/tseibel/Desktop/SPM/labels/' + folder2 + '/' + pic)
                    output = pyr_match(pic1, pic2, 10)
                    print pic
                    the_alg(output)


def the_alg(output):
    output[1].reverse()
    answer = []
    for each_list in output:
        L = len(output[0]) - 1
        index = 0
        out = 0
        for x in each_list:
            #print 'L: ', L
            #print 'x: ', x
            #print 'index: ', index
            value =  ((1 / 2) ** ( index ) )  * ( x / ( 2 ** ( 2 * ( L - index ) ) ) )
            #value =  ((1 / 2) ** ( L - index ) )  * ( x / ( 2 ** ( 2 * ( L - index ) ) ) )
            #value = ( 1 / ( 2 ** ( L - index ) ) ) * ( x / ( 2 ** ( 2 * ( L - index ) ) ) )
            #print 'value :', value
            out += value
            index +=1
        answer.append(out)
    print answer[0]
    #print answer[1]
    #print answer[0]/answer[1]
    

mypath = '/home/tseibel/Desktop/SPM/labels/'
folders = [f for f in listdir(mypath)]

#test images
im = Image.open('/home/tseibel/Desktop/SPM/labels/train-labels/13.tif')
im2 = Image.open('/home/tseibel/Desktop/SPM/labels/unet-labels/13.tif')

pyr_all(folders, folders, mypath)
