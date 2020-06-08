from __future__ import division

import itertools
import csv
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
    all_dict = {}
    diff_all_dict = {}
    for folder1 in folders2:
        if folder1 in folders2:
            folders2.remove(folder1)
        for folder2 in folders2:
            new_dict = {}
            diff_new_dict = {}
            #print ''            
            #print folder1, folder2
            #names.append([folder1,folder2])
            pics1 = [f for f in listdir(mypath + folder1 + '/')]
            pics2 = [f for f in listdir(mypath + folder2 + '/')]
            for pic in pics1:
                if pic in pics2:
                    pic1 = Image.open('/home/tseibel/Desktop/SPM/labels/' + folder1 + '/' + pic)
                    pic2 = Image.open('/home/tseibel/Desktop/SPM/labels/' + folder2 + '/' + pic)
                    answer = pyr_match(pic1, pic2, 10)
                    #print pic
                    new_dict[pic], diff_new_dict[pic] = the_alg(answer[0])
            all_dict[folder1+folder2] = new_dict
            diff_all_dict[folder1+folder2] = diff_new_dict

    return all_dict, diff_all_dict
                    


def the_alg(output):
    answer = []
    diff_answer =[]
    L = len(output) - 1
    index = 0
    out = 0
    diff_out = 0
    for x in output:
        diff_value = ((1 / 2) ** ( index ) )  * (1 - ( x / ( 2 ** ( 2 * ( L - index ) ) ) ) )
        value =  ((1 / 2) ** ( index ) )  * ( x / ( 2 ** ( 2 * ( L - index ) ) ) )
        #value =  ((1 / 2) ** ( L - index ) )  * ( x / ( 2 ** ( 2 * ( L - index ) ) ) )
        #value = ( 1 / ( 2 ** ( L - index ) ) ) * ( x / ( 2 ** ( 2 * ( L - index ) ) ) )
        out += value
        diff_out += diff_value
        index +=1
    answer.append(out)
    diff_answer.append(diff_out)
    #print answer[0]
    return answer, diff_answer
    #print answer[0]/answer[1]
    
def dict_To_CSV(file_name, answers, diffs):
    images = []
    image_dict = {}
    diff_image_dict = {}
    for the_dict in answers.values():
        for image in the_dict.keys():
            if image not in images:
                image_dict[image] = []
                diff_image_dict[image] = []
                images.append(image)
    for session in answers:
        for image in images:
            if image in answers[session]:
                diff_image_dict[image].append(diffs[session][image])
                image_dict[image].append(answers[session][image])
            else:
                image_dict[image].append('NaN')
                diff_image_dict[image].append('NaN')
    with open(file_name, 'w') as csv_file:
        csvwriter = csv.writer(csv_file, delimiter='\t')
        csvwriter.writerow(['image name','ws/unet','train/unet','train/ws','ws/unet','train/unet','train/ws'])
        for image in image_dict:
            csvwriter.writerow([image, image_dict[image], diff_image_dict[image]])


mypath = '/home/tseibel/Desktop/SPM/labels/'
folders = [f for f in listdir(mypath)]

answers_dict, diff_dict = pyr_all(folders, folders, mypath)
 
dict_To_CSV('test.csv', answers_dict, diff_dict)
