#! /usr/bin/env python
import subprocess
import cv2
import os
import numpy as np
import urllib2
import GoogleSV as GSV

test_id = 'Xql1WdJlK_2QUM4CV8Zmdg'
BaseUri = 'http://maps.google.com/cbk'
zoom_lv = 4
ext = '.jpg'

def GetPanoByID_full(ID, DIR):
    if not os.path.isdir(DIR):
        return False
    subprocess.call('mkdir -p buffer_%s'%(ID), shell=True)

    row_range = range(7)
    col_range = range(13)
    for row in row_range:
        for col in col_range:
            data = GSV.GetPanoramaTile(ID, zoom_lv, col , row)
            f = open('buffer_%s/image_%d_%d.jpg'%(ID, row, col), 'wb')
            f.write(data)
            f.close()
    img = np.zeros([512*7, 512*13, 3], dtype=np.uint8)
    for row in row_range:
        for col in col_range:
            patch = cv2.imread('buffer_%s/image_%d_%d.jpg'%(ID, row, col))
            img[row*512:(row+1)*512, col*512:(col+1)*512, :] = patch
    cv2.imwrite('%s/pano_%s.jpg'%(DIR, ID), img[:-256, :])

    subprocess.call('rm -r buffer_%s'%(ID), shell=True)
            

def GetPanoByID_0_180(ID, DIR):
    if not os.path.isdir(DIR):
        return False
    try:
        if not os.path.isdir('./buffer_%s'%ID):
            subprocess.call('mkdir -p buffer_%s/tmp'%ID, shell=True)
        else:
            subprocess.call('rm -r buffer_%s'%ID, shell=True)
            subprocess.call('mkdir -p buffer_%s/tmp'%ID, shell=True)
        row_range = [2,3,4]
        col_range = [5,6,7]
        index = 0
        for row in row_range:
            for col in col_range:
                data = GSV.GetPanoramaTile(ID, zoom_lv, col, row)
                #subprocess.call("wget '%s'  -O buffer_%s/tmp/image_%.2d%s"%(url, ID, index, ext), shell=True)
                f = open('buffer_%s/tmp/image_%.2d%s'%(ID,index,ext), 'wb')
                f.write(data)
                f.close()
                index += 1
        lst = os.listdir('buffer_%s/tmp/'%ID)
        lst.sort()
        output = np.zeros([1536, 1536, 3], dtype = np.uint8)
        begin_row = 0
        begin_col = 0
        index = 0
        for row in range(0,3):
            begin_col = 0
            for col in range(0,3):
                img = cv2.imread('buffer_%s/tmp/'%ID+lst[index])
                output[begin_row:begin_row+512, begin_col: begin_col+512] = img
                begin_col += 512
                index += 1
            begin_row += 512
        cv2.imwrite('buffer_%s/pano_%s_0.jpg'%(ID,ID), output[208:928, 128:1408])
        subprocess.call('rm buffer_%s/tmp/*'%ID, shell=True)
        row_range = [2,3,4]
        col_range = [11,12,0,1]
        #col_range = [1, 0 , 12, 11]
        index = 0
        for row in row_range:
            for col in col_range:
                data = GSV.GetPanoramaTile(ID, zoom_lv, col, row)
                #subprocess.call("wget '%s'  -O buffer_%s/tmp/image_%.2d%s"%(url, ID, index, ext), shell=True)
                f = open('buffer_%s/tmp/image_%.2d%s'%(ID,index,ext), 'wb')
                f.write(data)
                f.close()
                index += 1
        lst = os.listdir('buffer_%s/tmp/'%ID)
        lst.sort()
        output = np.zeros([2048, 2048, 3], dtype = np.uint8)
        begin_row = 0
        begin_col = 0
        index = 0
        for row in range(0,3):
            begin_col = 0
            for col in range(0,4):
                img = cv2.imread('buffer_%s/tmp/'%ID+lst[index])
                output[begin_row:begin_row+512, begin_col: begin_col+512] = img
                begin_col += 512
                index += 1
            begin_row += 512
        cv2.imwrite('buffer_%s/pano_%s_180.jpg'%(ID,ID), output[208:928, 384:1664])
        subprocess.call('mv buffer_%s/*.jpg %s'%(ID,DIR), shell=True)
        subprocess.call('rm -r buffer_%s/'%ID, shell=True)
        return True
    except:
        subprocess.call('rm -r buffer_%s/'%ID, shell=True)
        return False

#GetPanoByID_0_180(test_id,'.') 
#GetPanoByID_full(test_id,'.') 
