import os
import sys
import cv2
import numpy as np
import numpy.matlib as nm
import scipy.misc
import GoogleSV as GSV

def WriteDepthMap(panoid, f_name):
    depth = GetDepthMap(panoid = panoid)
    DepthToNpy(depth, f_name)

def GetDepthMap(panoid = None, lat = None, lon = None, radius = 10):
    if panoid is None:
        panoid = GSV.GetIDByLoc(lat = lat, lon = lon, radius = radius)
        if panoid is None:
            return None
    metadata = GSV.GetPanoramaMetadata(panoid = panoid)
    return CreateDepthMap(metadata)

def CreateSphericalBuf(height, width):
    h = np.arange(height, dtype=np.float32)
    theta = (height - 1 - h) / height * np.pi
    sin_theta = nm.repmat(np.sin(theta), width, 1).T
    cos_theta = nm.repmat(np.cos(theta), width, 1).T
    w = np.arange(width, dtype=np.float32)
    phi = (width - 1 - w) / width * 2 * np.pi + np.pi / 2
    sin_phi = nm.repmat(np.sin(phi), height, 1)
    cos_phi = nm.repmat(np.cos(phi), height, 1)
    v = np.zeros((height, width, 3))
    v[:, :, 0] = sin_theta * cos_phi
    v[:, :, 1] = sin_theta * sin_phi
    v[:, :, 2] = cos_theta
    # v is the spherical for google raw data
    theta = nm.repmat(theta, width, 1).T
    phi = nm.repmat(phi, height, 1)
    #print theta
    #x = phi - 3.0/2 * np.pi
    #print phi / np.pi
    return v

def SphericalToEqirectangular(depth):
    shape = depth.shape
    height = shape[0]
    width = shape[1]
    center_of_row = (height - 1) / 2.0
    center_of_col = (width - 1) / 2.0
    lat = (np.arange(height, dtype = np.float32) - center_of_row) / height * np.pi
    lon = (np.arange(width, dtype = np.float32) - center_of_col) / width * 2 * np.pi
    x = lon / np.pi
    sin_of_lat = np.sin(lat)
    y = np.log( (1 + sin_of_lat) / (1 - sin_of_lat) ) / (4 * np.pi)
    #y_map = np.matlib.repmat(y, width, 1).T
    #x_map = np.matlib.repmat(x, height, 1)
    x = np.round(np.interp(x, [min(x), max(x)], [0, width-1]))
    y = np.round(np.interp(y, [min(y), max(y)], [0, height-1]))

    depth_map = np.zeros([height, width], dtype=np.float32)
    mesh = np.int32(np.meshgrid(y, x))
    #depth_map[mesh[0], mesh[1]] = depth[mesh[0], mesh[1]]
    depth_map = depth[mesh[0], mesh[1]].T
    return depth_map
def CreateDepthMap(data):
    numPlanes, height, width = [data.DepthHeader['numPlanes'],
                                data.DepthHeader['panoHeight'],
                                data.DepthHeader['panoWidth']]
    # Because indice 0 is always sky, we can ignore it
    plane_indices = np.array(data.DepthMapIndices)
    spherical_buf = CreateSphericalBuf(height, width).reshape([height * width, 3])
    depth_map = np.zeros((height * width), dtype=np.float32) + np.inf
    #depth_map = np.zeros((height,width), dtype=np.float32)
    space1 = np.linspace(1, 255, numPlanes)
    #space2 = np.linspace(255, 1, numPlanes)
    #space3 = np.linspace(1, 255, numPlanes)
    for plane_index in range(1, numPlanes):

        plane = data.DepthMapPlanes[plane_index]
        if plane['d'] < 0:
            print 'GG'
        normal_vector = np.array([plane['nx'], plane['ny'], plane['nz']], dtype=np.float32)
        depth_map[plane_indices==plane_index] =  \
        np.abs(plane['d'] / np.dot(spherical_buf[plane_indices==plane_index, :], normal_vector))
        #depth_map[plane_indices==plane_index] = space1[plane_index]
        #depth_map[plane_indices==plane_index, :] = np.array([space1[plane_index], space2[plane_index], space3[plane_index]])
    return depth_map.reshape([height, width])
def DepthToImage(depth):
    '''
    depth_map = depth * 255 / 50
    depth_map[np.nonzero(np.isnan(depth_map))] = 255
    depth_map[np.nonzero(depth_map > 255)] = 255
    depth_map /= 255
    '''
    #depth_map = depth
    #'''
    depth[np.isinf(depth)] = 3000
    depth_map = np.abs(depth) + 1
    #print np.max(np.log(depth_map))
    sc = 1 / np.max(np.log(depth_map))
    depth_map = sc * np.log(depth_map) 
    #depth_map = depth / np.amax(depth) * 255
    #'''
    scipy.misc.imsave('depth.png', depth_map)

def DepthToNpy(depth, store_name):
    np.save(store_name, depth)

if __name__ == '__main__':
    #CreateSphericalBuf(256, 512)
    #exit()
    #test_id = 'Bu_I2xKvp4GVH97EB4obLQ'
    test_id = 'pRANHWx41ZnTz5PsZs6JpA'
    data  = GSV.GetPanoramaMetadata(panoid = test_id)
    a = CreateDepthMap(data)
    DepthToNpy(a, './G.npy')
