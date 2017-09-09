#! /usr/bin/env python

import sys
sys.path.append('/media/external2/Dash_Cam_Point_Align')
from geopy.distance import VincentyDistance as VD
import numpy as np

step_latlon = 0.0009 # 100 m
step_ten_meter = step_latlon / 10 # step_latlon/10-->every ten meter
RADIUS = 3

def GetCircleBound(center): # lat, lon
    circle = []
    last_val = 0
    shutdown = 0
    add_val = step_latlon
    difference = 0
    while shutdown == 0: # lon
        dis = VD(center, (center[0], center[1]+add_val)).km
        if dis >= RADIUS:
            difference = add_val
            shutdown = 1
        add_val += step_latlon
    bound_lat = (center[0]-difference, center[0]+difference)
    bound_lon = (center[1]-difference, center[1]+difference)
    #bound_lat = np.arange(bound_lat[0], bound_lat[1], step_ten_meter)
    #bound_lon = np.arange(bound_lon[0], bound_lon[1], step_ten_meter)
    bound_lat = np.arange(bound_lat[0], bound_lat[1], step_latlon)
    bound_lon = np.arange(bound_lon[0], bound_lon[1], step_latlon)
    #print bound_lat 
    #print bound_lon
    for lat in bound_lat:
        for lon in bound_lon:
            if VD(center, (lat,lon)).km <= RADIUS:
                latlon = [lat,lon]
                circle.append(latlon)
    circle = np.array(circle)
    return circle

if __name__ == '__main__':
    test_loc = (24.7890302,121.0069242)

    bound = GetCircleBound(test_loc)
    print bound.shape
