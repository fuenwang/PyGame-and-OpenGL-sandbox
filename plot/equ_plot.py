import cv2
import json
import numpy as np
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from EquirecRotate import EquirecRotate as ER

def plot_grid(img):
    space = 13 # degree
    [h, w, c] = img.shape

    num = 180 // space
    interval = h // num
    pos = np.arange(num) * interval

    for i in range(num):
        cv2.line(img, (0, pos[i]), (w-1, pos[i]), color=(0, 0, 0), thickness=13)

    num = 360 // space
    interval = w // num
    pos = np.arange(num) * interval

    for i in range(num):
        #continue
        cv2.line(img, (pos[i], 0), (pos[i], h-1), color=(0, 0, 0), thickness=13)

def main():
    path = '/media/external/Fu-En.Wang/Data/360/final/rotated/023096db053da27b50cd745ececa2257/3.txt'
    color = (cv2.imread('%s/2_color.png'%path, cv2.IMREAD_COLOR))
    #color = (cv2.imread('../image.jpg', cv2.IMREAD_COLOR))
    color = cv2.resize(color, (6000, 3000))
    #plot_grid(color)
    #cv2.namedWindow('GG')
    #cv2.imshow('GG', color)
    #cv2.waitKey()
    #exit()

    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) / 255.0
    #color = color.reshape([-1, 3]) 

    #grid = ER(3328, 6656, 128).GetGrid().squeeze(0).view(-1, 3).data.cpu().numpy()
    grid = ER(3000, 6000, 128).GetGrid().squeeze(0).view(-1, 3).data.cpu().numpy()

    pg.init()
    display = (1080, 1080)
    pg.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable (GL_BLEND)

    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_TEXTURE_2D)
    #glEnable(GL_LINE_SMOOTH)
    #glEnable(GL_POINT_SMOOTH)
    #glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    #glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    #'''
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointerf(grid)
    glColorPointerf(color)

    gluPerspective(95, float(display[0]) / display[1], 0.1, 100000)
    # for tgt 2.png
    p = np.array([-2, 0, 2.5])
    p = p / np.linalg.norm(p) * 256
    gluLookAt(p[0], 0, p[2], 0, 0, 0, 0, -1, 0)
    # for ref 0.png
    #p = np.array([-1, 0, 2.5])
    #p = p / np.linalg.norm(p) * 256
    #gluLookAt(p[0], 0, p[2], 0, 0, 0, 0, -1, 0)

    
    first = True
    state = 'default'
    while True:
        for event in pg.event.get():
            # print event.type
            if event.type == pg.QUIT:
                # pass
                pg.quit()
                quit()
            elif event.type == pg.MOUSEBUTTONDOWN:
                p0 = pg.mouse.get_pos()
                state = 'drag'
            elif event.type == pg.MOUSEMOTION and state == 'drag':
                p1 = pg.mouse.get_pos()
                p0 = np.array([p0[0], p0[1]])
                p1 = np.array([p1[0], p1[1]])
                if np.linalg.norm(p0 - p1) > 50:
                    w = display[0]
                    h = display[1]
                    c_x = (w - 1) / 2
                    c_y = (h - 1) / 2
                    #v0 = np.array([2.0*p0[0]/w-1, -2.0*p0[1]/h+1, 1])
                    #v1 = np.array([2.0*p1[0]/w-1, -2.0*p1[1]/h+1, 1])
                    v0 = np.array(
                        [float(p0[0] - c_x) / w, float(p0[1] - c_y) / h, 1.3])
                    v1 = np.array(
                        [float(p1[0] - c_x) / w, float(p1[1] - c_y) / h, 1.3])
                    # if np.dot(v0, v1) > 0.999999:
                    #    break
                    v0 = v0 / np.linalg.norm(v0)
                    v1 = v1 / np.linalg.norm(v1)
                    axis = np.cross(v0, v1)
                    axis = axis / np.linalg.norm(axis)
                    angle = np.arccos(np.dot(v0, v1))
                    glRotatef(angle * 180 / np.pi, -axis[0], axis[1], -axis[2])
                    p1 = p0
            elif event.type == pg.MOUSEBUTTONUP:
                state = 'default'
        #glRotatef(20, 0, 1, 0)
        #glTranslatef(20, 0, 0)
        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glDrawArrays(GL_POINTS, 0, grid.shape[0])
        
        pg.display.flip()
        pg.time.wait(100)

main()
