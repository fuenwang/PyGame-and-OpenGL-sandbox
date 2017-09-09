import cv2
import json
import numpy as np
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


vertices= (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )

def Read(f_name):
    with open(f_name) as f:
        data = json.load(f)[0]

    return data['points']

def Map1(data):
    glBegin(GL_POINTS)
    for idx, val in data.items():
        #print val
        pt = val['coordinates']
        color = val['color']
        color = [float(x) / 255 for x in color]
        glColor3f(*color)
        glVertex3f(*pt)
    glEnd()

def InitMap(data):
    points = []
    colors = []
    for idx, val in data.items():
        #print val
        pt = val['coordinates']
        color = val['color']
        color = [float(x) / 255 for x in color]
        points.append(pt)
        colors.append(color)
    return points, colors

def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        glVertex3f(*vertices[edge[0]])
        glVertex3f(*vertices[edge[1]])
    glEnd()

def Img2Sphere(img_name, radius = 128):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    [h, w, _] = img.shape
    c_x = (w - 1) / 2.0
    c_y = (h - 1) / 2.0
    #img[:, :w/4, :] = [0, 0, 0]
    #img[:, 3*w/4:, :] = [0, 0, 0]

    x_range = -(np.arange(0, w) - c_x) / c_x * np.pi
    #x_range[x_range > np.pi / 2] = (np.pi / 2) - (x_range[x_range > np.pi / 2] - np.pi / 2)
    #x_range[x_range < -np.pi / 2] = (-np.pi / 2) - (x_range[x_range < -np.pi / 2] + np.pi / 2)
    #id1 = np.arange(len(x_range))[x_range > np.pi / 2] 
    #id2 = np.arange(len(x_range))[x_range < -np.pi / 2]

    y_range = -((np.arange(0, h) - c_y) / c_y * np.pi / 2)

    x_grid = np.tile(x_range, [h, 1])
    y_grid = np.tile(y_range, [w, 1]).T
    
    tmp = np.zeros(img.shape, np.float) # store x, y, z

    tmp[:, :, 0] = radius * np.cos(x_grid) * np.cos(y_grid)
    tmp[:, :, 1] = radius * np.cos(y_grid) * np.sin(x_grid)
    #tmp[:, :, 1] *= -1
    #tmp[:, id1, 1] *= -1
    #tmp[:, id2, 1] *= -1
    tmp[:, :, 2] = radius * np.sin(y_grid)
    #tmp = np.round(tmp).astype(np.int)
    points = tmp.reshape([h*w, 3])
    colors = img.reshape([h*w, 3]) / 255.0
    #print tmp[:, 3*w/4, :]
    return points, colors

def main():
    [points, colors] = Img2Sphere('image.jpg', 128)
    #exit()
    #data = Read('reconstruction.json')
    #[points, colors] = InitMap(data)

    pg.init()
    display = (1920, 980)
    pg.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    gluPerspective(90, float(display[0]) / display[1], 0.1, 1000)
    #glTranslatef(0, 0, -500)
    #glRotatef(0, 0, 0, 0)
    gluLookAt(0, 0, 0, 0, -1, 0, 0, 0, 1)
    
    first = True
    state = 'default'
    while True:
        for event in pg.event.get():
            #print event.type
            if event.type == pg.QUIT:
                #pass
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
                    v0 = np.array([float(p0[0] - c_x)/w, float(p0[1] - c_y)/h, 1.3])
                    v1 = np.array([float(p1[0] - c_x)/w, float(p1[1] - c_y)/h, 1.3])
                    #if np.dot(v0, v1) > 0.999999:
                    #    break
                    v0 = v0 / np.linalg.norm(v0)
                    v1 = v1 / np.linalg.norm(v1)
                    axis = np.cross(v0, v1)
                    axis = axis / np.linalg.norm(axis)
                    angle = np.arccos(np.dot(v0, v1))
                    glRotatef( angle * 180 / np.pi, -axis[0], axis[1], -axis[2])
                    p1 = p0
            elif event.type == pg.MOUSEBUTTONUP:
                state = 'default'
        #glRotatef(20, 0, 1, 0)
        #glTranslatef(20, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if first:
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointerf(points)
            glColorPointerf(colors)
            glDrawArrays(GL_POINTS, 0, len(points))
            first = False
        else:
            glDrawArrays(GL_POINTS, 0, len(points))
            
        pg.display.flip()
        pg.time.wait(100)

main()




















