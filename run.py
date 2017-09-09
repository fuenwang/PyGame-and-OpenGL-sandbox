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

def main():
    data = Read('reconstruction.json')
    [points, colors] = InitMap(data)

    pg.init()
    display = (1920, 980)
    pg.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    gluPerspective(60, float(display[0]) / display[1], 0.1, 1500)
    glTranslatef(150, 150, -500)
    glRotatef(0, 0, 0, 1)
    
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
                p0 = np.array(p0)
                p1 = np.array(p1)
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




















