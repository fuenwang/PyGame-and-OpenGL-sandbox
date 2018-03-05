import cv2
import numpy as np
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from Equirec2Cube import Equirec2Cube as E2C
from Equirec2Cube import Depth2Points as D2P
from EquirecRotate import EquirecRotate as ER
import torch
from torch.autograd import Variable

if __name__ == '__main__':
    cube = 512
    e2c = E2C(1, 512, 1024, cube, 90, RADIUS=128)
    er = ER(512, 1024, RADIUS=128)
    
    e2c_grid = e2c.GetGrid()
    e2c_grid = D2P(e2c_grid)(Variable(torch.zeros(6, 1, cube, cube)).cuda() + 128)
    e2c_grid = e2c_grid.view(6*cube*cube, 3).data.cpu().numpy()
    #er_grid = er.GetGrid().view(512*1024, 3).data.cpu().numpy()
    color = (cv2.imread('0_color.png', cv2.IMREAD_COLOR))
    print color.tostring
    exit()
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) / 255.0
    color = e2c.ToCubeTensor(torch.FloatTensor(color.reshape(1, 512, 1024, 3).swapaxes(1, 3).swapaxes(2, 3)).cuda())
    color = color.transpose(1, 3).transpose(1, 2)
    color = color.contiguous().view(6*cube*cube, 3).data.cpu().numpy()

    pg.init()
    display = (1080, 1080)
    pg.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(120, float(display[0]) / display[1], 0.1, 100000)
    gluLookAt(256, 256, 256, 0, 0, 0, 0, -1, 0)

    flag = True
    state = 'default'
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
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


        glClearColor(1., 1., 1., 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if flag:
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointerf(e2c_grid.tolist())
            glColorPointerf(color.tolist())
            glDrawArrays(GL_POINTS, 0, e2c_grid.shape[0])
            flag = False

        else:
            glDrawArrays(GL_POINTS, 0, e2c_grid.shape[0])

        pg.display.flip()
        pg.time.wait(100)
        #exit()
