import cv2
import numpy as np
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
from Equirec2Cube import Equirec2Cube as E2C
from Equirec2Cube import Depth2Points as D2P
from EquirecRotate import EquirecRotate as ER
import torch
from torch.autograd import Variable
l = 1
TEX = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ]
CUBE = [
        # back
        [
            [l, -l, -l],
            [-l, -l, -l],
            [-l, l, -l],
            [l, l, -l]
        ],
        # down
        [
            [-l, l, l],
            [l, l, l],
            [l, l, -l],
            [-l, l, -l]
        ],
        # front
        [
            [-l, -l, l],
            [l, -l, l],
            [l, l, l],
            [-l, l, l]
        ],
        # left
        [
            [-l, -l,-l],
            [-l, -l, l],
            [-l, l, l],
            [-l, l, -l]
        ],
        # right
        [
            [l, -l, l],
            [l, -l, -l],
            [l, l, -l],
            [l, l, l]
        ],
        # up
        [
            [-l, -l, -l],
            [l, -l, -l],
            [l, -l, l],
            [-l, -l, l]
        ]
    ]


def load_shaders(vert_url, frag_url):
    vert_str = "\n".join(open(vert_url).readlines())
    frag_str = "\n".join(open(frag_url).readlines())
    vert_shader = shaders.compileShader(vert_str, GL_VERTEX_SHADER)
    frag_shader = shaders.compileShader(frag_str, GL_FRAGMENT_SHADER)
    program = shaders.compileProgram(vert_shader, frag_shader)
    return program

def CubeTexture(color):
    # color is 6 x w x w x 3
    lst = [
            GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
            GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
            GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
            GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
            GL_TEXTURE_CUBE_MAP_POSITIVE_X,
            GL_TEXTURE_CUBE_MAP_POSITIVE_Y
        ]
    ID = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, ID)
    for i in range(6):
        #i = 5
        cube = color[i, :, :, :]
        glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, GL_RGB, cube.shape[0], cube.shape[0], 0,
                GL_RGB, GL_FLOAT, cube
            )
        '''
        #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        glBegin(GL_QUADS)
        glTexCoord2f(TEX[0][0], TEX[0][1]); glVertex3f(CUBE[i][0][0], CUBE[i][0][1], CUBE[i][0][2]);
        glTexCoord2f(TEX[1][0], TEX[1][1]); glVertex3f(CUBE[i][1][0], CUBE[i][1][1], CUBE[i][1][2]);
        glTexCoord2f(TEX[2][0], TEX[2][1]); glVertex3f(CUBE[i][2][0], CUBE[i][2][1], CUBE[i][2][2]);
        glTexCoord2f(TEX[3][0], TEX[3][1]); glVertex3f(CUBE[i][3][0], CUBE[i][3][1], CUBE[i][3][2]);
        glEnd()
        '''
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE); 
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    #glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
    return ID


if __name__ == '__main__':
    
    cube = 512
    e2c = E2C(1, 512, 1024, cube, 90, RADIUS=128)
    er = ER(512, 1024, RADIUS=128)
    
    color = (cv2.imread('0_color.png', cv2.IMREAD_COLOR))
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) / 255.0
    color = e2c.ToCubeTensor(torch.FloatTensor(color.reshape(1, 512, 1024, 3).swapaxes(1, 3).swapaxes(2, 3)).cuda())
    color = color.transpose(1, 3).transpose(1, 2).data.cpu().numpy()
    order = [4, 3, 5, 1, 2, 0]
    gg = []
    for i in order:
        gg.append(color[i, :, :, :][np.newaxis, :, :, :])
    color = np.concatenate(gg, axis=0)
    color[:, 0:6, :, :]  = 0
    color[:, :, 0:6, :]  = 0
    color[:, -5:, :, :]  = 0
    color[:, :, -5:, :]  = 0
    #color[0, :, :, :] = 0
    idx = 0
    tmp = color[idx, :, :, :].copy()
    print tmp
    skybox_right = [1, -1, -1, 1, -1,  1, 1,  1,  1, 1,  1,  1, 1,  1, -1, 1, -1, -1]
    skybox_left = [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1]
    skybox_top = [-1,  1, -1, 1,  1, -1, 1,  1,  1, 1,  1,  1, -1,  1,  1, -1,  1, -1]
    skybox_bottom = [-1, -1, -1, -1, -1,  1, 1, -1, -1, 1, -1, -1, -1, -1,  1, 1, -1,  1]
    skybox_back = [-1,  1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1,  1, -1, -1,  1, -1]
    skybox_front = [-1, -1,  1, -1,  1,  1, 1,  1,  1, 1,  1,  1, 1, -1,  1, -1, -1,  1]
    skybox_vertices = np.array([skybox_back, skybox_bottom, skybox_front, skybox_left, skybox_right, skybox_top], np.float32).flatten().reshape([-1, 3])*3
    #print skybox_vertices.shape
    pg.init()
    display = (1080, 1080)
    pg.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_TEXTURE_CUBE_MAP)
    glEnable (GL_BLEND)
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    gluPerspective(90, float(display[0]) / display[1], 0.1, 100000)
    gluLookAt(-8, 8, -8, 0, 0, 0, 0, 1, 0)
    flag = True
    state = 'default'

    program = load_shaders("./skybox.vert", "./skybox.frag")
    skybox_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, skybox_vbo)
    glBufferData(GL_ARRAY_BUFFER, skybox_vertices.nbytes, skybox_vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    ID = CubeTexture(color)

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

        glClearColor(1., 1., 1., 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #glBindTexture(GL_TEXTURE_CUBE_MAP, ID)
        #glEnableClientState(GL_VERTEX_ARRAY)
        #glVertexPointer(3, GL_FLOAT, 0, np.array(CUBE, np.float32).reshape([-1, 3]))
        #glDrawArrays(GL_QUADS, 0, 24)
        #'''
        glUseProgram(program)
        #glDepthMask(GL_FALSE)
        glBindTexture(GL_TEXTURE_CUBE_MAP, ID)
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, skybox_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        #glDepthMask(GL_TRUE)
        glUseProgram(0)
        #'''
        '''
        glBindTexture(GL_TEXTURE_CUBE_MAP, ID)
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, skybox_vbo)
        print skybox_vbo
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        print 'aaa'
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        print 'ggg'
        '''
        pg.display.flip()
        pg.time.wait(100)
        #exit()
