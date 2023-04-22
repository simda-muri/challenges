import numpy as np
from stl import mesh
import os

# Define the 12 triangles composing the cube
faces = np.array([
    [0,3,1],
    [1,3,2],
    [0,4,7],
    [0,7,3],
    [4,5,6],
    [4,6,7],
    [5,1,2],
    [5,2,6],
    [2,3,6],
    [3,7,6],
    [0,1,5],
    [0,5,4]])

normals= np.array([
    [0,0,-1],
    [0,0,-1],
    [-1,0,0],
    [-1,0,0],
    [0,0,1],
    [0,0,1],
    [1,0,0],
    [1,0,0],
    [0,-1,0],
    [0,-1,0],
    [0,1,0],
    [0,1,0]])

def write_normal_stl(f,normal):
    f.write('   facet normal '+str(normal[0])+ ' '+str(normal[1])+ ' '+str(normal[2]))
    f.write('\n')
def write_triangle_stl(f,triangle):
    f.write('       outer loop')
    f.write('\n')
    for k in range(3):
        f.write('           vertex '+str(triangle[k,0])+ ' '+str(triangle[k,1])+ ' '+str(triangle[k,2]))
        f.write('\n')
    f.write('       endloop')
    f.write('\n')

def save_stl(dir,fname,xx,yy,px_size,bot,top):
    with open('./' +dir+'/'+fname+'.stl', 'w') as f:
        f.write('solid '+fname)
        f.write('\n')
        for i in range(len(xx)):
            xi=xx[i]*px_size
            yi=yy[i]*px_size
            xi1=xi+px_size
            yi1=yi+px_size
            xi_m=int(xx[i])
            yi_m=int(yy[i])
            V_r = np.zeros((8,3))
            V_r[0,:]=[xi,yi1,bot[xi_m,yi_m]]
            V_r[1,:]=[xi1,yi1,bot[xi_m,yi_m]]
            V_r[2,:]=[xi1,yi,bot[xi_m,yi_m]]
            V_r[3,:]=[xi,yi,bot[xi_m,yi_m]]
            V_r[4,:]=[xi,yi1,top[xi_m,yi_m]]
            V_r[5,:]=[xi1,yi1,top[xi_m,yi_m]]
            V_r[6,:]=[xi1,yi,top[xi_m,yi_m]]
            V_r[7,:]=[xi,yi,top[xi_m,yi_m]]
            for t in range(len(faces)):
                write_normal_stl(f,normals[t,:])
                write_triangle_stl(f,V_r[faces[t,:],:])
                f.write('   endfacet')
                f.write('\n')
        f.write('endsolid '+fname)

    m=mesh.Mesh.from_file('./' +dir+'/'+fname+'.stl')
    data=m.remove_duplicate_polygons(m.data)
    m2=mesh.Mesh(data)
    m2.save('./' +dir+'/'+fname+'.stl')

def save_water_stl(dir,fname,px_size,rec_water):
    with open('./' +dir+'/'+fname+'.stl', 'w') as f:
        f.write('solid '+fname)
        f.write('\n')
        for i in range(rec_water.shape[0]):
            xi=int(rec_water[i,0])*px_size
            yi=int(rec_water[i,1])*px_size
            xi1=int(rec_water[i,0]+rec_water[i,2])*px_size
            yi1=int(rec_water[i,1]+rec_water[i,3])*px_size
            V_r = np.zeros((8,3))
            V_r[0,:]=[xi,yi1,0]
            V_r[1,:]=[xi1,yi1,0]
            V_r[2,:]=[xi1,yi,0]
            V_r[3,:]=[xi,yi,0]
            V_r[4,:]=[xi,yi1,0]
            V_r[5,:]=[xi1,yi1,0]
            V_r[6,:]=[xi1,yi,0]
            V_r[7,:]=[xi,yi,0]
            for t in range(len(faces)):
                write_normal_stl(f,normals[t,:])
                write_triangle_stl(f,V_r[faces[t,:],:])
                f.write('   endfacet')
                f.write('\n')
        f.write('endsolid '+fname)

    m=mesh.Mesh.from_file('./' +dir+'/'+fname+'.stl')
    data=m.remove_duplicate_polygons(m.data)
    m2=mesh.Mesh(data)
    m2.save('./' +dir+'/'+fname+'.stl')
