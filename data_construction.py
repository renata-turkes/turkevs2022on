# Some parts of the code are inspired from:
# https://github.com/raphaeltinarrage/velour/blob/main/velour/datasets.py
# https://github.com/CSU-TDA/PersistenceImages/blob/master/matlab_code/sixShapeClasses/generate_shape_data.m
# https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-DTM-filtrations.ipynb
# https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-cubical-complexes.ipynb
# https://github.com/dwyerk/boundaries/blob/master/concave_hulls.ipynb
# https://en.wikipedia.org/wiki/Parametric_equation#/media/File:Param_02.jpg
# https://en.wikipedia.org/wiki/File:Param_03.jpg
# https://en.wikipedia.org/wiki/File:Sphere_with_three_handles.png
# https://en.wikipedia.org/wiki/Handlebody
# Bubenik, Peter, et al. "Persistent homology detects curvature." Inverse Problems 36.2 (2020): 025008.



import numpy as np
import math
import pickle
import collections
from scipy.spatial import ConvexHull
import alphashape
from descartes import PolygonPatch
from shapely.geometry import Point 
from matplotlib.path import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
import matplotlib.pyplot as plt

import plots



PATH_CURRENT = "../"
NUM_POINT_CLOUDS = 100
NUM_POINTS = 1000
NUM_POINT_CLOUDS_SHAPE = 50




############################################## HOLES #################################################################



def sample_parabola(n = NUM_POINTS):
    x = np.random.uniform(-1, 1, size = n)
    y = x**2
    X = np.array([x, y]).T    
    return X



def sample_circle(n = NUM_POINTS, r = 1):   
    t = np.random.uniform(0, 2*np.pi, size = n)
    x = r * np.cos(t)
    y = r * np.sin(t)     
    X = np.array([x, y]).T    
    return X



def sample_closed_curve(n = NUM_POINTS, num_vertices = 3):    
    t = np.random.uniform(0, 2*np.pi, size = n)
    b = 1
    k = num_vertices
    a = k * b
    x = (a - b) * np.cos(t) + b * np.cos(t * ((a / b) - 1)) 
    y = (a - b) * np.sin(t) - b * np.sin(t * ((a / b) - 1))    
    X = np.array([x, y]).T
    return X



def sample_annulus(n = NUM_POINTS):
    t = np.random.uniform(0, 2*np.pi, size = n)
    x = (1 + np.random.uniform(0, 1, size = n)) * np.cos(t)
    y = (1 + np.random.uniform(0, 1, size = n)) * np.sin(t)
    X = np.array([x, y]).T  
    return X



def sample_two_circles(n = NUM_POINTS):   
    n1 = int(n/2)
    n2 = n - n1
    circle1 = sample_circle(n1, r = 1)
    circle2 = 2 + sample_circle(n2, r = 1)
    X = np.concatenate([circle1, circle2]) 
    return X



def sample_lemniscate(n = NUM_POINTS):
    t = np.random.uniform(0, 2*np.pi, size = n)
    x = np.cos(t)
    y = np.sin(2 * t)    
    X = np.array([x, y]).T
    return X



def sample_concentric_circles(n = NUM_POINTS):
    n1 = int(n/2)
    n2 = n - n1
    circle1 = sample_circle(n1, r = 1)
    circle2 = sample_circle(n2, r = 0.5)
    X = np.concatenate([circle1, circle2]) 
    return X



def sample_rose_curve(n = NUM_POINTS, a = 2):
    t = np.random.uniform(0, 2*np.pi, size = n)
    x = np.cos(a * t) * np.cos(t)
    y = np.cos(a * t) * np.sin(t)   
    X = np.array([x, y]).T
    return X



def sample_rose_nine_petals(n = NUM_POINTS):
    n_rose_three_petals = int(n/3)
    rose1 = sample_rose_curve(n_rose_three_petals, a = 3)
    rose2 = rotate(rose1, t = np.pi/4.5)
    rose3 = rotate(rose2, t = np.pi/4.5)
    n_rmn = n - 3 * n_rose_three_petals
    center = np.asarray([(0, 0)] * n_rmn)
    X = np.concatenate([rose1, rose2, rose3, center]) 
    return X



def sample_eye(n = NUM_POINTS):    
    t = np.random.uniform(0, 2*np.pi, size = n)
    a = 2
    b = 1
    c = 2
    d = 1
    x = np.cos(a * t) - np.cos(b * t)**3
    y = np.sin(c * t) - np.sin(d * t)**3
    X = np.array([x, y]).T 
    return X



def sample_eyes_1(n = NUM_POINTS):   
    t = np.random.uniform(0, 2*np.pi, size = n)
    a = 3
    b = 1
    c = 3
    d = 1
    x = np.cos(a * t) - np.cos(b * t)**3
    y = np.sin(c * t) - np.sin(d * t)**3
    X = np.array([x, y]).T
    return X



def sample_eyes_2(n = NUM_POINTS):
    t = np.random.uniform(0, 2*np.pi, size = n)
    a = 1
    b = 3
    c = 3
    d = 1
    x = np.cos(a * t) - np.cos(b * t)**3
    y = np.sin(c * t) - np.sin(d * t)**3
    X = np.array([x, y]).T
    return X



def sample_olympics(n = NUM_POINTS):
    t = np.random.uniform(0, 2*np.pi, size = int(n/5))
    circle = np.array([np.cos(t), np.sin(t)])  
    
    l = 1
    center1 = np.array([0, 0.75])
    center2 = np.array([-1*l, 0])
    center3 = np.array([-2*l, 0.75])
    center4 = np.array([1*l, 0])
    center5 = np.array([2*l, 0.75])
    
    X1 = np.transpose(np.transpose(circle) + center1)    
    X2 = np.transpose(np.transpose(circle) + center2)    
    X3 = np.transpose(np.transpose(circle) + center3)    
    X4 = np.transpose(np.transpose(circle) + center4)    
    X5 = np.transpose(np.transpose(circle) + center5)    
    
    X = np.concatenate([X1, X2, X3, X4, X5], 1)
    X = X.T  
    return X  



def sample_triangle(n = 100):   
    n_line = int(n/3)
    n_last_line = n - 2 * n_line    
    x1 = np.random.uniform(0, 1, size = n_line)    
    y1 = [0] * n_line    
    x2 = np.random.uniform(0, 0.5, size = n_line)  
    y2 = 2*x2    
    x3 = np.random.uniform(0.5, 1, size = n_last_line)  
    y3 = 2 - 2*x3
    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    X = np.array([x, y]).T    
    return X



def sample_square(n = NUM_POINTS):   
    n_line = int(n/4)
    n_last_line = n - 3 * n_line    
    x = np.random.uniform(0, 1, size = n_line)
    x1 = x
    x2 = x
    x3 = [0] * n_line
    x4 = [1] * n_last_line    
    y1 = [0] * n_line
    y2 = [1] * n_line    
    y3 = np.random.uniform(0, 1, size = n_line)
    y4 = np.random.uniform(0, 1, size = n_last_line)
    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    X = np.array([x, y]).T    
    return X



def sample_nine_squares(n = NUM_POINTS):    
    n_square = int(n / 9)
    X1 = sample_square(n_square)
    X_shift = np.zeros(X1.shape)
    X_shift[:, 0] = 1.1
    Y_shift = np.zeros(X1.shape)
    Y_shift[:, 1] = 1.1  
    X2 = X1 + X_shift
    X3 = X2 + X_shift
    X4 = X1 + Y_shift
    X5 = X4 + X_shift
    X6 = X5 + X_shift
    X7 = X4 + Y_shift
    X8 = X7 + X_shift
    X9 = X8 + X_shift       
    n_first_square = n - 8 * n_square
    X1 = sample_square(n_first_square)      
    X = np.concatenate([X1, X2, X3, X4, X5, X6, X7, X8, X9])    
    return X 



def sample_sphere(n = NUM_POINTS, r = 1):
    theta = np.random.uniform(0, 2*np.pi, size = n)
    phi = np.arccos(np.random.uniform(-1, 1, size = n))    
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi) 
    X = np.array([x, y, z]).T    
    return X   



def sample_dimple(n = NUM_POINTS):  
    theta = np.random.uniform(0, 2*np.pi, size = n)
    phi = np.arccos(np.random.uniform(-1, 1, size = n))
    x = ( 1 + 0.4 * np.sin(3 * phi) * np.cos(theta) ) * np.cos(theta) * np.sin(phi)
    y = ( 1 + 0.4 * np.sin(3 * phi) * np.cos(theta) ) * np.sin(theta) * np.sin(phi)
    z = ( 1 + 0.4 * np.sin(3 * phi) * np.cos(theta) ) * np.cos(phi)
    X = np.array([x, y, z]).T
    return X  



def sample_fountain(n = NUM_POINTS):
    theta = np.random.uniform(0, 2*np.pi, size = n)
    phi = np.arccos(np.random.uniform(-1, 1, size = n))
    x = ( 1 + 0.4 * np.sin(7 * phi) * np.cos(theta) ) * np.cos(theta) * np.sin(phi)
    y = ( 1 + 0.4 * np.sin(7 * phi) * np.cos(theta) ) * np.sin(theta) * np.sin(phi)
    z = ( 1 + 0.4 * np.sin(7 * phi) * np.cos(theta) ) * np.cos(phi) 
    X = np.array([x, y, z]).T
    return X  



def sample_two_spheres(n = NUM_POINTS):       
    n1 = int(n/2)
    n2 = n - n1
    sphere1 = sample_sphere(n1, r = 1)
    sphere2 = 2 + sample_sphere(n2, r = 1)
    X = np.concatenate([sphere1, sphere2]) 
    return X



def sample_3d_curve(n = NUM_POINTS, gamma = 1):
    t = np.random.uniform(0, 2*np.pi, size = n)
    Xdata = np.cos(t)
    Ydata = np.sin(t)
    XXnormal = gamma * np.cos(t)**2
    YYnormal = gamma * np.sin(t)**2
    XYnormal = gamma * np.cos(t) * np.sin(t)    
    X = np.vstack((Xdata,Ydata, XXnormal, XYnormal, XYnormal, YYnormal)).T   
    X = np.array([X[:, 0], X[:, 1], X[:, 2]]).T
    return X



def sample_torus(n = NUM_POINTS, r1 = 1, r2 = 0.5):        
    theta = np.random.uniform(0, 2*np.pi, size = n)
    phi = np.random.uniform(0, 2*np.pi, size = n)
    x = (r1 + r2 * np.cos(theta)) * np.cos(phi)
    y = (r1 + r2 * np.cos(theta)) * np.sin(phi)
    z = r2 * np.sin(theta)    
    X = np.array([x, y, z]).T
    return X



def sample_double_torus(n = NUM_POINTS, r1 = 1, r2 = 0.5):
    n1 = int(n/2)
    n2 = n - n1
    torus1 = sample_torus(n1, r1, r2)
    torus2 = 1 + 0.2 + sample_torus(n2, r1, r2)
    X = np.concatenate([torus1, torus2])
    return X



def sample_necklace(n = NUM_POINTS):
    X1 = sample_sphere(int(n/4)) + [2, 0, 0]
    X2 = sample_sphere(int(n/4)) + [-1, 2*0.866, 0]
    X3 = sample_sphere(int(n/4)) + [-1, -2*0.866, 0]
    X4 = 2 * sample_circle(int(n/4)) 
    X4 = np.array((X4[:, 0], X4[:, 1], np.zeros(int(n/4)))).T
    X = np.concatenate((X1, X2, X3, X4))   
    return X



def sample_handlebody(n = NUM_POINTS, r = 1, r1 = 0.5, r2 = 0.25): 
    # This is not the actual number of handle points, as only the ones outside the sphere (e.g., around half) will be kept.
    n_torus = int(n / 4)    
    
    torus = sample_torus(n_torus, r1, r2)    
    torus1 = np.asarray([r, 0, 0]) + torus  
    handle1 = []
    for p in range(n_torus):
        x = torus1[p, 0]
        y = torus1[p, 1]
        z = torus1[p, 2]
        if x**2 + y**2 + z**2 > r**2:
            handle1.append([x, y, z])
    handle1 = np.asarray(handle1)
    n_handle1 = handle1.shape[0]
            
    torus2 = np.asarray([-r, 0, 0]) + torus      
    handle2 = []
    for p in range(n_torus):
        x = torus2[p, 0]
        y = torus2[p, 1]
        z = torus2[p, 2]
        if x**2 + y**2 + z**2 > r**2:
            handle2.append([x, y, z])
    handle2 = np.asarray(handle2)
    n_handle2 = handle2.shape[0]
            
    torus3 = np.asarray([0, r, 0]) + torus 
    handle3 = []
    for p in range(n_torus):
        x = torus3[p, 0]
        y = torus3[p, 1]
        z = torus3[p, 2]
        if x**2 + y**2 + z**2 > r**2:
            handle3.append([x, y, z])
    handle3 = np.asarray(handle3)
    n_handle3 = handle3.shape[0]
            
    torus4 = np.asarray([0, -r, 0]) + torus
    handle4 = []
    for p in range(n_torus):
        x = torus4[p, 0]
        y = torus4[p, 1]
        z = torus4[p, 2]
        if x**2 + y**2 + z**2 > r**2:
            handle4.append([x, y, z])
    handle4 = np.asarray(handle4)
    n_handle4 = handle4.shape[0]   
    
    n_rmn = n - n_handle1 - n_handle2 - n_handle3 - n_handle4
    n_sphere = int(0.9 * n_rmn)
    n_circle = n_rmn - n_sphere
    
    sphere = sample_sphere(n_sphere, r)  
    
    circle_2d = sample_circle(n_circle, r)    
    circle = np.zeros((n_circle, 3))
    for p in range(n_circle):
        circle[p] = np.asarray([circle_2d[p][0], circle_2d[p][1], 2*r])   

    X = np.concatenate([circle, sphere, handle1, handle2, handle3, handle4])
    return X
 


def sample_concentric_spheres(n = NUM_POINTS):
    n1 = int(n/2)
    n2 = n - n1
    
    theta1 = np.random.uniform(0, 2*np.pi, size = n1)
    phi1 = np.arccos(np.random.uniform(-1, 1, size = n1))
    x1 = np.cos(theta1) * np.sin(phi1)
    y1 = np.sin(theta1) * np.sin(phi1)
    z1 = np.cos(phi1) 

    theta2 = np.random.uniform(0, 2*np.pi, size = n2) 
    phi2 = np.arccos(np.random.uniform(-1, 1, size = n2))
    x2 = 3 * np.cos(theta2) * np.sin(phi2)
    y2 = 3 * np.sin(theta2) * np.sin(phi2)
    z2 = 3 * np.cos(phi2)

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    z = np.concatenate([z1, z2])   
    X = np.array([x, y, z]).T
    return X   



def sample_coconut_shell(n = NUM_POINTS):
    theta = np.random.uniform(0, 2*np.pi, size = n)
    phi = np.arccos(np.random.uniform(-1, 1, size = n))
    x = (5 + 1 * np.random.rand(n)) * np.cos(theta) * np.sin(phi)
    y = (5 + 1 * np.random.rand(n)) * np.sin(theta) * np.sin(phi)
    z = (5 + 1 * np.random.rand(n)) * np.cos(phi) 
    X = np.array([x, y, z]).T
    return X   



def sample_trefoil_knot(n = NUM_POINTS):
    t = np.random.uniform(0, 2*np.pi, size = n)
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = - np.sin(3 * t)
    X = np.array([x, y, z]).T 
    return X



def sample_klein_bottle(n = NUM_POINTS):
    r = 3 # r > 2
    theta = np.random.uniform(0, 2*np.pi, size = n)
    phi = np.random.uniform(0, 2*np.pi, size = n)
    x = (r + np.cos(theta/2) * np.sin(phi) - np.sin(theta/2) * np.sin(2 * phi)) * np.cos(theta) 
    y = (r + np.cos(theta/2) * np.sin(phi) - np.sin(theta/2) * np.sin(2 * phi)) * np.sin(theta) 
    z = np.sin(theta/2) * np.sin(phi) + np.cos(theta/2) * np.sin(2*phi)
    return X



def sample_cross_capped_disk(n = NUM_POINTS):
    u = np.random.uniform(0, 2*np.pi, size = n)
    v = np.random.uniform(0, 2*np.pi, size = n)
    r = 1
    x = r * (1 + np.cos(v)) * np.cos(u)
    y = r * (1 + np.cos(v)) * np.sin(u)
    z = -np.tanh(u - np.pi) * r * np.sin(v)
    X = np.array([x, y, z]).T 
    return X



def sample_self_intersecting_disk(n = NUM_POINTS):
    u = np.random.uniform(0, 2*np.pi, size = n)
    v = np.random.uniform(0, 1, size = n)
    r = 1
    x = r * v * np.cos(2 * u)
    y = r * v * np.sin(2 * u)
    z = r * v * np.cos(u)
    X = np.array([x, y, z]).T 
    return X



def sample_crater(n = NUM_POINTS):
    # https://github.com/GUDHI/TDA-tutorial/tree/master/datasets
    f = open("datasets/crater_tuto", "rb")
    X = pickle.load(f)
    X = X[0:n, :]
    f.close()
    return X



def sample_point_cloud(n = NUM_POINTS, shape = " "):     
    if shape == "parabola":
        X = sample_parabola(n)
    elif shape == "circle":
        X = sample_circle(n)        
    elif shape == "curve3":
        X = sample_closed_curve(n, num_vertices = 3)
    elif shape == "curve4":
        X = sample_closed_curve(n, num_vertices = 4)
    elif shape == "curve5" or shape == "pentagon":
        X = sample_closed_curve(n, num_vertices = 5)
    elif shape == "annulus":
        X = sample_annulus(n)
    elif shape == "two circles":
        X = sample_two_circles(n)
    elif shape == "lemniscate":
        X = sample_lemniscate(n)
    elif shape == "concentric circles":
        X = sample_concentric_circles(n) 
    elif shape == "rose curve 4":
        X = sample_rose_curve(n, a = 2)
    elif shape == "face":
        X = sample_eyes_1(n)
    elif shape == "rose curve 9":
        # X = sample_rose_curve(n, a = 9)
        X = sample_rose_nine_petals(n)
    elif shape == "olympics":
        X = sample_olympics(n)
    elif shape == "square":
        X = sample_square(n)
    elif shape == "nine squares":
        X = sample_nine_squares(n)
    elif shape == "triangle":
        X = sample_triangle(n)        
    elif shape == "sphere":
        X = sample_sphere(n)
    elif shape == "fountain":
        X = sample_fountain(n)
    elif shape == "two spheres":
        X = sample_two_spheres(n)
    elif shape == "3d curve":
        X = sample_3d_curve(n)
    elif shape == "torus":
        X = sample_torus(n)
    elif shape == "double torus":
        X = sample_double_torus(n)
    elif shape == "necklace":
        X = sample_necklace(n)
    elif shape == "handlebody":
        X = sample_handlebody(n) 
    else:
        "Error: There is no function that constructs a point cloud with the given shape! Check your spelling."       
    return X



def build_dataset_holes(N = NUM_POINT_CLOUDS, n = NUM_POINTS):
    shapes = ["parabola", "sphere", "fountain", "two spheres",             # 0 holes
              "circle", "pentagon", "annulus", "3d curve",                 # 1 hole
              "two circles", "lemniscate", "concentric circles", "torus",  # 2 holes
              "rose curve 4", "face", "double torus", "necklace",          # 4 holes
              "rose curve 9", "olympics", "nine squares", "handlebody"]    # 9 holes
    num_cycles = [0, 0, 0, 0,
                  1, 1, 1, 1,
                  2, 2, 2, 2,
                  4, 4, 4, 4,
                  9, 9, 9, 9]
    
    num_shapes = len(shapes)
    num_point_clouds_shape = int(N / num_shapes)
    if num_point_clouds_shape == 0:
        print("There are 20 shapes in the holes dataset, so that at least 20 point clouds must be created!")

    # point_clouds is a list, rather than an array, since some point clouds are 2D, and some are 3D.
    point_clouds = []
    labels = []
    labels_shape = []
    for i, shape in enumerate(shapes):        
        for j in range(num_point_clouds_shape):
            point_cloud = sample_point_cloud(n, shape)
            point_cloud = normalize(point_cloud)
            label = num_cycles[i]
            point_clouds = point_clouds + [point_cloud]
            labels = labels + [label] 
            labels_shape = labels_shape + [shape]
        
        # An example of a point cloud of a given shape (the last generated point cloud).
        # plots.plot_point_cloud(point_cloud, title = shape + ", label = " + str(label) + ", num_points = " + str(point_cloud.shape[0]))
    
    labels = np.asarray(labels)
    labels_shape = np.asarray(labels_shape)
    
    print("Dataset of point clouds for the detection of the number of holes:")
    print("num_shapes = ", num_shapes)
    print("type(point_clouds) = ", type(point_clouds))
    print("len(point_clouds) = ", len(point_clouds))
    print("point_clouds[0].shape = ", point_clouds[0].shape)
    print()
    print("type(labels) = ", type(labels))
    print("len(labels) = ", len(labels))
    print("Numbers of point clouds with each label value = ", collections.Counter(labels))
    
    return point_clouds, labels, labels_shape



############################################## CURVATURE #################################################################



def sample_unit_disk_euclidean(n = 100):
    
    # The angle is sampled uniformly.
    t = np.random.uniform(0, 2*np.pi, size = n)

    # The radius is sampled in such a way that the probability of a point lying within the disk of radius r should equal 
    # the proportion to the area of that disk relative to the area of the disk of radius 1.
    u = np.random.uniform(0, 1, size = n)
    r = np.sqrt(u)

    x = r * np.cos(t)
    y = r * np.sin(t)   
    X = np.array([x, y]).T   
    return X

    

def sample_unit_disk_spherical(n = 100, K = 1):
    R = 1 / np.sqrt(K)

    # The angle is sampled uniformly.
    t = np.random.uniform(0, 2*np.pi, size = n)

    # The radius is sampled in such a way that the probability of a point lying within the disk of radius r should equal 
    # the proportion to the area of that disk relative to the area of the disk of radius 1.
    u = np.random.uniform(0, 1, size = n)
    r = ( 2 / np.sqrt(K) ) * np.arcsin( np.sqrt(u) * np.sin( np.sqrt(K)/2 ) )

    x = R * np.sin(r / R) * np.cos(t)
    y = R * np.sin(r / R) * np.sin(t)
    z = R * np.cos(r / R)
    X = np.array([x, y, z]).T   
    return X



def sample_unit_disk_hyperbolic(n = 100, K = -1):
    R = 1 / np.sqrt(-K)

    # The angle is sampled uniformly.
    t = np.random.uniform(0, 2*np.pi, size = n)

    # The radius is sampled in such a way that the probability of a point lying within the disk of radius r should equal 
    # the proportion to the area of that disk relative to the area of the disk of radius 1.
    u = np.random.uniform(0, 1, size = n)
    r = ( 2 / np.sqrt(-K) ) * np.arcsinh( np.sqrt(u) * np.sinh( np.sqrt(-K)/2 ) )

    x = R * np.tanh(r / (2 * R)) * np.cos(t)
    y = R * np.tanh(r / (2 * R)) * np.sin(t)    
    X = np.array([x, y]).T    
    return X



def sample_unit_disk(n, K):
    
    # Euclidean case, K = 0.
    if K == 0:
        point_cloud = sample_unit_disk_euclidean(n)  
    
    # Spherical case, K in [0.04, 0.08, 0.12, ..., 1.92, 1.96, 2].
    elif K > 0:
        point_cloud = sample_unit_disk_spherical(n, K)  
    
    # Hyperbolic case, K in [-2, -1.96, -1.92, ..., -0.12, -0.08, -0.04].
    else:
        point_cloud = sample_unit_disk_hyperbolic(n, K) 
   
    return point_cloud



def calculate_distance_matrix_spherical(X, K):
    R = 1 / np.sqrt(K)
    num_points = X.shape[0]
    dis_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i):
            v1 = X[i]
            v2 = X[j]
            dis_matrix[i, j] = R * math.atan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))           
    # Distance matrix elements above the diagonal.
    dis_matrix =  dis_matrix + dis_matrix.transpose()            
    return dis_matrix



def calculate_distance_matrix_hyperbolic(X, K):
    R = 1 / np.sqrt(-K)
    num_points = X.shape[0]
    dis_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i):
            v1 = X[i]
            v2 = X[j]            
            c1 = v1/R
            c1 = complex(c1[0], c1[1])
            c2 = v2/R
            c2 = complex(c2[0], c2[1])            
            dis_matrix[i, j] = 1 * R * np.arctanh( np.abs(c1 - c2) / np.abs(1 - c1*np.conj(c2)) )
            # dis_matrix[i, j] = 2 * R * np.arctanh( np.abs(c1 - c2) / np.abs(1 - c1*np.conj(c2)) )
    # Distance matrix elements above the diagonal.
    dis_matrix =  dis_matrix + dis_matrix.transpose() 
    return dis_matrix



def build_dataset_curvature_train(n = NUM_POINTS, num_train_curvatures = 101, num_point_clouds_train_curvature = 10):     
    
    # num_curvatures = 101 so that we have 50 negative curvatures, zero curvature, 50 positive curvatures.
    Ks = np.linspace(-2, 2, num = num_train_curvatures)     

    # point_clouds is a list, rather than an array, since some point clouds are 2D, and some are 3D.
    point_clouds = []
    labels = []
    distance_matrices = []
    
    for K in Ks:        
        for j in range(num_point_clouds_train_curvature):
            
            # Euclidean case, K = 0.
            if K == 0:
                point_cloud = sample_unit_disk_euclidean(n)  
                distance_matrix = euclidean_distances(point_cloud)            
            
            # Spherical case, K in [0.04, 0.08, 0.12, ..., 1.92, 1.96, 2].
            elif K > 0:
                point_cloud = sample_unit_disk_spherical(n, K)  
                distance_matrix = calculate_distance_matrix_spherical(point_cloud, K)            
            
            # Hyperbolic case, K in [-2, -1.96, -1.92, ..., -0.12, -0.08, -0.04].
            elif K < 0:
                point_cloud = sample_unit_disk_hyperbolic(n, K) 
                distance_matrix = calculate_distance_matrix_hyperbolic(point_cloud, K)                            
            
            point_clouds = point_clouds + [point_cloud]
            distance_matrices = distance_matrices + [distance_matrix]
            label = K
            labels = labels + [label]             

    distance_matrices = np.asarray(distance_matrices)
    labels = np.asarray(labels)

    print("Train dataset of point clouds for the detection of curvature:")
    print("type(point_clouds) = ", type(point_clouds))
    print("len(point_clouds) = ", len(point_clouds))
    print("point_clouds[0].shape = ", point_clouds[0].shape)
    print("distance_matrices.shape = ", distance_matrices.shape)
    print("labels.shape = ", labels.shape)
    print("Numbers of point clouds with each label value: ", collections.Counter(np.around(labels, 2)))
    print()    
    return point_clouds, labels, distance_matrices



def build_dataset_curvature_test(n = NUM_POINTS, N = 100):     
    
    Ks =  np.random.uniform(-2, 2, size = N)   
       
    # point_clouds is a list, rather than an array, since some point clouds are 2D, and some are 3D.
    point_clouds = []
    labels = []
    distance_matrices = []
    
    for K in Ks:   
                   
        # Euclidean case, K = 0.
        if K == 0:
            point_cloud = sample_unit_disk_euclidean(n)  
            distance_matrix = euclidean_distances(point_cloud)            
        
        # Spherical case, K in [0.04, 0.08, 0.12, ..., 1.92, 1.96, 2].
        elif K > 0:
            point_cloud = sample_unit_disk_spherical(n, K)  
            distance_matrix = calculate_distance_matrix_spherical(point_cloud, K)            
       
        # Hyperbolic case, K in [-2, -1.96, -1.92, ..., -0.12, -0.08, -0.04].
        elif K < 0:
            point_cloud = sample_unit_disk_hyperbolic(n, K) 
            distance_matrix = calculate_distance_matrix_hyperbolic(point_cloud, K)                            
            
        point_clouds = point_clouds + [point_cloud]
        distance_matrices = distance_matrices + [distance_matrix]
        label = K
        labels = labels + [label] 
    
    distance_matrices = np.asarray(distance_matrices)
    labels = np.asarray(labels)
    
    print("Test dataset of point clouds for the detection of curvature:")
    print("type(point_clouds) = ", type(point_clouds))
    print("len(point_clouds) = ", len(point_clouds))
    print("point_clouds[0].shape = ", point_clouds[0].shape)
    print("distance_matrices.shape = ", distance_matrices.shape)
    print("labels.shape = ", labels.shape)
    print("Numbers of point clouds with each label value: ", collections.Counter(np.around(labels, 2)))
    print()        
    return point_clouds, labels, distance_matrices



def build_dataset_curvature(n = NUM_POINTS, num_train_curvatures = 101, num_point_clouds_train_curvature = 10, num_test_point_clouds = 100):
    point_clouds_train, labels_train, distance_matrices_train = build_dataset_curvature_train(n, num_train_curvatures, num_point_clouds_train_curvature)
    point_clouds_test, labels_test, distance_matrices_test = build_dataset_curvature_test(n, num_test_point_clouds)
    point_clouds = point_clouds_train + point_clouds_test
    labels = np.concatenate((labels_train, labels_test))
    distance_matrices = np.concatenate((distance_matrices_train, distance_matrices_test))
    return point_clouds, labels, distance_matrices




############################################## CONVEXITY #################################################################



def sample_triangle_solid(n = NUM_POINTS):       
    n1 = int(n/2)
    n2 = n - n1    
    x1 = np.random.uniform(0, 0.5, size = n1)    
    y1 = np.random.uniform(0, 2*x1, size = n1)        
    x2 = np.random.uniform(0.5, 1, size = n2)  
    y2 = np.random.uniform(0, -2*x2 + 2, size = n2)     
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    X = np.array([x, y]).T      
    return X



def sample_square_solid(n = NUM_POINTS):   
    x = np.random.uniform(0, 1, size = n)
    y = np.random.uniform(0, 1, size = n)
    X = np.array([x, y]).T    
    return X



def sample_pentagon_solid(n = NUM_POINTS, x_min = 0, x_max = 1, y_min = 0, y_max = 1):
    
    # 1) Define pentagon (corner) vertices.
    points = np.array([[0.33, 0], [0.66, 0], [1, 0.5], [0.5, 1], [0, 0.5]])
    
    # 2) Build the convex hull, type(hull) =  <class 'scipy.spatial.qhull.ConvexHull'>
    hull = ConvexHull(points)
    
    # 3) Sample point cloud with the given number of points inside the hull.
    X = sample_random_point_cloud_within_hull(hull, n)
    
    return X



def sample_circle_solid(n = NUM_POINTS):   
    t = np.random.uniform(0, 2*np.pi, size = n)
    r = np.random.uniform(0, 1, size = n)
    x = r * np.cos(t)
    y = r * np.sin(t)     
    X = np.array([x, y]).T    
    return X



def sample_closed_curve_solid(n = NUM_POINTS, num_vertices = 3):    
    t = np.random.uniform(0, 2*np.pi, size = n)
    b = np.random.uniform(0, 1, size = n)
    k = num_vertices
    a = k * b
    x = (a - b) * np.cos(t) + b * np.cos(t * ((a / b) - 1))
    y = (a - b) * np.sin(t) - b * np.sin(t * ((a / b) - 1))
    X = np.array([x, y]).T
    return X



def sample_point_cloud_solid(n = NUM_POINTS, shape = " "):    
    # Convex solids.
    if shape == "triangle":
        X = sample_triangle_solid(n)
    elif shape == "square":
        X = sample_square_solid(n)
    elif shape == "pentagon":
        X = sample_pentagon_solid(n)
    elif shape == "circle":
        X = sample_circle_solid(n)
    # Concave solids.
    elif shape == "curve3":
        X = sample_closed_curve_solid(n, num_vertices = 3)
    elif shape == "curve4":
        X = sample_closed_curve_solid(n, num_vertices = 4)
    elif shape == "curve5":
        X = sample_closed_curve_solid(n, num_vertices = 5)
    elif shape == "annulus":
        X = sample_annulus(n)    
    X = normalize(X)
    return X



def build_dataset_convexity_regular(N = NUM_POINT_CLOUDS, n = NUM_POINTS):
    shapes = ["triangle", "square", "pentagon", "circle",    # 1 (convex)
              "curve3", "curve4", "curve5", "annulus"]       # 0 (concave)
    convexity = [1, 1, 1, 1, 0, 0, 0, 0]    
    
    point_clouds = np.zeros((N, n, 2))
    labels = np.zeros(N)
    num_shapes = len(shapes)
    num_point_clouds_shape = int(N / num_shapes)
    for i, shape in enumerate(shapes):        
        for j in range(num_point_clouds_shape):   
            point_cloud = sample_point_cloud_solid(n, shape)   
            label = convexity[i]
            point_clouds[num_point_clouds_shape * i + j] = point_cloud
            labels[num_point_clouds_shape * i + j] = label
               
        # An example of a point cloud of a given shape (the last generated point cloud).
        # plots.plot_point_cloud(point_cloud, title = shape + ", label = " + str(label) + ", num_points = " + str(point_cloud.shape[0]))
    
    print("Dataset of 'regular' point clouds for the detection of convexity:")
    print("num_shapes = ", num_shapes)
    print("point_clouds.shape = ", point_clouds.shape)
    print("labels.shape = ", labels.shape)
    print("Numbers of point clouds with each label value: ", collections.Counter(labels))
    print()
    return point_clouds, labels



def sample_random_point_cloud(n = 100, x_min = 0, x_max = 1, y_min = 0, y_max = 1):
    x = np.random.uniform(x_min, x_max, size = n)
    y = np.random.uniform(y_min, y_max, size = n)
    X = np.array([x, y]).T    
    return X


   
def sample_random_point_cloud_within_hull(hull, n = 100):
    bbox = [hull.min_bound, hull.max_bound]
    hull_path = Path( hull.points[hull.vertices] )    
    
    X = np.empty((n, 2))
    for i in range(n):
        
        # Draw a random point in the bounding box of the convex hull.
        X[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])

        # Check if the random point is inside the convex hull, otherwise raw it again.            
        while hull_path.contains_point(X[i]) == False:
            X[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
    
    return X



def sample_random_point_cloud_within_polygon(polygon, n = 100):
    x_min, y_min, x_max, y_max = polygon.bounds
    
    X = np.empty((n, 2))
    for i in range(n):
        
        # Draw a random point in the bounding box of the polygon.
        point = Point(np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max))       

        # Check if the random point is inside the polygon, otherwise draw it again.            
        while polygon.contains(point) == False:
            point = Point(np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max))    

        X[i] = np.asarray(point)
        
    return X


    
def sample_random_convex_point_cloud(n = 100, x_min = 0, x_max = 1, y_min = 0, y_max = 1):
    
    # 1) Generate only a few points to build the convex hull from.
    points = sample_random_point_cloud(n = 10, x_min = 0, x_max = 1, y_min = 0, y_max = 1)
    
    # 2) Build the convex hull, type(hull) =  <class 'scipy.spatial.qhull.ConvexHull'>
    hull = ConvexHull(points)
    
    # 3) Sample point cloud with the given number of points inside the hull.
    X = sample_random_point_cloud_within_hull(hull, n)
    
    return X



# Auxiliary function to test concavity.
def same_rows(a, b, tol = 5):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return (np.all(np.any(rows_close, axis=-1), axis=-1) and
            np.all(np.any(rows_close, axis=0), axis=0))



def sample_random_concave_point_cloud(n = 100, x_min = 0, x_max = 1, y_min = 0, y_max = 1):
    
    # Generating alpha shape from a random sample of points provides no guarantee that the resulting shape is concave.
    # We therefore repeat the procedure as long as the shape obtained is convex, i.e.
    # as long as the alpha polygon is the same as the convex hull of the given points.   
    while True:
        
         # 1) Generate only a few points to build the convex hull from.   
        points = sample_random_point_cloud(n = 10, x_min = 0, x_max = 1, y_min = 0, y_max = 1) 
    
        # 2) Build the concave hull, type(hull) =  <class 'shapely.geometry.polygon.Polygon'>.  
        alpha = 1 * alphashape.optimizealpha(points) # Calculation of optimal alpha is optional, and takes a lot of time if the number of points is > 10.
        polygon = alphashape.alphashape(points, alpha) 
        
        # 3) Retrieve the vertices of the polygon, type(polygon) =  <class 'shapely.geometry.polygon.Polygon'>.  
        x, y = polygon.exterior.coords.xy
        x = np.asarray(x)
        y = np.asarray(y)
        vertices_polygon = np.array([x, y]).T   
    
        # 4) Retrieve the vertices of the convex hull of the polygon, type(hull) =  <class 'scipy.spatial.qhull.ConvexHull'>
        convex_hull = ConvexHull(points)
        vertices_convex_hull = convex_hull.points[convex_hull.vertices]
    
        # Check if the the sets of vertices of the polygon and the convex hull are different, implying that the polygon is indeed concave.
        if not same_rows(vertices_polygon, vertices_convex_hull):           
            break

    X = sample_random_point_cloud_within_polygon(polygon, n)   
    return X



def build_dataset_convexity_random(N = NUM_POINT_CLOUDS, n = NUM_POINTS):    
    point_clouds = np.zeros((N, n, 2))
    labels = np.zeros(N)
    
    N_convex = int(N/2)
    
    for i in range(0, N_convex):           
        point_clouds[i] = sample_random_convex_point_cloud(n)    
        labels[i] = 1 
        plots.plot_point_cloud(point_clouds[i], title = "i = " + str(i) + ", label = " + str(labels[i]))
    
    for i in range(N_convex, N):           
        point_clouds[i] = sample_random_concave_point_cloud(n)    
        labels[i] = 0
        plots.plot_point_cloud(point_clouds[i], title = "i = " + str(i) + ", label = " + str(labels[i]))
    
    print("Dataset of 'random' point clouds for the detection of convexity:")
    print("point_clouds.shape = ", point_clouds.shape)
    print("labels.shape = ", labels.shape)
    print("Numbers of point clouds with each label value: ", collections.Counter(labels))
    print()
    return point_clouds, labels



############################################## GENERAL #################################################################



def normalize(X):
    dim = X.shape[1]
    for i in range(dim):
        X[:, i] = ( X[:, i] - np.min(X[:, i]) ) / ( np.max(X[:, i]) - np.min(X[:, i]) )
    return X                               
    # X = X - np.min(X)
    # X = X / np.max(X)
    # return (X - np.min(X)) / (np.max(X) - np.min(X))X
    


def translate(X, tx = None, ty = None, tz = None):
    if tx is None and ty is None and tz is None:
        tx = np.random.uniform(-1, 1)
        ty = np.random.uniform(-1, 1)
        tz = np.random.uniform(-1, 1)
    X_trnsf = np.zeros(X.shape)
    dim = X.shape[1]
    if dim == 2:
        for i in range(X.shape[0]):
            X_trnsf[i] = [X[i][0] + tx, X[i][1] + ty]
    elif dim == 3:
        for i in range(X.shape[0]):
            X_trnsf[i] = [X[i][0] + tx, X[i][1] + ty, X[i][2] + tz]        
    return X_trnsf



def transform(X, trnsf_matrix):
    X_trnsf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        X_trnsf[i] = np.matmul(trnsf_matrix, X[i])
        # X_trnsf[i] = np.dot(X.reshape((-1, X.shape[1])), trnsf_matrix) # https://github.com/charlesq34/pointnet/blob/master/provider.py, line 50
    return X_trnsf



def rotate(X, t = None):
    if t is None:
        t = np.random.uniform(-np.pi/9, np.pi/9) # as in affNIST, angle in [-20 degrees, 20 degrees] = [-pi/9 rad, pi/9 rad]
    dim = X.shape[1]
    if dim == 2: 
        rotation_matrix = np.asarray([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    if dim == 3:
        rotation_matrix = np.asarray([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    X = transform(X, rotation_matrix)
    return X



def stretch(X, t = None):
    if t is None:
        t = np.random.uniform(0.8, 1.2) # as in affNIST
    dim = X.shape[1]
    stretch_matrix = np.identity(dim)
    stretch_matrix[0, 0] = t
    X = transform(X, stretch_matrix)
    return X



def shear(X, t = None):
    if t is None:
        t = np.random.uniform(-0.2, 0.2) # as in affNIST     
    dim = X.shape[1]
    shear_matrix = np.identity(dim)
    shear_matrix[0, 1] = t
    X = transform(X, shear_matrix)
    return X



def gaussian(X, std = 0):
    if std == 0:
        std = np.random.uniform(0, 0.01)  
    X_gauss = X + np.random.normal(0, std, size = X.shape) 
    return X_gauss



def outliers(X, perc_outliers = None):
    if perc_outliers is None:
        perc_outliers = np.random.uniform(0, 0.1)     
    # We do not add outliers to the point cloud, but rather replace some of the existing point cloud points with outliers,
    # so that any point cloud in the dataset would have the same number of points.
    X_out = np.copy(X)
    num_points = X.shape[0]
    num_outliers = int(perc_outliers * num_points)
    indices = np.random.choice(np.arange(num_points), size = num_outliers) 
    dim = X.shape[1]
    for i in indices:      
        for j in range(dim):
            X_out[i][j] = np.random.uniform(np.min(X[:, j]), np.max(X[:, j]))        
    return X_out
    


def calculate_point_clouds_under_trnsf(point_clouds, transformation = " "):
    # point_clouds is a list, rather than an array, since some point clouds are 2D, and some are 3D.
    point_clouds_trnsf = []
    for point_cloud in point_clouds:
        if transformation == "translation":
            point_cloud_trnsf = translate(point_cloud)
        elif transformation == "rotation":
            point_cloud_trnsf = rotate(point_cloud)
        elif transformation == "stretch":
            point_cloud_trnsf = stretch(point_cloud) 
        elif transformation == "shear":
            point_cloud_trnsf = shear(point_cloud)
        elif transformation == "gaussian":
            point_cloud_trnsf = gaussian(point_cloud)  
        elif transformation == "outliers":
            point_cloud_trnsf = outliers(point_cloud)          
        point_clouds_trnsf = point_clouds_trnsf + [point_cloud_trnsf]
    return point_clouds_trnsf



def calculate_3d_point_clouds(point_clouds):  
    num_samples = len(point_clouds)
    num_points = point_clouds[0].shape[0]
    point_clouds_3d = np.zeros((num_samples, num_points, 3))
    for s in range(num_samples):
        pc = point_clouds[s]               
        # Original PointNet implementation (https://github.com/charlesq34/pointnet)
        # train.py, line 182: current_data = current_data[:, 0:NUM_POINT, :], with NUM_POINT = 1024
        pc = pc[0:1024]             
        dim_pc = pc.shape[1]
        if dim_pc == 2:
            num_points_pc = pc.shape[0]
            for p in range(num_points_pc):
                point_clouds_3d[s][p] = np.asarray([pc[p][0], pc[p][1], 0])
        else:
            point_clouds_3d[s] = pc
    print("point_clouds_3d.shape = ", point_clouds_3d.shape)
    return point_clouds_3d



def calculate_3d_point_clouds_under_trnsfs(point_clouds_trnsfs, trnsfs):
    print("Calculating 3D points clouds from the given 2D and 3D point clouds under any given transformation ...")    
    point_clouds_3d_trnsfs = {}
    for trnsf in trnsfs:
        point_clouds = point_clouds_trnsfs[trnsf]  
        point_clouds_3d = calculate_3d_point_clouds(point_clouds)
        point_clouds_3d_trnsfs[trnsf] = point_clouds_3d
    return point_clouds_3d_trnsfs



def flatten_symmetric_matrices(matrices):           
    num_matrices = matrices.shape[0]
    matrix_size = matrices.shape[1]    
    num_above_diag = int( (matrix_size-1)*matrix_size / 2 )
    matrices_flat = np.zeros((num_matrices, num_above_diag))
    for m in range(num_matrices):
        matrices_flat[m] = matrices[m][np.triu_indices(matrix_size, k=1)] 
    return matrices_flat



def calculate_distance_matrices_flat(point_clouds):
    # print("Calculating distance matrices (above the diagonal, and flattened) from the given point clouds...")
    # If point clouds have more than 1000 points, calculating distance matrices because computationally too demanding, but also redundant.
    point_clouds_sparse = []
    num_samples = len(point_clouds)
    for s in range(num_samples):
        point_clouds_sparse.append(point_clouds[s][0:100, :])    
    num_samples = len(point_clouds)
    num_features = point_clouds[0].shape[0]
    distance_matrices = np.zeros((num_samples, 100, 100))
    for s in range(num_samples):
        distance_matrices[s] = euclidean_distances(point_clouds_sparse[s])
        distance_matrices[s] = np.around(distance_matrices[s], 3)          
    distance_matrices_flat = flatten_symmetric_matrices(distance_matrices)     
    # distance_matrices_flat = preprocessing.StandardScaler().fit_transform(distance_matrices_flat)   
    print("distance_matrices_flat.shape = ", distance_matrices_flat.shape)
    return distance_matrices_flat



def calculate_distance_matrices_flat_under_trnsfs(point_clouds_trnsfs, trnsfs):    
    print("Calculating distance matrices for point clouds under any given transformation ...")    
    distance_matrices_flat_trnsfs = {}
    for trnsf in trnsfs:
        point_clouds = point_clouds_trnsfs[trnsf]        
        distance_matrices_flat = calculate_distance_matrices_flat(point_clouds)        
        distance_matrices_flat_trnsfs[trnsf] = distance_matrices_flat 
    return distance_matrices_flat_trnsfs



def encode_labels(labels):
    print("Label encoding into consecutive integers, e.g., 0, 1, 2, 4, 9 -> 0, 1, 2, 3, 4.")
    print("Before encoding, label values are: ", np.unique(labels))
    label_encoder = LabelEncoder() # We have to define it, to be able to reference it later for the inverse transform: labels = label_encoder.inverse_transform(labels_con)
    labels_con = label_encoder.fit_transform(labels)
    labels_con = np.asarray(labels_con)
    print("After encoding, label values are: ", np.unique(labels_con))
    print("labels_con.shape = ", labels_con.shape)     
    return labels_con, label_encoder



def calculate_rank_vector(vector):
    a = {}
    rank = 1
    for num in sorted(vector):
        if num not in a:
            a[num] = rank
            rank = rank+1
    rank_vector = [a[i] for i in vector]
    rank_vector = np.asarray(rank_vector)
    return rank_vector



def calculate_rank_matrix(matrix):
    matrix_flat = matrix.flatten()   
    rank_matrix_flat = calculate_rank_vector(matrix_flat)
    rank_matrix = rank_matrix_flat.reshape(matrix.shape)
    return rank_matrix



def import_point_clouds(name = None):      
    print("Importing point clouds...")   
    if name == "holes" or name == "curvature" or name == "convexity":
        with open(PATH_CURRENT + "DATASETS/" + name + "/point_clouds.pkl", "rb") as f:
            point_clouds = pickle.load(f)
    else:
        print("Error: The data to import can only be one of the saved datasets: holes, curvature or convexity!")
    print("type(point_clouds) = ", type(point_clouds))
    print("len(point_clouds) = ", len(point_clouds))
    print("point_clouds[0].shape = ", point_clouds[0].shape)
    return point_clouds



def import_distance_matrices(name = None):      
    print("Importing distance matrices...")   
    if name == "holes" or name == "curvature" or name == "convexity":
        with open(PATH_CURRENT + "DATASETS/" + name + "/distance_matrices.pkl", "rb") as f:
            distance_matrices = pickle.load(f)
    else:
        print("Error: The data to import can only be one of the saved datasets: holes, curvature or convexity!")
    print("distance_matrices.shape = ", distance_matrices.shape)
    return distance_matrices



def import_distance_matrices_flat(name = None):      
    print("Importing distance matrices (flat)...")   
    if name == "holes" or name == "curvature" or name == "convexity":
        with open(PATH_CURRENT + "DATASETS/" + name + "/distance_matrices_flat.pkl", "rb") as f:
            distance_matrices_flat = pickle.load(f)
    else:
        print("Error: The data to import can only be one of the saved datasets: holes, curvature or convexity!")
    print("distance_matrices_flat.shape = ", distance_matrices_flat.shape)
    return distance_matrices_flat



def import_labels(name = None):      
    print("Importing labels...")   
    if name == "holes" or name == "curvature" or name == "convexity":
        with open(PATH_CURRENT + "DATASETS/" + name + "/labels.pkl", "rb") as f:
            labels = pickle.load(f) 
    else:
        print("Error: The data to import can only be one of the saved datasets: holes, curvature or convexity!")
    print("type(labels) = ", type(labels))
    print("len(labels) = ", len(labels))
    print("Number of point clouds with each label value: ", collections.Counter(np.round(labels, 2)))    
    return labels



def import_labels_shape(name = None):      
    print("Importing labels_shape ...")   
    if name == "holes" or name == "curvature" or name == "convexity":
        with open(PATH_CURRENT + "DATASETS/" + name + "/labels_shape.pkl", "rb") as f:
            labels_shape = pickle.load(f) 
    else:
        print("Error: The data to import can only be one of the saved datasets: holes, curvature or convexity!")
    print("type(labels_shape) = ", type(labels_shape))
    print("len(labels_shape) = ", len(labels_shape))  
    return labels_shape



def import_train_and_test_indices(name = None):
    print("Importing train and test indices...") 
    if name == "holes" or name == "curvature" or name == "convexity":
        with open(PATH_CURRENT + "DATASETS/" + name + "/train_indices.pkl", "rb") as f:
            train_indices = pickle.load(f) 
        with open(PATH_CURRENT + "DATASETS/" + name + "/test_indices.pkl", "rb") as f:
            test_indices = pickle.load(f)     
    else:
        print("Error: The data to import can only be one of the saved datasets: holes, curvature or convexity!")
    print("train_indices.shape = ", train_indices.shape)
    print("test_indices.shape = ", test_indices.shape)    
    return train_indices, test_indices



def image_to_point_cloud(image):  
    num_black_pixels = np.sum(image)
    point_cloud = np.zeros((num_black_pixels, 2))
    point = 0       
    num_x_pixels = image.shape[0]
    num_y_pixels = image.shape[1]
    for i in range(num_x_pixels):
        for j in range(num_y_pixels):
            if image[i, j] > 0:
                point_cloud[point, 0] = j
                point_cloud[point, 1] = num_y_pixels - i
                point = point + 1        
    return point_cloud



def point_cloud_to_image(point_cloud, num_x_pixels = 20, num_y_pixels = 20):
    image = np.zeros((num_x_pixels, num_y_pixels), dtype = np.int8)    
    
    x_coords = point_cloud[:, 0]
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    x_bin_width = (x_max - x_min) / num_x_pixels
    
    y_coords = point_cloud[:, 1]
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)    
    y_bin_width = (y_max - y_min) / num_y_pixels
    
    num_points = point_cloud.shape[0]
    for i in range(num_points):        
        x = x_coords[i]
        v = int( (x-x_min) / x_bin_width)        
        y = y_coords[i]
        u = int( (y-y_min) / y_bin_width)
        u = num_x_pixels - u
        if u == num_x_pixels:
            u = num_x_pixels - 1
        if v == num_y_pixels:
            v = num_y_pixels - 1             
        image[u, v] = 1        

    return image