import numpy as np
import math
from scipy import sparse
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import euclidean_distances
import gudhi as gd
from gudhi.weighted_rips_complex import WeightedRipsComplex
from gudhi.dtm_rips_complex import DTMRipsComplex
from ripser import ripser
import matplotlib.pyplot as plt
import time

import data_construction
import plots





def plot_filtration(X, fil_complex = "rips", fil_fun = "distance", fil_fun_vals = None):
    if fil_complex == "rips" or fil_complex == "alpha":
        plot_simplicial_filtration(X, fil_complex, fil_fun)
    elif fil_complex == "cubical":
        plot_cubical_filtration(fil_fun_vals)
    else:
        print("Error: The filtration can only be plotted for fil_complex in {rips, alpha, cubical}!")
        
        

######################################## GENERAL PH FUNCTIONS, BEGIN #################################################


        
def calculate_pds(X, fil_complex = "rips", fil_fun = "distance", m = 0.001, p = 1):    
    if fil_complex == "rips" or fil_complex == "alpha":
        simplex_tree = calculate_simplicial_simplex_tree(X, fil_complex, fil_fun, m, p)                 
    elif fil_complex == "cubical":
        simplex_tree, _ = calculate_cubical_simplex_tree(X, fil_fun)        
    simplex_tree.persistence()     
    pd0 = simplex_tree.persistence_intervals_in_dimension(0)    
    # pd0 = pd0[0:(len(pd0)-1)] # Do not consider the last interval with d = np.inf.    
    pd1 = simplex_tree.persistence_intervals_in_dimension(1)
    return pd0, pd1



def calculate_pds_point_clouds(point_clouds, fil_complex = "rips", fil_fun = "distance", m = 0.001, p = 1):
    pds0 = []
    pds1 = []
    for X in point_clouds:
        pd0, pd1 = calculate_pds(X, fil_complex, fil_fun, m, p)
        pds0.append(pd0)
        pds1.append(pd1)   
    
    # Transform list of 0-dim PDs with different number of cycles into an array of PDs with the same number of cycles.     
    pds0_length = [len(pd) for pd in pds0]
    max_pd_length  = max(pds0_length)    
    pds0 = extend_pds_to_length(pds0, max_pd_length)
    
    # Transform list of 1-dim PDs with different number of cycles into an array of PDs with the same number of cycles.    
    pds1_length = [len(pd) for pd in pds1]
    max_pd_length  = max(pds1_length)   
    pds1 = extend_pds_to_length(pds1, max_pd_length)
    
    # print("pds0.shape = ", pds0.shape)
    # print("pds1.shape = ", pds1.shape)
    return pds0, pds1



def calculate_pds_distance_matrices(distance_matrices):
    print("Calculating persistence diagrams (PDs) for the given dataset of distance matrices...")
    t0 = time.time()
    pds0 = []
    pds1 = []
    for dis_mat in distance_matrices:
        rips_complex = gd.RipsComplex(distance_matrix = dis_mat) #, max_edge_length = max_distance
        simplex_tree = rips_complex.create_simplex_tree(max_dimension = 2)
        simplex_tree.persistence() 
        pd0 = simplex_tree.persistence_intervals_in_dimension(0)        
        pd0 = pd0[0:(len(pd0)-1)] # Do not consider the last interval with d = np.inf.        
        pd1 = simplex_tree.persistence_intervals_in_dimension(1)
        pds0.append(pd0)
        pds1.append(pd1)
        print("PD is calculated for a distance matrix!")
    t1 = time.time()  
    
    # Transform list of 0-dim PDs with different number of cycles into an array of PDs with the same number of cycles.    
    pds0_length = [len(pd) for pd in pds0]
    max_pd_length  = max(pds0_length)        
    pds0 = extend_pds_to_length(pds0, max_pd_length)
    
    # Transform list of 1-dim PDs with different number of cycles into an array of PDs with the same number of cycles.    
    pds1_length = [len(pd) for pd in pds1]
    max_pd_length  = max(pds1_length)   
    pds1 = extend_pds_to_length(pds1, max_pd_length)

    print("pds0.shape = ", pds0.shape)
    print("pds1.shape = ", pds1.shape)
    print("Runtime = ", np.around(t1-t0, 2),  "seconds.") 
    return pds0, pds1



def calculate_sparse_distance_matrix(distance_matrix, threshold = np.inf):
    num_points = distance_matrix.shape[0]
    [I, J] = np.meshgrid(np.arange(num_points), np.arange(num_points))
    I = I[distance_matrix <= threshold]
    J = J[distance_matrix <= threshold]
    V = distance_matrix[distance_matrix <= threshold]
    return sparse.coo_matrix((V, (I, J)), shape=(num_points, num_points)).tocsr()



def calculate_pds_distance_matrices_ripser(distance_matrices):
    print("Calculating persistence diagrams (PDs) for the given dataset of distance matrices (using Ripser)...")
    t0 = time.time()
    pds0 = []
    pds1 = []
    for dis_mat in distance_matrices:
        dis_mat_sparse = calculate_sparse_distance_matrix(dis_mat)
        pds = ripser(dis_mat_sparse, distance_matrix = True)
        pd0 = pds["dgms"][0]
        pd0 = pd0[0:(len(pd0)-1)] # Do not consider the last interval with d = np.inf.   
        pd1 = pds["dgms"][1]
        pds0.append(pd0)
        pds1.append(pd1)
    t1 = time.time() 
    
    # Transform list of 0-dim PDs with different number of cycles into an array of PDs with the same number of cycles.    
    pds0_length = [len(pd) for pd in pds0]
    max_pd_length  = max(pds0_length)        
    pds0 = extend_pds_to_length(pds0, max_pd_length)
    
    # Transform list of 1-dim PDs with different number of cycles into an array of PDs with the same number of cycles.    
    pds1_length = [len(pd) for pd in pds1]
    max_pd_length  = max(pds1_length)   
    pds1 = extend_pds_to_length(pds1, max_pd_length)

    print("pds0.shape = ", pds0.shape)
    print("pds1.shape = ", pds1.shape)
    print("Runtime = ", np.around(t1-t0, 2),  "seconds.")   
    return pds0, pds1



# Transform list of PDs with different number of cycles into an array of PDs with the same number of cycles.
def extend_pds_to_length(pds, length):
    pds_ext = np.zeros((len(pds), length, 2))
    for s, pd in enumerate(pds):
        for i in range(length):
            if i < len(pd):
                pds_ext[s][i] = pds[s][i]
            else:
                pds_ext[s][i] = np.asarray([0, 0]) 
    return pds_ext



def num_persistent_cycles(PD, treshold_lifespan):    
    num = 0
    for i in range(PD.shape[0]):
        b = PD[i][0]
        d = PD[i][1]
        l = d - b
        if l > treshold_lifespan:
            num = num + 1
    return num



def sorted_lifespans_pd(PD, size):
    lifespans = []
    for i in range(PD.shape[0]):
        b = PD[i][0]
        d = PD[i][1]
        l = d - b
        lifespans.append(l)   
    lifespans.sort(reverse = True)   

    length = len(lifespans)
    if length < size:
        for i in range(length, size):
            lifespans = lifespans + [0]
    lifespans = np.around(lifespans, 5)
    
    return lifespans[0:size]



def sorted_lifespans_pds(pds, size = None):
    if size is None:
        pds_len = [len(pd) for pd in pds]
        size = max(pds_len)        
    lifespans_pds = [sorted_lifespans_pd(pd, size) for pd in pds]
    lifespans_pds = np.asarray(lifespans_pds)    
    return lifespans_pds



######################################## GENERAL PH FUNCTIONS, END #################################################





################################## PH ON SIMPLICIAL COMPLEXES, BEGIN ###############################################



# Greater values of $p$ tend to sparsify the persistence diagram.
# https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-DTM-filtrations.ipynb 
def dtm(X, query_pts, m = 0.01):
    '''
    Computes the values of the DTM (with exponent p = 2) of the empirical measure of a point cloud X. 
    Requires sklearn.neighbors.KDTree to search nearest neighbors.
    https://github.com/raphaeltinarrage/velour/blob/main/velour/geometry.py
    
    Input:
        X:          n x d numpy array representing n points in R^d.
        query_pts:  k x n numpy array of query points.
        m:          float in [0, 1]), DTM parameter reflecting the number of neighbors.
    
    Output: 
        dtm_vals:    k x 1 numpy array, DTM of the query points.
    
    Example:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        Q = np.array([[0,0], [5,5]])
        dtm_vals = dtm(X, Q, 0.3)
    '''
    n = X.shape[0]     
    k = math.floor(m * n) + 1
    kdt = KDTree(X, leaf_size = 30, metric = "euclidean")
    NN_Dist, NN = kdt.query(query_pts, k, return_distance = True)  
    dtm_vals = np.sqrt(np.sum(NN_Dist * NN_Dist, axis=1) / k)
    return dtm_vals



def fil_fun_val_edge_weighted_rips(fx, fy, d, p = np.inf, num_iterations = 10):
    '''
    Computes the filtration value of the edge [x,y] in the weighted Rips filtration.
    If p is not 1, 2 or 'np.inf, an implicit equation is solved.
    The equation to solve is G(I) = d, where G(I) = (I**p-fx**p)**(1/p)+(I**p-fy**p)**(1/p).
    We use a dichotomic method.
    https://github.com/raphaeltinarrage/velour/blob/main/velour/persistence.py
    
    Input:
        fx:          float, filtration function value of vertex x
        fy:          float, filtration function value of vertex y
        d:           float, distance between vertices x and y
        p:           float in [1, np.inf], parameter of the weighted Rips filtration
        num_iter:    int, optional number of iterations of the dichotomic method
        
    Output: 
        val:         float, filtration function value of the edge [x,y], i.e. solution of G(I) = d.
    '''
    
    if p == np.inf:
        fxy = max([fx, fy, d/2])
    else:
        fmax = max([fx, fy])
        if d < (abs(fx**p - fy**p))**(1/p):
            fxy = fmax
        elif p == 1:
            fxy = (fx + fy + d)/2
        elif p == 2:
            fxy = np.sqrt( ( (fx + fy)**2 + d**2 )*( (fx - fy)**2 +d**2 ) ) / (2*d)            
        else:
            Imin = fmax
            Imax = (d**p + fmax**p)**(1/p)
            for i in range(num_iterations):
                I = (Imin + Imax) / 2
                g = (I**p - fx**p)**(1/p) + (I**p - fy**p)**(1/p)
                if g < d:
                    Imin=I
                else:
                    Imax=I
            fxy = I
    
    return fxy



def calculate_simplicial_simplex_tree(X, sim_complex = "rips", fil_fun = "distance", m = 0.001, p = 1): # max_dim  = 2, filtration_max = np.inf):
    '''
    This is a heuristic method, that speeds-up the computation.
    It computes the DTM-filtration seen as a subset of the Delaunay filtration.
    https://github.com/raphaeltinarrage/velour/blob/main/velour/persistence.py
    
    Input:
        X:                 n x d numpy array, representing n points in R^d.
        m:                 float in [0, 1), DTM parameter. 
        p:                 float in [0, np.inf]), DTM-filtration parameter. 
        max_dim:           int, optional, maximal dimension to expand the complex.
        max_fil_fun_val:   float, optional, maximal filtration value of the filtration.
    
    Output:
        simplex_tree:      gudhi.SimplexTree.
    '''       
        
    if sim_complex == "rips":
        simplicial_complex = gd.RipsComplex(X, max_edge_length = np.inf)
        simplex_tree_distance = simplicial_complex.create_simplex_tree(max_dimension = 2) 
        Y = np.copy(X)        
    elif sim_complex == "alpha":
        # PH on AlphaComplex is computationally much less demanding than PH on RipsComplex.
        # https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-ConfRegions-PersDiag-datapoints.ipynb:
        # When computing confidence regions for alpha complexes, we need to be careful with the 
        # scale of values of the filtration because the filtration value of each simplex is computed as 
        # the square of the circumradius of the simplex (if the circumsphere is empty).
        # https://github.com/GUDHI/TDA-tutorial/blob/master/utils/utils_quantization.py        
        simplicial_complex = gd.AlphaComplex(X)
        simplex_tree_distance = simplicial_complex.create_simplex_tree() # Although not yet necessary, it must be calculated in order to run simplicial_complex.get_point().
        
        # Y = np.array([simplicial_complex.get_point(i) for i in range(X.shape[0])]) # gudhi.AlphaComplex may change the ordering of the points.
        # In a rare scenario, it can happen that X.shape[0] != gd.AlphaComplex(X).create_simplex_tree().num_vertices(), returning an error.
        num_pc_points = X.shape[0]
        num_simplex_points = simplex_tree_distance.num_vertices()
        num_query_points = min(num_pc_points, num_simplex_points)
        Y = np.array([simplicial_complex.get_point(i) for i in range(num_query_points)]) # gudhi.AlphaComplex may change the ordering of the points.

    if fil_fun == "distance":
        num_vertices = X.shape[0]
        fil_fun_vals_vertices = [0] * num_vertices
    elif fil_fun == "dtm":
        # PH is not sensitive to outliers.
        fil_fun_vals_vertices = dtm(X, Y, m) 
    elif fil_fun == "height":
         fil_fun_vals_vertices = 1- X[:, 1]
            
    # print("fil_fun_vals_vertices = ", np.around(fil_fun_vals_vertices, 2))    
     # Rips and DTM filtration function: simplicial_complex = DTMRipsComplex(X, dis_matrix, k = 1, q = 2)
    # Rips and any filtration function: simplicial_complex = WeightedRipsComplex(distance_matrix = dis_matrix, weights = fil_fun_vals_vertices)
    # Similarly to the gudhi implementation of WeightedRipsComplex (p = np.inf, https://gudhi.inria.fr/python/latest/_modules/gudhi/weighted_rips_complex.html#WeightedRipsComplex)
    # all the filtration values are doubled compared to the definition in the paper for the consistency with RipsComplex.    
    # fx = 2 * wx
    # fxy = 2 * fxy
    
    simplex_tree_filtration = gd.SimplexTree()
    dis_matrix = euclidean_distances(Y)          
    for (simplex, _) in simplex_tree_distance.get_skeleton(2):  
        # Add vertices.
        if len(simplex) == 1:
            i = simplex[0]
            simplex_tree_filtration.insert([i], filtration = 2 * fil_fun_vals_vertices[i])
        # Add edges.
        if len(simplex) == 2:                               
            i = simplex[0]
            j = simplex[1]
            fil_fun_val_edge = fil_fun_val_edge_weighted_rips(fil_fun_vals_vertices[i], fil_fun_vals_vertices[j], dis_matrix[i, j], p)        
            simplex_tree_filtration.insert([i,j], filtration = 2 * fil_fun_val_edge)

       
    # PH is not sensitive to the range of x, y, and z values for the point cloud coordinates, or to scaling.
    # The lifespan will reflect the size of the cycles, and since shapes can be plotted in different ranges.
    # we want to scale relative to the point cloud.
    # xmin = np.min(X[:, 0])
    # xmax = np.max(X[:, 0])
    # ymin = np.min(X[:, 1])
    # ymax = np.max(X[:, 1])
    # max_dist = np.sqrt((xmax - xmin)**2 + (ymax-ymin)**2)     
    fil_fun_vals = []
    for (simplex, fil_fun_val) in simplex_tree_filtration.get_skeleton(2):
        fil_fun_vals.append(fil_fun_val)
    
    min_fil_fun_val = min(fil_fun_vals) 
    max_fil_fun_val = max(fil_fun_vals)    
    # print("min_fil_fun_val = ",  np.around(min_fil_fun_val, 2))
    # print("max_fil_fun_val = ",  np.around(max_fil_fun_val, 2))
    simplex_tree_normalized = gd.SimplexTree()                         
    for (simplex, fil_fun_val) in simplex_tree_filtration.get_skeleton(2):     
        if len(simplex) == 1:
            i = simplex[0]
            simplex_tree_normalized.insert([i], filtration = fil_fun_val / max_fil_fun_val)
        if len(simplex) == 2:                               
            i = simplex[0]
            j = simplex[1]
            simplex_tree_normalized.insert([i,j], filtration = fil_fun_val / max_fil_fun_val)
            
    
    # PH is not sensitive to the size of the cycle.
    # It is difficult to guess the suitable threshold perc, better let this be learnt from the data (labels)?
    simplex_tree_binarized = gd.SimplexTree()  
    perc = 0.15
    for (simplex, fil_fun_val) in simplex_tree_normalized.get_skeleton(2):     
        if len(simplex) == 1:
            i = simplex[0]
            if fil_fun_val < perc * max_fil_fun_val:
                simplex_tree_binarized.insert([i], filtration = 0)
            else:
                simplex_tree_binarized.insert([i], filtration = 1)
        if len(simplex) == 2:                               
            i = simplex[0]
            j = simplex[1]
            if fil_fun_val < perc * max_fil_fun_val:
                simplex_tree_binarized.insert([i,j], filtration = 0)
            else:
                simplex_tree_binarized.insert([i,j], filtration = 1)
            
  
    simplex_tree_final = simplex_tree_filtration
    # simplex_tree_final = simplex_tree_normalized
    
    max_dim = 2
    simplex_tree_final.expansion(max_dim)    
    
    # st.prune_above_filtration(filtration_max)
    # or before st.insert() statements above check if  value<filtration_max:
    
    # result_str = 'Alpha Weighted Rips Complex is of dimension ' + repr(st.dimension()) + ' - ' + \
    #     repr(st.num_simplices()) + ' simplices - ' + \
    #     repr(st.num_vertices()) + ' vertices.' +\
    #     ' Filtration maximal value is ' + str(filtration_max) + '.'
    # print(result_str)
    
    # print("simplex_tree.num_vertices() = ", simplex_tree_final.num_vertices())
    # print("simplex_tree.num_simplices() = ", simplex_tree_final.num_simplices())
    # print("simplex_tree = ", print_simplicial_complex(simplex_tree_final))
     
    return simplex_tree_final



def print_simplicial_complex(simplex_tree):
    print("Simplices of length 1, i.e., vertices:")
    for (simplex, fil_fun_val) in simplex_tree.get_skeleton(2):     
        if len(simplex) == 1:
            print("simplex = ", simplex, ",\t filtration function value = ", np.around(fil_fun_val, 2))
    print()
    
    print("Simplices of length 2, i.e., edges:")
    for (simplex, fil_fun_val) in simplex_tree.get_skeleton(2):     
        if len(simplex) == 2:
            print("simplex = ", simplex, ",\t filtration function value = ", np.around(fil_fun_val, 2))
    print()

    print("Simplices of length 3, i.e., triangles:")
    for (simplex, fil_fun_val) in simplex_tree.get_skeleton(2):     
        if len(simplex) == 3:
            print("simplex = ", simplex, ",\t filtration function value = ", np.around(fil_fun_val, 2))
    print()




def plot_simplicial_complex(X, simplex_tree, max_fil_fun_val = np.inf, axes = None, path = " "):
    
    if axes == None:
        axes = plt.axes()
    
    for simplex_fil_fun_val in simplex_tree.get_simplices():
        
        simplex = simplex_fil_fun_val[0]
        fil_fun_val = simplex_fil_fun_val[1]
        
        if fil_fun_val <= max_fil_fun_val:
            # Plot a vertex:    
            if len(simplex) == 1:
                i = simplex[0]
                axes.plot(X[i, 0], X[i, 1], "ro", markersize = 4)
            # Plot an edge:
            elif len(simplex) == 2:
                i = simplex[0]
                j = simplex[1]                  
                edge_vertex_1 = X[i]
                edge_vertex_2 = X[j]
                x_values = [edge_vertex_1[0], edge_vertex_2[0]]
                y_values = [edge_vertex_1[1], edge_vertex_2[1]]                    
                axes.plot(x_values, y_values)   
            # Fill a triangle:
            elif len(simplex) == 3: 
                i = simplex[0]
                j = simplex[1]  
                k = simplex[2]  
                triangle_vertex_1 = X[i]
                triangle_vertex_2 = X[j]
                triangle_vertex_3 = X[k]
                x_values = [triangle_vertex_1[0], triangle_vertex_2[0], triangle_vertex_3[0]]
                y_values = [triangle_vertex_1[1], triangle_vertex_2[1], triangle_vertex_3[1]]                    
                axes.fill(x_values, y_values)    
                
    X_min = np.min(X)
    X_max = np.max(X)
    X_min = -1
    X_max = 1
    axes.set_xlim(X_min, X_max)
    axes.set_ylim(X_min, X_max)    
    axes.set_aspect("equal")
    axes.set_title("K_%.2f" %max_fil_fun_val, fontsize = 40)

    
    
    
def plot_simplicial_complex_3d(X, simplex_tree, max_fil_fun_val = np.inf, axes = None, path = " "):
    
    if axes == None:
        axes = plt.axes(projection = "3d")
    
    for simplex_fil_fun_val in simplex_tree.get_simplices():
        
        simplex = simplex_fil_fun_val[0]
        fil_fun_val = simplex_fil_fun_val[1]
        
        if fil_fun_val <= max_fil_fun_val:
            # Plot a vertex:    
            if len(simplex) == 1:
                i = simplex[0]
                axes.plot(X[i, 0], X[i, 1], X[i, 2], "ro", markersize = 4)
            # Plot an edge:
            elif len(simplex) == 2:
                i = simplex[0]
                j = simplex[1]                  
                edge_vertex_1 = X[i]
                edge_vertex_2 = X[j]
                x_values = [edge_vertex_1[0], edge_vertex_2[0]]
                y_values = [edge_vertex_1[1], edge_vertex_2[1]]                    
                axes.plot(x_values, y_values)   
            # Fill a triangle:
            elif len(simplex) == 3: 
                i = simplex[0]
                j = simplex[1]  
                k = simplex[2]  
                triangle_vertex_1 = X[i]
                triangle_vertex_2 = X[j]
                triangle_vertex_3 = X[k]
                x_values = [triangle_vertex_1[0], triangle_vertex_2[0], triangle_vertex_3[0]]
                y_values = [triangle_vertex_1[1], triangle_vertex_2[1], triangle_vertex_3[1]]                    
                axes.fill(x_values, y_values)    
                
    # X_min = np.min(X)
    # X_max = np.max(X)
    # X_min = -1
    # X_max = 1
    # axes.set_xlim(X_min, X_max)
    # axes.set_ylim(X_min, X_max)
    
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    axes.set_zlim(0, 1)
    
    
    # axes.set_aspect("equal")
    axes.set_title("K_%.2f" %max_fil_fun_val, fontsize = 40)
    plt.show()

    
    
def plot_simplicial_filtration(X, sim_complex = "rips", fil_fun = "distance", m = 0.001, p = 1, max_fil_fun_vals = [0, 0.25, 0.5, 0.75, 1], path = " "):
    
    num_filtered_complexes = len(max_fil_fun_vals)    
    num_fig_rows = 1
    num_fig_cols = num_filtered_complexes
    subfig_width = 3.5
    subfig_height = 3.5
    fig, axes = plt.subplots(num_fig_rows, num_fig_cols, figsize = (num_fig_cols * subfig_width, num_fig_rows * subfig_height)) 
    
#     dim = X.shape[1]
#     if dim == 2:
#         fig, axes = plt.subplots(num_fig_rows, num_fig_cols, figsize = (num_fig_cols * subfig_width, num_fig_rows * subfig_height)) 
#     if dim == 3:
#         fig, axes = plt.subplots(num_fig_rows, num_fig_cols, figsize = (num_fig_cols * subfig_width, num_fig_rows * subfig_height), projection = "3d")
    
    simplex_tree = calculate_simplicial_simplex_tree(X, sim_complex, fil_fun, m, p)
    
    # max_fil_fun_val = 0
    # for simplex_fil_fun_val in simplex_tree.get_simplices():
    #     fil_fun_val = simplex_fil_fun_val[1]
    #     if fil_fun_val > max_fil_fun_val:
    #         max_fil_fun_val = fil_fun_val
    # print("max_fil_fun_val = ", np.around(max_fil_fun_val, 2))   
    
    for i, max_fil_fun_val in enumerate(max_fil_fun_vals):
        plot_simplicial_complex(X, simplex_tree, max_fil_fun_val, axes = axes[i])
        
    # if path != " ":
    #     plt.savefig(path, bbox_inches = "tight")   
    # plt.show()
    
    # PD = simplex_tree.persistence() 
    # gd.plot_persistence_diagram(PD)
    # plt.show()
    # PD0 = simplex_tree.persistence_intervals_in_dimension(0)
    # PD1 = simplex_tree.persistence_intervals_in_dimension(1)  
    # print("PD1 = ", PD1)
    
    plt.show()

    

def calculate_and_plot_simplicial_filtration_and_pds(X, sim_complex, fil_fun, m = 0.001, p = 1,
                                                     max_fil_fun_vals = [0, 0.25, 0.5, 0.75, 1], 
                                                     min_X = -0.25, max_X = 1.25, min_PD = -0.25, max_PD = 1.25):
    num_filtered_complexes = len(max_fil_fun_vals)
    
    num_fig_rows = 1
    num_fig_cols = 1 + num_filtered_complexes + 2
    subfig_width = 2
    subfig_height = 2
    fig, axes = plt.subplots(num_fig_rows, num_fig_cols, figsize = (num_fig_cols * subfig_width, num_fig_rows * subfig_height)) 
    fig.tight_layout(pad = 0.5) 


    # Plot point cloud.
    axes[0].scatter(X[:,0], X[:,1], s = 2.5, c = "blue")       
    axes[0].set_xlim(min_X, max_X)
    axes[0].set_ylim(min_X, max_X)
    axes[0].set_aspect("equal")
    axes[0].set_title("Point cloud X", fontsize = 20)


    # Calculate simplex tree, necessary to calculate and plot the filtration and PDs. 
    t0 = time.time() 
    simplex_tree = calculate_simplicial_simplex_tree(X, sim_complex, fil_fun, m, p)  
    t1 = time.time()
    # print("Runtime for calculation of a simplicial simplex tree =", np.around(t1-t0, 2))

    # Plot filtration.
    for ax, max_fil_fun_val in enumerate(max_fil_fun_vals):     
        for simplex_fil_fun_val in simplex_tree.get_simplices():
            simplex = simplex_fil_fun_val[0]
            fil_fun_val = simplex_fil_fun_val[1]
            if fil_fun_val <= max_fil_fun_val:
                # Plot a vertex:    
                if len(simplex) == 1:
                    i = simplex[0]
                    axes[ax+1].plot(X[i, 0], X[i, 1], "bo", markersize = 2) #, c = "blue")
                # Plot an edge:
                elif len(simplex) == 2:
                    i = simplex[0]
                    j = simplex[1]                  
                    edge_vertex_1 = X[i]
                    edge_vertex_2 = X[j]
                    x_values = [edge_vertex_1[0], edge_vertex_2[0]]
                    y_values = [edge_vertex_1[1], edge_vertex_2[1]]                    
                    axes[ax+1].plot(x_values, y_values)   
                # Fill a triangle:
                elif len(simplex) == 3: 
                    i = simplex[0]
                    j = simplex[1]  
                    k = simplex[2]  
                    triangle_vertex_1 = X[i]
                    triangle_vertex_2 = X[j]
                    triangle_vertex_3 = X[k]
                    x_values = [triangle_vertex_1[0], triangle_vertex_2[0], triangle_vertex_3[0]]
                    y_values = [triangle_vertex_1[1], triangle_vertex_2[1], triangle_vertex_3[1]]                    
                    axes[ax+1].fill(x_values, y_values)       
        axes[ax+1].set_xlim(min_X, max_X)
        axes[ax+1].set_ylim(min_X, max_X)
        axes[ax+1].set_aspect("equal")
        axes[ax+1].set_title("K_%.2f" %max_fil_fun_val, fontsize = 20)
        
    
    # Calculate and plot PD.
    t0 = time.time()
    simplex_tree.persistence() 
    PD0 = simplex_tree.persistence_intervals_in_dimension(0)
    PD1 = simplex_tree.persistence_intervals_in_dimension(1)
    t1 = time.time()
    # print("Runtime for calculation of PDs =", np.around(t1-t0, 2))
    print("0-dim PD = \n", np.around(PD0, 2))
    print("1-dim PD = \n", np.around(PD1, 2))
    print("0-dim PD sorted lifespans = ", np.around(sorted_lifespans_pd(PD0, 10), 2))
    print("1-dim PD sorted lifespans = ", np.around(sorted_lifespans_pd(PD1, 10), 2))
    print()

    # Plot 0-dim PD.
    PD0[PD0 == np.inf] = max_PD    
    axes[num_fig_cols - 2].scatter(PD0[:, 0], PD0[:, 1], 35, c = "green")
    x = np.arange(min_PD, max_PD, 0.01)
    axes[num_fig_cols - 2].plot(x, x, c = 'black') # plot the diagonal  
    intervals_unique, multiplicities = np.unique(np.around(PD0, 2), axis = 0, return_counts = True) 
    for i, multiplicity in enumerate(multiplicities):
        if multiplicity > 1:
            axes[num_fig_cols - 2].annotate(multiplicity, xy = (intervals_unique[i, 0], intervals_unique[i, 1]), 
                         ha = "left", va = "bottom", xytext = (5, 0), textcoords = "offset points", 
                         fontsize = 15, color = "black") 
    axes[num_fig_cols - 2].set_xlim(min_PD, max_PD)
    axes[num_fig_cols - 2].set_ylim(min_PD, max_PD)
    axes[num_fig_cols - 2].set_aspect("equal")
    axes[num_fig_cols - 2].set_title("0-dim PD", fontsize = 20)

    # Plot 1-dim PD.
    PD1[PD1 == np.inf] = max_PD
    axes[num_fig_cols - 1].scatter(PD1[:, 0], PD1[:, 1], 35, c = "green")
    x = np.arange(min_PD, max_PD, 0.01)
    axes[num_fig_cols - 1].plot(x, x, c = 'black') # plot the diagonal  
    intervals_unique, multiplicities = np.unique(np.around(PD1, 2), axis = 0, return_counts = True) 
    # print("Persistence intervals' multiplicities = ", multiplicities)
    for i, multiplicity in enumerate(multiplicities):
        if multiplicity > 1:
            axes[num_fig_cols - 1].annotate(multiplicity, xy = (intervals_unique[i, 0], intervals_unique[i, 1]), 
                             ha = "left", va = "bottom", xytext = (5, 0), textcoords = "offset points", 
                             fontsize = 10, color = "black")   
    axes[num_fig_cols - 1].set_xlim(min_PD, max_PD)
    axes[num_fig_cols - 1].set_ylim(min_PD, max_PD)
    axes[num_fig_cols - 1].set_aspect("equal")
    axes[num_fig_cols - 1].set_title("1-dim PD", fontsize = 20)
    
    
    return fig

################################## PH ON SIMPLICIAL COMPLEXES, END #################################################





################################## PH ON CUBICAL COMPLEXES, BEGIN #################################################


def height_filtration_function(image, x_pixel, y_pixel):  
    '''
    Calculate the radial filtration function values for each image in the dataset.
    Radial filtration function corresponds to the distance from a given reference pixel.
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
     x_pixel, y_pixel: integers, coordinates of the reference pixel.
    
    Output:
    filt_func_vals_data: a num_data_poins x num_pixels numpy array, with each row representing the values of the 
    radial filtration function on each pixel of an image.
    '''
    num_x_pixels = image.shape[0]
    num_y_pixels = image.shape[1]  
    fil_fun_vals = np.zeros((num_x_pixels * num_y_pixels))               
      
    fil_fun_vals = np.zeros((num_x_pixels, num_y_pixels))
    point_reference = np.array([x_pixel, y_pixel])
    for i in range(num_x_pixels):
        for j in range(num_y_pixels):
            point = np.array([i, j])
            fil_fun_vals[i, j] = i
            # fil_fun_vals[i, j] = np.linalg.norm(point - point_reference, ord = 2)  
            # filt_func_vals[i, j] = np.inner(point, point_reference)   
    
    
    fil_fun_vals[image == 0] = 2 * np.max(fil_fun_vals)  
    
    
    fil_fun_vals = fil_fun_vals / np.max(fil_fun_vals)
    
    # plot = plt.matshow(fil_fun_vals, cmap = "hot") # cmap = plt.cm.gray_r)
    # plt.title("filtration\nfunction", fontsize = 20, pad = 10)
    
    # plt.colorbar(plot) #, orientation = "horizontal")
    
    # fig.subplots_adjust(right = 0.835)
    # colorbar_axes = fig.add_axes([-0.01, 0.15, 0.008, 0.72])
    # fig.colorbar(plot, cax = colorbar_axes, shrink = 0.9) 
    # plt.colorbar(plot, cax = colorbar_axes, shrink = 0.9) 
    
    print("height fil_fun_vals = \n", np.around(fil_fun_vals, 2))
    
    fil_fun_vals = fil_fun_vals.reshape((num_x_pixels * num_y_pixels, ))                
    fil_fun_vals = fil_fun_vals / np.max(fil_fun_vals)
    
    return fil_fun_vals


def density_filtration_function(image, max_dist):  
    '''
    Calculate the density filtration function values for each image in the dataset. 
    Density filtration function counts the number of dark-enough pixels in an image, within a given distance.
    
    Input:
    data: a num_data_poins x num_pixels numpy array, with each row representing the greyscale pixel values of an image.
    max_dist: a non-negative integer, representing the size of the considered neighborhood for each pixel in an image.
    
    Output:
    filt_func_vals_data: a num_data_poins x num_pixels numpy array, with each row representing the values of the 
    density filtration function on each pixel of an image.
    '''
    num_x_pixels = image.shape[0]
    num_y_pixels = image.shape[1]
    fil_fun_vals = np.zeros((num_x_pixels * num_y_pixels))    
    point_cloud_complete = np.zeros((num_x_pixels * num_y_pixels, 2))
    p = 0
    for i in range(num_x_pixels):
        for j in range(num_y_pixels):
            point_cloud_complete[p, 0] = j
            point_cloud_complete[p, 1] = num_y_pixels - i
            p = p + 1            
    point_cloud = image_to_point_cloud(image)        
    kdt = KDTree(point_cloud, leaf_size = 30, metric = "euclidean") 
    num_nbhs = kdt.query_radius(point_cloud_complete, r = max_dist, count_only = True)
    fil_fun_vals = num_nbhs
    max_num_nbhs = 2 * max_dist**2 + 2 * max_dist + 1 # num of pixels in euclidean ball with radius max_dist
    fil_fun_vals = max_num_nbhs - fil_fun_vals
    
    
    fil_fun_vals[image.reshape((num_x_pixels * num_y_pixels, )) == 0] = 2 * np.max(fil_fun_vals) 
    
    
    # Cut-off density value (so that density does not also reflect the size of the hole).
    # max_filt_func_val = np.max(filt_func_vals)
    # filt_func_vals[filt_func_vals > 0.65 * max_filt_func_val] = max_filt_func_val # 0.4 in carriere2018statistical  
    fil_fun_vals = fil_fun_vals / np.max(fil_fun_vals)    
    
    print("density fil_fun_vals = ", np.around(fil_fun_vals, 2))
    return fil_fun_vals



def calculate_cubical_simplex_tree(X, fil_fun = "binary", num_x_pixels = 10, num_y_pixels = 10):
    image = data_construction.point_cloud_to_image(X, num_x_pixels, num_y_pixels)
    # plt.matshow(image)
    # plt.show()

    if fil_fun == "binary": 
        fil_fun_vals = 1 - image.reshape((num_x_pixels * num_y_pixels, ))  
    elif fil_fun == "height":
        fil_fun_vals = height_filtration_function(image, x_pixel = 0, y_pixel = int(num_y_pixels/2))    
    elif fil_fun == "height-density":
        fil_fun_vals = 1 * height_filtration_function(image, x_pixel = 0, y_pixel = int(num_y_pixels/2)) + 5 * density_filtration_function(image, max_dist = 1)
        fil_fun_vals = fil_fun_vals / np.max(fil_fun_vals)
    else:
        print("Ërror: The cubical complex is only implemented for fil_fun in {binary, height}!")
        return
    
    # print("Plotting filtered cubical complexes... ")
    # print("fil_fun_vals = \n", np.around(fil_fun_vals.reshape((num_x_pixels, num_y_pixels)), 2))
    # plot_cubical_filtration(fil_fun_vals.reshape((num_x_pixels, num_y_pixels)))

    
    cubical_complex = gd.CubicalComplex(dimensions = [num_x_pixels, num_y_pixels], top_dimensional_cells = fil_fun_vals)
    simplex_tree = cubical_complex
    print("simplex_tree.num_simplices() = ", simplex_tree.num_simplices())
    return simplex_tree, fil_fun_vals.reshape((num_x_pixels, num_y_pixels))
    

    
def plot_cubical_filtration(fil_fun_vals, fil_fun_vals_percs = [0.1, 0.2, 0.4, 0.6, 0.8], path = " "):
    
    num_filtered_complexes = len(fil_fun_vals_percs)    
    num_fig_rows = 1
    num_fig_cols = num_filtered_complexes
    subfig_width = 3.5
    subfig_height = 3.5
    fig, axes = plt.subplots(num_fig_rows, num_fig_cols, figsize = (num_fig_cols * subfig_width, num_fig_rows * subfig_height)) 
    
    nx = fil_fun_vals.shape[0]
    ny = fil_fun_vals.shape[1]
    max_fil_fun_val = np.max(fil_fun_vals)
    for p, perc in enumerate(fil_fun_vals_percs):        
        max_fil_fun_val_s = perc * max_fil_fun_val
        # print("max_fil_fun_val_s = ", np.around(max_fil_fun_val_s, 2))
        cub_complex = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                cub_complex[i, j] = 1 - (fil_fun_vals[i, j] <= perc * max_fil_fun_val_s)
        axes[p].matshow(cub_complex, vmin = 0, vmax = 1, cmap = plt.cm.gray_r)
        axes[p].set_title("K_%.2f" %max_fil_fun_val_s, fontsize = 20)
        
        # title = axes[1+v].title
        # title.set_position([.5, 1.5])
        # axes[1+v].set_xticks([])
        # axes[1+v].set_yticks([])
        # axes[1+v].set_xticklabels([])
        # axes[1+v].set_yticklabels([])
        # plt.show()
    plt.savefig(path, bbox_inches = "tight") 
    plt.show()
        
        
        
def calculate_and_plot_cubical_filtration_and_pds(X, fil_fun, max_fil_fun_vals = [0, 0.25, 0.5, 0.75, 1], min_PD = -0.25, max_PD = 1.25):
    num_filtered_complexes = len(max_fil_fun_vals)
    
    num_fig_rows = 1
    num_fig_cols = 1 + num_filtered_complexes + 1 # 2
    subfig_height = 2
    subfig_width = 2
    fig, axes = plt.subplots(num_fig_rows, num_fig_cols, figsize = (num_fig_cols * subfig_width, num_fig_rows * subfig_height)) 
    fig.tight_layout(pad = 0.5) 


    # Calculate and plot image.
    num_x_pixels = 10
    num_y_pixels = 10
    image = data_construction.point_cloud_to_image(X, num_x_pixels, num_y_pixels)
    axes[0].matshow(image, cmap = plt.cm.gray_r)
    axes[0].set_title("Image", fontsize = 20)


    # Calculate simplex tree, necessary to calculate and plot the filtration and PDs.        
    t0 = time.time()
    simplex_tree, fil_fun_vals = calculate_cubical_simplex_tree(X, fil_fun, num_x_pixels, num_y_pixels)  
    t1 = time.time()
    # print("Runtime for calculation of cubical simplex tree =", np.around(t1-t0, 2))
 
    # Plot filtration.
    for ax, max_fil_fun_val in enumerate(max_fil_fun_vals):  
        cub_complex = np.zeros((num_x_pixels, num_y_pixels))
        for i in range(num_x_pixels):
            for j in range(num_y_pixels):
                cub_complex[i, j] = 1 - (fil_fun_vals[i, j] <= max_fil_fun_val)
        axes[ax+1].matshow(cub_complex, vmin = 0, vmax = 1, cmap = plt.cm.gray_r)
        axes[ax+1].set_title("K_%.2f" %max_fil_fun_val, fontsize = 20)
        # title = axes[1+v].title
        # title.set_position([.5, 1.5])
        # axes[1+v].set_xticks([])
        # axes[1+v].set_yticks([])
        # axes[1+v].set_xticklabels([])
        # axes[1+v].set_yticklabels([])
    
    
    # Calculate and plot PD.
    t0 = time.time()
    simplex_tree.persistence() 
    PD0 = simplex_tree.persistence_intervals_in_dimension(0)
    PD1 = simplex_tree.persistence_intervals_in_dimension(1)
    t1 = time.time()
    # print("Runtime for calculation of PDs =", np.around(t1-t0, 2), "\n")
    # print("0-dim PD = \n", np.around(PD0, 2))
    # print("1-dim PD = \n", np.around(PD1, 2))

    # Plot 0-dim PD.
    PD0[PD0 == np.inf] = max_PD
    axes[num_fig_cols - 1].scatter(PD0[:, 0], PD0[:, 1], 40, c = "green")
    x = np.arange(min_PD, max_PD, 0.01)
    axes[num_fig_cols - 1].plot(x, x, c = 'black') # plot the diagonal  
    intervals_unique, multiplicities = np.unique(PD0, axis = 0, return_counts = True) 
    for i, multiplicity in enumerate(multiplicities):
        if multiplicity > 1:
            axes[num_fig_cols - 1].annotate(multiplicity, xy = (intervals_unique[i, 0], intervals_unique[i, 1]), 
                         ha = "left", va = "bottom", xytext = (5, 0), textcoords = "offset points", 
                         fontsize = 15, color = "black") 
    axes[num_fig_cols - 1].set_xlim(min_PD, max_PD)
    axes[num_fig_cols - 1].set_ylim(min_PD, max_PD)
    axes[num_fig_cols - 1].set_aspect("equal")
    axes[num_fig_cols - 1].set_title("0-dim PD", fontsize = 20, pad = 20)

    # Plot 1-dim PD.
    # PD1[PD1 == np.inf] = max_PD
    # axes[num_fig_cols - 1].scatter(PD1[:, 0], PD1[:, 1], 40, c = "green")
    # x = np.arange(min_PD, max_PD, 0.01)
    # axes[num_fig_cols - 1].plot(x, x, c = 'black') # plot the diagonal  
    # intervals_unique, multiplicities = np.unique(PD1, axis = 0, return_counts = True) 
    # for i, multiplicity in enumerate(multiplicities):
    #     if multiplicity > 1:
    #         axes[num_fig_cols - 1].annotate(multiplicity, xy = (intervals_unique[i, 0], intervals_unique[i, 1]), 
    #                          ha = "left", va = "bottom", xytext = (5, 0), textcoords = "offset points", 
    #                          fontsize = 10, color = "black")   
    # axes[num_fig_cols - 1].set_xlim(min_PD, max_PD)
    # axes[num_fig_cols - 1].set_ylim(min_PD, max_PD)
    # axes[num_fig_cols - 1].set_aspect("equal")
    # axes[num_fig_cols - 1].set_title("1-dim PD", fontsize = 20, pad = 20)
    
    plt.show()
    
    
    return fig




def calculate_pd_max_lifespans_height_filtration(X, num_x_pixels = 20, num_y_pixels = 20):
    
    # plots.plot_point_cloud(X, path = "convexity/point_cloud")
    
    lifespans = []
    
    nx = num_x_pixels
    ny = num_y_pixels    
    
    # Image.
    image = data_construction.point_cloud_to_image(X, num_x_pixels, num_y_pixels)
    # plt.matshow(image, cmap = plt.cm.gray_r)
    # plt.title("Image", fontsize = 20)
    # plt.savefig("convexity/image", bbox_inches = "tight")      
    # plt.show()

      
    # Filtration function.    
    nxm = int(nx/2)
    nym = int(ny/2)
    pixels_orig = np.array([[0, 0], [0, nym], [0, ny-1], [nxm, ny-1], [nx-1, ny-1], [nx-1, nym], [nx-1, 0], [nxm, 0], [nxm, nym]])
    pixels_dir = np.array([[0+1, 0+1], [0+1, nym], [0+1, ny-1-1], [nxm, ny-1-1], [nx-1-1, ny-1-1], [nx-1-1, nym], [nx-1-1, 0+1], [nxm, 0+1], [nxm-1, nym]])
    num_height_dirs = len(pixels_dir)
    for p in range(num_height_dirs):
        pixel_orig = pixels_orig[p]
        pixel_dir = pixels_dir[p]
 
        # Summarize and visualize origin and direction pixel.
        # print("\n\n\n")
        # print("pixel_orig = ", pixel_orig)
        # print("pixel_dir = ", pixel_dir)        
        # image_orig = np.zeros((num_x_pixels, num_y_pixels), dtype = np.int8)   
        # u = pixel_orig[0]
        # v = pixel_orig[1]
        # image_orig[u, v] = 1    
        # plt.matshow(image_orig)
        # plt.title("Origin pixel for height filtration")
        # plt.show()
        # image_dir = np.zeros((num_x_pixels, num_y_pixels), dtype = np.int8)   
        # u = pixel_dir[0]
        # v = pixel_dir[1]
        # image_dir[u, v] = 1    
        # plt.matshow(image_dir)
        # plt.title("Direction pixel for height filtration")
        # plt.show()

        # Calculate filtration function.
        fil_fun_vals = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                point = np.array([i, j])
                vector_1 = pixel_orig - point
                vector_2 = pixel_orig - pixel_dir
                fil_fun_vals[i, j] = np.dot(vector_1, vector_2)              
                # When the origin pixel is in the middle of the image,
                # half of the filtration function values will be negative,
                # so we need to change this.
                fil_fun_vals[i, j] = np.abs(fil_fun_vals[i, j])        
        fil_fun_vals[image == 0] = 1 * np.max(fil_fun_vals)  
        fil_fun_vals = fil_fun_vals / np.max(fil_fun_vals)    
        
        # Summarize and visualize filtration function.
        # print("height fil_fun_vals = \n", np.around(fil_fun_vals, 2)) 
        # plot = plt.matshow(fil_fun_vals, cmap = "hot") # cmap = plt.cm.gray_r)
        # plt.title("filtration\nfunction", fontsize = 20, pad = 10)
        # plt.colorbar(plot) #, orientation = "horizontal")
        # plt.show()   
        
        # plot_cubical_filtration(fil_fun_vals, path = "convexity/filtration_direction_%d" %(p+1))

        fil_fun_vals = fil_fun_vals.reshape((num_x_pixels * num_y_pixels, ))                

        
        # PDs.
        cubical_complex = gd.CubicalComplex(dimensions = [nx, ny], top_dimensional_cells = fil_fun_vals)
        simplex_tree = cubical_complex
        simplex_tree.persistence() 
        PD0 = simplex_tree.persistence_intervals_in_dimension(0)
        # PD1 = simplex_tree.persistence_intervals_in_dimension(1)
        # print("PD0 = ", np.around(PD0, 2))
        
        # plots.plot_pd(PD0, path = "convexity/pd0_direction_%d" %(p+1))        
        
        lifespan = sorted_lifespans_pd(PD0, size = 2)[1]
        # print("For this direction of the heigh filtration, the second largest lifespan in 0-dim PD = ", np.around(lifespan, 2))
        
        lifespans.append(lifespan)
            
    lifespans = np.asarray(lifespans)
    lifespans = np.sort(lifespans)    
    lifespans = np.flip(lifespans)    
    # print("lifespans = ", np.around(lifespans, 2))
    return lifespans



def calculate_ph_height_point_clouds(point_clouds):
    print("Calculating persistence diagrams (PDs) wrt height filtration, and their lifespans, for the given dataset...")
    t0 = time.time()
    ph = [calculate_pd_max_lifespans_height_filtration(pc) for pc in point_clouds] 
    ph = np.array(ph)
        
    # Instead of looking at the lifespans of 2nd most persisting 0-dim cycles (connected components)
    # across all directions of the height filtration, we might only look at the maximum across directions.
    ph = np.amax(ph, axis = 1)
    ph = ph.reshape(-1, 1)       
    
    # # Transform list of PDs with different number of cycles into an array of PDs with the same number of cycles.    
    # PDs_length = [len(PD) for PD in PDs]
    # max_PDs_length  = max(PDs_length)    
    # PDs_array = np.zeros((m, max_PDs_length, 2))
    # for s, PD in enumerate(PDs):
    #     for i in range(max_PDs_length):
    #         if i < len(PD):
    #             PDs_array[s][i] = PDs[s][i]
    #         else:
    #             PDs_array[s][i] = np.asarray([0, 0])    
    
    t1 = time.time()
    print("Runtime = ", np.around(t1-t0, 2),  "seconds.")
    return ph


################################## PH ON CUBICAL COMPLEXES, END #################################################