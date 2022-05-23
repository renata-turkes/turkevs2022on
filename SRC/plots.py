import numpy as np
from descartes import PolygonPatch
from shapely.geometry import Point 
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import alphashape
import gudhi as gd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt





def plot_point_cloud(X, xymax = 0, title = " ", path = " "):
    dim = X.shape[1]
    if dim == 2:
        # Holes:
        # plt.scatter(X[:,0], X[:,1], s = 15, color = "red")  
        # Curvature and convexity:
        plt.scatter(X[:,0], X[:,1], s = 15, color = "red")  
        axes = plt.gca()
        # Holes and convexity:
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        # Curvature:
        # axes.set_xlim(-1, 1)
        # axes.set_ylim(-1, 1)
        # Convexity:
        # axes.set_xlim(-0.25, 1.25)
        # axes.set_ylim(-0.25, 1.25)
        axes.set_aspect("equal")
    elif dim == 3:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection = "3d")
        # Holes.
        # axes.scatter(X[:, 0], X[:, 1], X[:, 2], s = 15, color = "red")  
        # Curvature and convexity:
        axes.scatter(X[:, 0], X[:, 1], X[:, 2], s = 5, color = "red")  
        # axes.set_xlim(-0.25, 1.25)
        # axes.set_ylim(-0.25, 1.25)
        # axes.set_zlim(-0.25, 1.25)
        # Holes:
        # axes.set_xlim(0, 1)
        # axes.set_ylim(0, 1)
        # axes.set_zlim(0, 1)
        # Curvature:
        axes.set_xlim(-1, 1)
        axes.set_ylim(-1, 1)
        axes.set_zlim(0, 3)
    else:
        print("Error: Function plot_point_cloud() only plots point clouds of dim = 2 or dim = 3.")    

    if xymax != 0:
        plt.xlim(-xymax, xymax)
        plt.ylim(-xymax, xymax)
        # plt.xlim(0, xymax)
        # plt.ylim(0, xymax)
    plt.title(title, fontsize = 10)
    
    # if path != " ":
    #     plt.savefig(path, bbox_inches = "tight")        
    # plt.show()
    
    fig = plt.gcf()
    if path != " ":        
        fig.savefig(path, bbox_inches = "tight") 
    # plt.show()
    # plt.clf()
    return fig
        
        
        
def plot_rips_simplicial_complex(X, max_distance, xymax = 0, title = " ", path = " "):
    
    dim = X.shape[1]
    
    if dim == 2:
        num_points = X.shape[0]
        distances = euclidean_distances(X)   
        plt.scatter(X[:, 0], X[:, 1], c = "black", s = 3)          
        for i in range(num_points): 
            for j in range(i):           
                if distances[i, j] < max_distance:
                    edge_vertex_1 = X[i]
                    edge_vertex_2 = X[j]
                    x_values = [edge_vertex_1[0], edge_vertex_2[0]]
                    y_values = [edge_vertex_1[1], edge_vertex_2[1]]                    
                    plt.plot(x_values, y_values)    
                    
    if dim == 3:
        rips_complex = gd.RipsComplex(X, max_edge_length = max_distance)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension = 2)
        triangles = np.array([s[0] for s in st.get_skeleton(2) if len(s[0]) == 3])
        fig = plt.figure()
        ax = fig.gca(projection = "3d")
        ax.plot_trisurf(X[:,0], points[:,1], points[:,2], triangles = triangles)
        
    if xymax != 0:
        plt.xlim(-xymax, xymax)
        plt.ylim(-xymax, xymax)   
    plt.title(title)
    if path != " ":
        plt.savefig(path, bbox_inches = "tight")
    plt.show()

    
    
def plot_alpha_simplicial_complex(X, alpha = 0.005, xyzmax = 0, title = " "):
    # https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-alpha-complex-visualization.ipynb
    alpha_complex = gd.AlphaComplex(points = X)
    simplex_tree = alpha_complex.create_simplex_tree()

    points = np.array([alpha_complex.get_point(i) for i in range(simplex_tree.num_vertices())])
    triangles = np.array([s[0] for s in simplex_tree.get_skeleton(2) if len(s[0]) == 3 and s[1] <= alpha])

    fig = plt.figure()
    ax = fig.gca(projection = "3d")
    l = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles = triangles)
    if xyzmax != 0:
        ax.set_xlim(-xyzmax, xyzmax)
        ax.set_ylim(-xyzmax, xyzmax)
        ax.set_zlim(-xyzmax, xyzmax)
    plt.title(title)
    plt.show()


    
def plot_pd(PD, xymax = 0, title = " ", path = " "):    
    '''
    Plot the given persistence diagram (PD).
    
    Input:
        PD: a num_cycles x 2 numpy array of birth and death values for each cycle, i.e., a persistence diagram
        xymax: a positive float, maximum value of the plot x and y axis.  
    ''' 
    
    PD_copy = np.copy(PD)
    PD_copy[PD_copy == np.inf] = 0.999
    
    plt.scatter(PD_copy[:, 0], PD_copy[:, 1], 40, c = "green")
    
    if xymax == 0 and len(PD_copy) != 0:
        xymax = np.max(PD_copy)    
  
    plt.xlim(-0.01, 1.1 * xymax)
    plt.ylim(-0.01, 1.1 * xymax)
    x = np.arange(0, 1.1 * xymax, 0.01)
    plt.plot(x, x, c = 'black') # plot the diagonal  
    PD_approx = np.around(PD_copy, 2)
    intervals_unique, multiplicities = np.unique(PD_approx, axis = 0, return_counts = True) 
    # for i, multiplicity in enumerate(multiplicities):
    #     if multiplicity > 1:
    #         plt.annotate(multiplicity, xy = (intervals_unique[i, 0], intervals_unique[i, 1]), 
    #                      ha = "left", va = "bottom", xytext = (5, 0), textcoords = "offset points", 
    #                      fontsize = 15, color = "black")   
    plt.title(title, fontsize = 50)
    # plt.xticks([])
    # plt.yticks([])
    # plt.xticklabels([])
    # plt.yticklabels([])
    axes = plt.gca()
    axes.set_aspect("equal")
    if path != " ":
        plt.savefig(path, bbox_inches = "tight")        
    plt.show()
    
    
    
def plot_barcode(PD, dmax, lmin = 0.1, title = " "):
        
    PD = [t for t in PD if t[1]-t[0] > lmin]      # select large enough bars
    PD = [[t[0], min(t[1], dmax)] for t in PD]   # threshold the bars exceeding dmax

    plt.figure(figsize = (10,2))
    bar_width = 0.75    
    for j in range(len(PD)):
        t = PD[j]
        plt.fill([t[0], t[1], t[1], t[0]], [j, j, j+bar_width, j+bar_width], fill = True, c = "red", lw = 1)  
    plt.title(title)
    plt.show()
    

    
# Learning curves.    
def plot_curves(x_vals, y1_vals, y2_vals, y1_std = 0, y2_std = 0, y1_label = " ", y2_label = " ", xlabel = " ", ylabel = " ", title = " ", output_path = " "):
    plt.plot(x_vals, y1_vals, color = "blue", marker = "o", markersize = 5, label = y1_label)
    plt.fill_between(x_vals, y1_vals + y1_std, y1_vals - y1_std, alpha = 0.15, color = "blue")
    plt.plot(x_vals, y2_vals, color = "green", marker = '+', markersize = 5, linestyle = '--', label = y2_label)
    plt.fill_between(x_vals, y2_vals + y2_std, y2_vals - y2_std, alpha = 0.15, color = "green")  
    plt.title(title)
    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    plt.ylim(0, 1.1)
    plt.grid()
    plt.legend(loc = "lower right")
    if output_path != " ":
        fig = plt.gcf()
        fig.savefig(output_path, bbox_inches = "tight") 
    plt.show()
    plt.clf()
    
    
    
# type(hull) =  <class 'scipy.spatial.qhull.ConvexHull'>
def plot_convex_hull(hull, points):
    for simplex in hull.simplices:
        plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], '-k', color = "red")
    plt.scatter(points[:, 0], points[:, 1], color = "green") # sampling points
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
    
    
    
# type(hull) =  <class 'shapely.geometry.polygon.Polygon'>
def plot_polygon(polygon, points):
    polygon_vertices = polygon.exterior.coords.xy    
    fig, ax = plt.subplots()    
    ax.scatter(polygon_vertices[0], polygon_vertices[1], color = "red")
    ax.add_patch( PolygonPatch(polygon, fill = False, color = "red") )
    ax.scatter(points[:, 0], points[:, 1], color = "green") # sampling points
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()
    
    
    
def plot_bar_chart(x_ticks_labels, y_values_per_bar_container, legend_labels):  
    '''
    Plot the grouped bar chart with given values.
    
    Input:
    > x_tick_labels: a list of string labels on the x-axis.
    > y_values_per_bar_container: a dictionary, with each element being a list of y-axis values for a specific 
    group/color/legend item.
    > legend labels: a list of string legend labels.
    ''' 
    num_x_ticks = len(x_ticks_labels) 
    x_ticks = np.arange(num_x_ticks)      
    num_bars_per_x_tick = len(y_values_per_bar_container) # = len(legend_labels)   
    width_bar = 0.65 * 1/num_bars_per_x_tick
    width_from_x_tick = np.arange(num_bars_per_x_tick) - np.floor(num_bars_per_x_tick/2)  # (..., -3, -2, -1, 0, 1, 2, 3, ...)
    fig, axes = plt.subplots(figsize = (20, 7)) 
    cmap = plt.get_cmap('tab10') # we have two levels of each type of noise
    bar_colors = [cmap.colors[t] for t in range(num_bars_per_x_tick)]     
    for t, legend_label in enumerate(legend_labels):    
        axes.bar(x_ticks + width_from_x_tick[t] * width_bar, y_values_per_bar_container[legend_label], width_bar, 
                 label = legend_label, color = bar_colors[t])        
    axes.set_xticks(x_ticks)
    axes.set_xticklabels(x_ticks_labels, fontsize = 20)
    axes.set_axisbelow(True)
    axes.set_ylim(0, 1)
    axes.yaxis.grid()
    axes.set_ylabel("accuracy", fontsize = 20)  
    
    legend = axes.legend(legend_labels, ncol = 1, fontsize = 20, bbox_to_anchor=(1.2, 1)) 
    for t, line in enumerate(legend.get_lines()):
        line.set_linewidth(4.0)    
    # plt.show()
    
    return fig, axes