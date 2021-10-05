from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np


def abline(slope, intercept, style = 'k--',ax = None):
    """Plot a line from slope and intercept
    Taken from https://stackoverflow.com/a/43811762/636276
    """
    if ax is None:
        ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, style)


def get_plot_lims(X, padding = 0.12):
    # X is a (n,2) numpy matrix where n is the number of samples
    # returns the list [xmin, xmax, ymin, ymax]
    
    mins = np.min(X,0)
    maxs = np.max(X,0)

    xmin = mins[0]; xmax = maxs[0]; ymin = mins[1]; ymax = maxs[1]
    width = xmax - xmin; height = ymax - ymin

    xmax = xmax + padding*width; xmin = xmin - padding*width
    ymax = ymax + padding*height; ymin = ymin - padding*height
    return [xmin, xmax, ymin, ymax]


def get_grid(lims, n_grid=30):
    # creates a square grid of size n_grid by n_grid 
    # spanning x-axis [lims[0], lims[1]] and y-axis [lims[2], lims[3]]
    # returns the grid points as a (n_grid^2, 2) numpy matrix
    
    x_grid_points = np.linspace(lims[0],lims[1],n_grid)
    y_grid_points = np.linspace(lims[2],lims[3],n_grid)
    x1_test, x2_test =  np.meshgrid(x_grid_points,y_grid_points)

    x1 = x1_test.flatten()
    x2 = x2_test.flatten()

    x_grid = np.array([x1,x2]).T
    return x_grid

def plot_decision_regions(y_grid,lims,ax = None):
    # input:
    #     y_grid = a numpy vector of length n_grid^2 taking values in {0,1,...,n_classes-1}
    #     lims = [xmin, xmax, ymin, ymax], e.g., the output of 'get_plot_lims'
    
    # plot the decision regions of y_grid, 
    # it is expected that y_grid is a list of length n_grid^2
    
    if ax is None:
        ax = plt.gca()
    n_grid = int(np.sqrt(y_grid.shape[0]))
    Z = (y_grid.reshape((n_grid,n_grid)))
    ax.contourf(Z,
                extent = lims,
                levels=1,
                cmap='bone',
                vmin=0.2,
                vmax=0.8,
                alpha=0.2)


#     ax.imshow(Z,extent=lims, alpha=0.2, aspect='auto')
    
def show_result(x, c_show, lims,alpha=1,lw=0):
    plt.scatter(x[:,0], x[:,1], c=c_show, alpha=alpha,lw=lw)
    plt.xlim(lims[0], lims[1]);plt.ylim(lims[2], lims[3])
    return None


def acc_vs_acc_plot(a,b,label_a,label_b):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(a, b, c='k')
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    

def plot_hyperplanes(weights,biases,ax = None):
    if ax is None:
        ax = plt.gca()
        
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    for i in range(len(biases)):
        abline(-weights[0,i]/weights[1,i], -biases[i]/weights[1,i],"k:",ax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
