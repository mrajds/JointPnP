from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_voxel(voxel, as_cubes = True, dense_voxel_grid = True, img_path = None):
    fig = pyplot.figure()
    if as_cubes:
        if not dense_voxel_grid:
            dense_voxel_grid = convert_to_dense_voxel_grid(voxel, 32)
        else:
            dense_voxel_grid = voxel
        ax = fig.gca(projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.voxels(dense_voxel_grid, edgecolor='k')
        pyplot.show()
        return
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    X = np.array(voxel[:,0])
    Y = np.array(voxel[:,1])
    Z = np.array(voxel[:,2])
    #ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('voxel')

    scat = ax.scatter(X, Y, Z)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    
    pyplot.grid()
    pyplot.show()
    if img_path is not None:
        pyplot.savefig(img_path)
    
def convert_to_sparse_voxel_grid(voxel_grid):
    sparse_voxel_grid = []
    voxel_dim = voxel_grid.shape
    for i in xrange(voxel_dim[0]):
        for j in xrange(voxel_dim[1]):
            for k in xrange(voxel_dim[2]):
                if voxel_grid[i, j, k] == 1.:
                    sparse_voxel_grid.append([i, j, k])
    return np.asarray(sparse_voxel_grid)

def convert_to_dense_voxel_grid(sparse_voxel_grid, one_axis_length):
    voxel_grid = np.zeros((one_axis_length, one_axis_length, one_axis_length))
    nr_voxels = sparse_voxel_grid.shape[0]
    for i in range(nr_voxels):
        position = sparse_voxel_grid[i,:]
        voxel_grid[position[0], position[1], position[2]] = 1.
    return voxel_grid

def main():
    voxel_grid = np.random.rand(20, 20, 20)
    voxel_grid[voxel_grid >= 0.99] = 1.
    voxel_grid[voxel_grid <= 0.99] = 0.
    sparse_voxel_grid = convert_to_sparse_voxel_grid(voxel_grid)
    dense_voxel_grid = convert_to_dense_voxel_grid(sparse_voxel_grid, 20)
    visualize_voxel(dense_voxel_grid)
    np.testing.assert_array_equal(voxel_grid, dense_voxel_grid)

if __name__ == '__main__':
    main()
