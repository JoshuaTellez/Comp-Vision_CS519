# Modules used
import sys
import shutil
import numpy as np
from subprocess import call
from PIL import Image

# From library by authors
from lib.solver import Solver
from lib.voxel import voxel2obj

# my reimplimentation
from my_3DR2N2.my_res_gru_net import My_ResidualGRUNet

# Load image(s)
def load_demo_images(num_imgs,img_file):

    # Resize the image(s) to be compatible
    size = 127, 127
    ims = []

    # Load all images
    for i in range(num_imgs):
        # Make images compatible
        im = Image.open(img_file + '/%d.png' % i).convert('RGB')

        # Resize image
        im.thumbnail(size)
        ims.append([np.array(im).transpose(
            (2, 0, 1)).astype(np.float32) / 255.])

    return np.array(ims)


def main():
    # Save prediction into a file named 'prediction.obj' or the given argument
    pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'

    num_imgs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    img_file = sys.argv[3]

    # load images
    demo_imgs = load_demo_images(num_imgs,img_file)

    # Define a network and a solver. Solver provides a wrapper for the test function.
    my_net = My_ResidualGRUNet(batch=1) # instantiate a network
    my_net.load('my_3DR2N2/weights.npy')                        # load downloaded weights
    solver = Solver(my_net)                # instantiate a solver

    # Run the network
    voxel_prediction, _ = solver.test_output(demo_imgs)

    # Save the prediction to a mesh file).
    voxel2obj(pred_file_name, voxel_prediction[0, :, 1, :, :] > 0.4)

    if shutil.which('meshlab') is not None:
        call(['meshlab', pred_file_name])
    else:
        print('Meshlab not found: please use visualization of your choice to view %s' %
              pred_file_name)

main()
