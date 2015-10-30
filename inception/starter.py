# from IPython import display
from matplotlib import pyplot
import numpy as np
import os
import sys
# %matplotlib inline
import pdb

# Make sure that you set this to the location your caffe2 library lies.
caffe2_root = '/home/shruti/caffe2'
sys.path.insert(0, os.path.join(caffe2_root, 'gen'))

# After setting the caffe2 root path, we will import all the caffe2 libraries needed.
from caffe2.proto import caffe2_pb2
from pycaffe2 import visualize, core, workspace, net_drawer

# net is the network definition.
net = caffe2_pb2.NetDef()
net.ParseFromString(open('inception_net.pb').read())
# tensors contain the parameter tensors.
tensors = caffe2_pb2.TensorProtos()
tensors.ParseFromString(open('inception_tensors.pb').read())

# Note that the following line hides the intermediate blobs and only shows the operators.
# If you want to show all the blobs as well, use the commented GetPydotGraph line.
graph = net_drawer.GetPydotGraphMinimal(net.op, name="inception", rankdir='TB')
# graph = net_drawer.GetPydotGraph(net.op, name="inception", rankdir='TB')

#display.Image(graph.create_png(), width=200)

DEVICE_OPTION = caffe2_pb2.DeviceOption()
# Let's use CPU in our example.
DEVICE_OPTION.device_type = caffe2_pb2.CPU

# If you have a GPU and want to run things there, uncomment the below two lines.
# If you have multiple GPUs, you also might want to specify a gpu id.
# DEVICE_OPTION.device_type = caffe2_pb2.CUDA
# DEVICE_OPTION.cuda_gpu_id = 0

# Caffe2 has a concept of "workspace", which is similar to that of Matlab. Each workspace
# is a self-contained set of tensors and networks. In this case, we will just use the default
# workspace so we won't dive too deep into it.
workspace.SwitchWorkspace('default')

# First, we feed all the parameters to the workspace.
for param in tensors.protos:
    workspace.FeedBlob(param.name, param, DEVICE_OPTION)
# The network expects an input blob called "input", which we create here.
# The content of the input blob is going to be fed when we actually do
# classification.
workspace.CreateBlob("input")
# Specify the device option of the network, and then create it.
net.device_option.CopyFrom(DEVICE_OPTION)
workspace.CreateNet(net)


def ClassifyImageWithInception(image_file, show_image=False, output_name="softmax2"):
    from skimage import io, transform
    img = io.imread(image_file)
    # Crop the center
    shorter_edge = min(img.shape[:2])
    crop_height = (img.shape[0] - shorter_edge) / 2
    crop_width = (img.shape[1] - shorter_edge) / 2
    cropped_img = img[crop_height:crop_height + shorter_edge, crop_width:crop_width + shorter_edge]
    # Resize the image to 224 * 224
    resized_img = transform.resize(cropped_img, (224, 224))
    if show_image:
        pyplot.imshow(resized_img)
    # normalize the image and feed it into the network. The network expects
    # a four-dimensional tensor, since it can process images in batches. In our
    # case, we will basically make the image as a batch of size one.
    normalized_img = resized_img.reshape((1, 224, 224, 3)).astype(np.float32) * 256 - 117
    workspace.FeedBlob("input", normalized_img, DEVICE_OPTION)
    workspace.RunNet("inception")
    return workspace.FetchBlob(output_name)


# We will also load the synsets file where we can look up the actual words for each of our prediction.
synsets = [l.strip() for l in open('synsets.txt').readlines()]

pdb.set_trace()
predictions = ClassifyImageWithInception("la.jpg").flatten()
idx = np.argmax(predictions)
print 'Prediction: %d, synset %s' % (idx, synsets[idx])

indices = np.argsort(predictions)
print 'Top five predictions:'
for idx in indices[:-6:-1]:
    print '%6d (prob %.4f) synset %s' % (idx, predictions[idx], synsets[idx])
#pyplot.plot(predictions)

#filters = workspace.FetchBlob('conv2d0_w')
# We normalize the filters for visualization.
#filters = (filters - filters.min()) / (filters.max() - filters.min())
#_ = visualize.PatchVisualizer(gap=2).ShowMultiple(filters, bg_func=np.max)

#conv2d0 = workspace.FetchBlob('conv2d0')
#print 'First layer output shape:', conv2d0.shape
#_ = visualize.PatchVisualizer(gap=10).ShowChannels(conv2d0[0, :, :, :16], bg_func=np.max, cmap=pyplot.cm.hot)

#mixed5b = workspace.FetchBlob('mixed5b')
#print 'First layer output shape:', mixed5b.shape
#visualize.ShowChannels(mixed5b[0, :, :, :144], bg_func=np.max, cmap=pyplot.cm.hot)

