
# coding: utf-8

# # 图像分类
# 
# 在此项目中，将对 [CIFAR-10 数据集]中的图片进行分类。该数据集包含飞机、猫狗和其他物体。首先对这些图片做了预处理，然后用所有样本训练一个卷积神经网络。图片需要标准化（normalized），标签需要采用 one-hot 编码。在项目中构建卷积的、最大池化（max pooling）、丢弃（dropout）和完全连接（fully connected）的层。最后，在样本图片上看到神经网络的预测结果。
# 
# ## 获取数据

# In[1]:


from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

floyd_cifar10_location = '/input/cifar-10/python.tar.gz'
if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(tar_gz_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            tar_gz_path,
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open(tar_gz_path) as tar:
        tar.extractall()
        tar.close()


# ## 探索数据
# 
# 该数据集分成了几部分／批次（batches），可以避免机器在计算时内存不足。CIFAR-10 数据集包含 5 个部分，名称分别为 `data_batch_1`、`data_batch_2`，以此类推。每个部分都包含以下某个类别的标签和图片：
# 
# * 飞机
# * 汽车
# * 鸟类
# * 猫
# * 鹿
# * 狗
# * 青蛙
# * 马
# * 船只
# * 卡车

# In[2]:


import helper
import numpy as np

batch_id = 1
sample_id = 5
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)


# ## 实现预处理函数
# 
# ### 标准化

# In[3]:


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    return x / 255


# ### One-hot 编码

# In[4]:


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    return np.array([[1 if j==i else 0 for j in range(10)] for i in x])


# ### 随机化数据
# 
# 在之前探索数据时，已经了解到，样本的顺序是随机的。再随机化一次也不会有什么关系，但是对于这个数据集没有必要。
# 

# ## 预处理所有数据并保存
# 
# 运行下方的代码单元，将预处理所有 CIFAR-10 数据，并保存到文件中。下面的代码还使用了 10% 的训练数据，用来验证。
# 

# In[5]:


# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


# # 检查点
# 
# 在这里设置了第一个检查点。如果你什么时候决定再回到该记事本，或需要重新启动该记事本，你可以从这里开始。预处理的数据已保存到本地。
# 

# In[6]:


import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


# ## 构建网络
# 
# ### 输入

# In[7]:


import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]],name="x")


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, [None, n_classes],name="y")


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32,name="keep_prob")

tf.reset_default_graph()


# ### 卷积和最大池化层

# In[8]:


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    filter_weights = tf.Variable(
        tf.truncated_normal([conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_num_outputs]
#                         ,stddev = 1.0/ conv_ksize[0] / conv_ksize[1] / x_tensor.get_shape().as_list()[3]  )
                             )/np.sqrt(conv_ksize[0] * conv_ksize[1] * x_tensor.get_shape().as_list()[3] )
    )  # (height, width, input_depth, output_depth)
    filter_bias = tf.Variable(tf.zeros(conv_num_outputs))
    strides = conv_strides  # (batch, height, width, depth)
    strides = (1, conv_strides[0], conv_strides[1], 1)
    padding = 'SAME'
    conv = tf.nn.conv2d(x_tensor, filter_weights, strides, padding) + filter_bias
    conv = tf.nn.relu(conv)

    filter_shape = [1, pool_ksize[0], pool_ksize[1], 1]
    strides = [1, pool_strides[0], pool_strides[1], 1]
    padding = 'SAME'
    pool = tf.nn.max_pool(conv, filter_shape, strides, padding)

    return pool 


# ### 扁平化层

# In[9]:


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    x_shape = x_tensor.get_shape().as_list()
    return tf.reshape(x_tensor, [-1, x_shape[1]*x_shape[2]*x_shape[3]])



# ### 完全连接的层

# In[10]:


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
#     w = tf.Variable(tf.random_normal([x_tensor.get_shape().as_list()[1], num_outputs],stddev=1.0/x_tensor.get_shape().as_list()[1]))
    w = tf.Variable(tf.truncated_normal([x_tensor.get_shape().as_list()[1], num_outputs])/np.sqrt(x_tensor.get_shape().as_list()[1]))

    b = tf.Variable(tf.truncated_normal([num_outputs]))
    dense1 = tf.nn.relu(tf.add(tf.matmul(x_tensor, w), b))
    return dense1


# ### 输出层

# In[11]:


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
#     w = tf.Variable(tf.random_normal([x_tensor.get_shape().as_list()[1], num_outputs],stddev = 1.0/x_tensor.get_shape().as_list()[1]))
    w = tf.Variable(tf.random_normal([x_tensor.get_shape().as_list()[1], num_outputs])/np.sqrt(x_tensor.get_shape().as_list()[1]))

    b = tf.Variable(tf.random_normal([num_outputs]))
    out = tf.matmul(x_tensor, w) + b
    return out


# ### 创建卷积模型

# In[12]:


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # Apply3 Convolution and Max Pool layers
    conv1 = conv2d_maxpool(x, 16, [3, 3], [1,1], [2,2], [2,2])
    conv2 = conv2d_maxpool(conv1, 32, [3, 3], [1, 1], [2, 2], [2, 2])
    conv3 = conv2d_maxpool(conv2, 64, [2, 2], [1, 1], [2, 2], [2, 2])

    # Apply a Flatten Layer
    dense1 = flatten(conv3)

    #Apply3 Fully Connected Layers
    dense1 = fully_conn(dense1, 512)
    dense1 = tf.nn.dropout(dense1,keep_prob)
    dense2 = fully_conn(dense1, 256)
    dense2 = tf.nn.dropout(dense2,keep_prob)
    dense3 = fully_conn(dense2, 128)
    dense3 = tf.nn.dropout(dense2,keep_prob)
    
    # Apply an Output Layer
    out = output(dense2,10)
    
    return out

# Build the Neural Network 

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# ## 训练神经网络
# 
# ### 单次优化

# In[13]:


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer, feed_dict={x:feature_batch,y:label_batch,keep_prob:keep_probability})
  


# ### 显示数据

# In[14]:


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    loss = sess.run(cost ,feed_dict = {x:feature_batch,y:label_batch,keep_prob:1.})
    acc = sess.run(accuracy, feed_dict={x:valid_features,y:valid_labels,keep_prob:1.})
    print("Loss = " + "{:>5.4f}".format(loss) + ", Validation Accuracy = " + "{:.6f}".format(acc))


# ### 超参数

# In[15]:


epochs = 35
batch_size = 128
keep_probability = 0.6


# ### 在单个 CIFAR-10 部分上训练

# In[16]:


print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)


# ### 完全训练模型
# 
# 现在，单个 CIFAR-10 部分的准确率已经不错了，再试试所有五个部分吧。

# In[17]:


save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)


# # 检查点
# 
# 模型已保存到本地。
# 
# ## 测试模型
# 
# 利用测试数据集测试你的模型。这将是最终的准确率。你的准确率应该高于 50%。如果没达到，请继续调整模型结构和参数。

# In[18]:


import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for test_feature_batch, test_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()


# ## 准确率怎么样
# 
# 在这里准确率只有70%左右，为何准确率不能更高了？首先，对于简单的 CNN 网络来说，７0% 已经不低了。纯粹猜测的准确率为10%。但是，你可能注意到有人的准确率远远超过80%，(http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130)。这是因为我们我们还需要掌握一些其他的技巧。
# 
