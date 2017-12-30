# Installing Tensorflow, Theano and Keras in Spyder

I recently got some issues with my computer and had to reinstall my system, including Python. 
I installed python through Anaconda. The default version of Python is 3.6.3. Tensorflow 
couldn't be imported through Spyder. If you are using windows system, you may refer to [this
blog to install Tensorflow, Theano and Keras](https://medium.com/@pushkarmandot/installing-tensorflow-theano-and-keras-in-spyder-84de7eb0f0df)

Here are basics commands:
1) Create a new environment "py35":

> conda create -n py35 python=3.5 anaconda

2) Install spyder in the new environment
> activate py35

> conda install spyder

3) Install packages

> conda install theano 

> conda install tensorflow

> conda install keras

4) More libaries: 

> pip install numpy

> pip install scipy

> pip install scikit-learn 

> pip install matplotlib

> pip install pandas

5) Test:
> python # start python 3.5 version

> import theano

> import tensorflow

> import keras


6) More info, you may refer to [TensorFlow install page](https://www.tensorflow.org/install/install_windows)

# Contents
