#Ms. PacMan Deep Reinforcement Learning
## Using Deep Q Networks

## Report by: Christopher Havenstein
## For MSDS 7335: Black Box Machine Learning

### Original Code available through "The MIT License." In this license, permission was granted to "use, copy, modify, merge, publish, distribute, sublicense, and/or sell ... "without restriction[.]" To view the full MIT license, which granted me access to use this code freely, please navigate to \AdvancedPacmanDQNs-master\LICENSE.md for more details. I modified the initial code slightly from here and trained my own "agent" (to be later described) with a Deep Q Policy Network.



# Installation

## To get the code to run:

Please follow the steps in \AdvancedPacmanDQNs-master\INSTALLATION.md to begin setup. For convenience, I have already placed keras-rl (the 'rl' folder) in the "AdvancedPacmanDQNs-master\" directory for you. However, I also had to put this "rl" folder in a subdirectory within the directory I installed Anaconda3. For me, the path was "C:\Users\redacted\Anaconda3\envs\tf_gpu\Lib\site-packages\rl" and then the code would run. Hopefully this

## To get TensorFlow GPU to work (in Windows):

**There isn't an easy way to say this, but installing TensorFlow GPU takes persistence and grit**. However, [the guide at](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc) helped me to get started. These paraphrased steps below are the general process.

1. Check to see if your GPU works with TensorFlow GPU [here](https://developer.nvidia.com/cuda-gpus). Yes, you need a compatible discrete graphics card for Tensorflow GPU to work.
2. Acquire the Cuda Toolkit [here](https://developer.nvidia.com/cuda-downloads).
3. Acquire cuDNN after creatung an account on Nvidia's Developer Site [here](https://developer.nvidia.com/cudnn).
4. Correctly extract cuDNN to the correct directories where you installed the Cuda Toolkit as defined [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html) and below.
	1. "Copy <installpath>\cuda\bin\cudnn64_7.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin."
	2. "Copy <installpath>\cuda\ include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include."
	3. "Copy <installpath>\cuda\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64."
5. Ensure that "CUDA_HOME" is in your environment path variables. You may need to Google that if you don't know where to check.
6. If you already have Anaconda installed for Python 3.5 or 3.6, you can skip this step. This step involves installing Anaconda.
7. In the Anaconda Prompt, which should have been installed when you installed Anaconda, install the package libraries if you haven't already that are listed in "\AdvancedPacmanDQNs-master\INSTALLATION.md" within this repository.
8. In the Anaconda Prompt, run "pip install tensorflow-gpu" to install TensorFlow GPU.
9. Then you can try to run the code included in this repository for the agent I trained in this repository at "AdvancedPacmanDQNs-master\agents\noisyNstepPDD.py" and read the starting messages right after TensorFlow starts running.

# Description
## coming soon...