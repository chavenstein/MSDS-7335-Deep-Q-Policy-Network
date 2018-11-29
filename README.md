## Ms. Pac-Man Deep Reinforcement Learning Using Deep Q Networks

## Report by: Christopher Havenstein
## For MSDS 7335: Black Box Machine Learning

### Original Code available through "The MIT License." In this license, permission was granted to "use, copy, modify, merge, publish, distribute, sublicense, and/or sell ... "without restriction[.]" To view the full MIT license, which granted me access to use this code freely, please navigate to \AdvancedPacmanDQNs-master\LICENSE.md for more details. I modified the initial code slightly from here and trained my own "agent" (to be later described) with a Deep Q Policy Network.



# Installation

## To get the code to run:

Please follow the steps in \AdvancedPacmanDQNs-master\INSTALLATION.md to begin setup. For convenience, I have already placed keras-rl (the 'rl' folder) in the "AdvancedPacmanDQNs-master\" directory for you. However, I also had to put this "rl" folder in a subdirectory within the directory I installed Anaconda3. For me, the path was "C:\Users\redacted\Anaconda3\envs\tf_gpu\Lib\site-packages\rl" and then the code would run. Hopefully this will help you circumvent the hassle with getting keras-rl to work, since I spent a while before realizing I had to also copy the "rl" folder into my local directory path above.


## To get TensorFlow GPU to work (in Windows):

**There isn't an easy way to say this, but installing TensorFlow GPU takes persistence and grit**. However, [this guide](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc) helped me to get started. These paraphrased steps below are the general process.

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


# Deep Reinforcement Learning Information

## How I got started:

I first started learning about deep reinforcement learning by reading an article [here](http://karpathy.github.io/2016/05/31/rl/). I used the code [here](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5) to train an **"agent"** for about 19 hrs to play Pong. After I learned the basics there, I folowed an example with the code in this repo to learn how to train an **agent** to play Ms. Pac-Man.


## Deep Reinforcement Learning overview:

In the context of playing video games, an **agent** is an AI player that learns from the pixels on the screen about the current **state** of the game. As you might expect, depending on the video game, an **agent** has a set of **actions** available to perform as a player. For example, in Pong, there are only two **actions** available to the **agent** - move up or move down. However, the *agent* must learn the correct *policies* to maximize a particular **reward**. Conceptually, deep reinforcement learning is about an **agent** learning the **policies** that maximize a particular **reward.** Often in video games, the **reward** that the **agent** is reinforced to learn **policies** for involves maximizing the **agent's** score in that video game. 


## How Does Machine Learning come in?

First, video games are great to train **agents** since we can infinitely create images to train the **agent** and the reward - often in games this is the **agent's** score - is already provided. If you are familiar with deep learning from images, often the goal there is to classify these images or localize objects by drawing bounding boxes around objects. The problem is you need labels to train the model to classify images or objects in this situation. If you don't already have a convolutional neural network model with prelearned weights for transfer learning (specifically to predict those labels) you have a lot of work to do. (Including collecting a large training set of labeled images then spending days/weeks/months to train the model.) So, deep reinforcement learning for educational purposes is simple to collect data for since we have Python libraries with game environments already to train agents - like [Gym](https://gym.openai.com/envs/#atari).

Okay, great, but how are these networks actually structured for deep reinforcement learning? Well, we start with using convolultional neural networks to read the pixels from the game screen. Typically, we preprocess the raw images first to reduce their size (in pixels), remove colors (AKA, only black and white remains), and create difference images. Difference images are basically images that combine information from multiple images to determine directions objects in the game are moving in. The neural network convolutions with filters create feature maps that after convolution and pooling steps are eventually mapped to a fully-connected neural network. The fully connected neural network's input layers are the vectors of numbers created through convolution and pooling steps from the raw pixels. Then, a number of hidden layers are added in the fully-connected neural network based on the complexity and number of output neurons in the output layer. The number of output neurons in the output layer is typically equal to the number of **actions** available in the game for the **agent** to perform.

So, we mentioned **policies** earlier, how do those play into the network? In the [initial resource I used](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5), policies are a series of actions that lead to a reward for the **agent.** So, the **agent** will learn policies overtime. In Ms. Pac-Man, that might mean --up -> up -> left -> right-- gave the **agent** points to increase it's score. Or, a series of actions caused no increase to the **agent's** score. Overtime, through gradient descent, **policy gradients** are calculated, and the resulting learned weights will be updated based on their impact to the **reward** for the **agent.** The probility of following those policies that grant the highest rewards will increase over time. Then, the **agent** will tend to take **actions** that follow those policies. For the math behind policy gradients, go again to [this blog](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5). Hopefully, this gives you a baseline intuition about deep reinforcement learning. Though, you can learn it much more detail.


## What are Deep Q Policy Networks?

Deep Q Policy Networks are basically more complex policy networks which use multiple convolutions and  multiple fully connected neural networks. To learn about Deep Q Policy Networks, I went [here](https://towardsdatascience.com/advanced-dqns-playing-pac-man-with-deep-reinforcement-learning-3ffbd99e0814). Deep Q Policy Networks are also known as "value-based learning." To paraphrase, the output values from the fully-connected neural network are the **agent's** guesses about what the reward is esimated to be if a given action is taken. This typically involves a function called the "Q function" which takes the form of Q(s,a) where the input **state** is **s** and the **action** to be taken is **a**. The Q value is calculated by the following: "The total value of taking action a in state s, is the sum of the future rewards r, adjusted by a discount factor, gamma, indicating how far into the future the agent is expected to plan."

![Q-Value](https://cdn-images-1.medium.com/max/800/0*WinVpmWDI7P4xa3w.jpg)

In short, this is a fancy way of saying that the **agent** takes the **action** that leads to the maximum expected **reward** score. However, the **agent** cannot see into the future, so we use deep learning. We map pixel images to Q values, then the neural network approximates the actual but unknown Q function. Through enough training (which is often a lot in deep learning), the **agent** learns to predict the Q function. The training occurs by using gradient descent on the loss function, below.

![LossFunction](https://cdn-images-1.medium.com/max/800/0*LNODZERcBXjMgr2_.jpg)

Think of this loss function as a difference between the actual/target Q values and the **agent's** current best guess of the Q values. The target Q values are the immediate reward added to the Q value of the action taken in the next state. It should be noted that this is a like trying to predict a "moving target" since the target Q values are being calculated by the same network we are training. This actually is a key point about reinforcement learning in general, we have an data set that is constantly changing.


## Different Deep Q Policy Network Agents 
### and How to Use Them 

In this code, there are multiple types of archtectures that are available as **agents** for you to try. These different architectures are: (1) a "vanilla" Deep Q Policy Network; (2) a "pdd" Deep Q Policy Network; and (3) a "noisyNstepPDD" Deep Q Policy Network. You can try them by navigating to "\AdvancedPacmanDQNs-master\agents" in this repo then by running the appropriate file in your Anaconda Prompt (e.g. running "python noisyNstepPDD.py"). I cannot take full credit for writing these, I found these as examples, and through the MIT License I was able to provide them to you and pass the learnings with my experiences to you. 

Each of these agents can be run in a training or testing mode. You can typically find this in a line of the **agent's** code like the following: "parser.add_argument('--mode', choices=['train', 'test'], default='test')", where you would edit the default argument to either train or test. You can also run the program with this argument added (e.g. "python noisyNstepPDD.py --mode test" or python noisyNstepPDD.py --mode train). Please note that the moment you run an **agent** in training mode, you risk erasing the current saved prelearned weights. I recommend that you back up the current weights in the "\AdvancedPacmanDQNs-master\model_saves" folder prior to playing with this code.

I personally spent most of my time playing with the "noisyNstepPDD.py" agent. With this one, I trained my own model to gain the experience. By using TensorFlow-GPU, I trained the network on my laptop (running Windows 10, a i7-77HQ CPU, 32 GB RAM, and a Nvidia GTX 1060 with 6 GB VRAM) for 6 days. If I didn't have TensorFlow-GPU configured, I probably would still be training the model. The output of this training was a "REWARD_DATA" file, and a series of learned model weights (.h5f) files - ending with the learned weights after the agent learned for 30,000,000 of Ms. Pac-Man's steps. These reward and weight files are saved in the "\AdvancedPacmanDQNs-master\model_saves\" folder with a sub-folder for each type of agent.

While I don't have a lot of time to describe each type of policy network for these three **agents** here, I'll describe some about the one I used, the "noisyNstepPDD.py" **agent**. With this type of **agent**, the Q function is separated into a sum of two independent parts. The first  part is the "value-function" for the value of the **agent** being in the current **state**. This formula is shown below. 

![Value Function](https://cdn-images-1.medium.com/max/800/0*03T316qxTB1DhMFL.jpg)

The second part is the advantage function which basically models the importance of each **action**, such that the **agent** takes the **best action** possible, even if it doesn't have an effect on the score right now.

![Avantage Function](https://cdn-images-1.medium.com/max/800/0*kJebWmv28J_9L7ZQ.jpg)

The methods of how to choose the best **action** in the current **state** are typically similar to those used in maze solving algorithms. There is a weighted probability of each available **action** and the goal is often to chose the highest weighted probability of receiving the best **reward**. However, if all of the weighted probabilities are not greater than a threshold value, the **agent** should take an **action** randomly to gain more information. Often in reinforcement learning, a greedy and random approach is used to find the **action** given the current **state** that is expected to give the best **reward**.

There are more details regarding how the "noisyNstepPDD.py" **agent** works, and I would recommend reading through [this blog](https://towardsdatascience.com/advanced-dqns-playing-pac-man-with-deep-reinforcement-learning-3ffbd99e0814) to get more information. In class, I will demo the agent I trained with the "noisyNstepPDD" approach. 

Hopefully, this readme markdown file gives you enough information to get started with deep reinforcement learning with Ms. Pac-Man. I hope that you have fun, and if you have any questions, I'll do my best to help if you email me at chavenstein@smu.edu or chavenst@gmail.com.



