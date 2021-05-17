# Project 1: Navigation

### Project description
The folder contains the codes and a report for project 1 of the Deep Reinforcement Learning nanodegree. The project is based on a Unity enviroment, where an agent is travelling in a workspace to collect bananas. Every time the agent collects a yellow banana, it gets a reward of +1 and every time it collects a blue banana, it receives a reward of -1. Therefore, the goal for this project is to train the agent with a Deep Q-Network (DQN) so that it collects as many yellow bananas as possible while avoiding blue bananas. The environment is considered solved if the agent get an avarage score of +13 over 100 consecutive episodes. <br/>

<img src="banana.gif" />

The size of the state space is 37, which contains the agent's velocity, as well as ray-based perception of objects. The size of the action space is 4, indicating 4 directions the agent can move (forward, backward, left, right). <br/>

### Getting Started
Below are the dependencies you need to install so as to run the codes smoothly. 
tensorflow==1.7.1 <br/>
Pillow>=4.2.1 <br/>
matplotlib <br/>
numpy>=1.11.0 <br/>
jupyter <br/>
pytest>=3.2.2 <br/>
docopt <br/>
pyyaml <br/>
protobuf==3.5.2 <br/>
grpcio==1.11.0 <br/>
torch==0.4.0 <br/>
pandas <br/>
scipy <br/>
ipykernel <br/>

You will also need to download the environment based on your operating system. <br/>
Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip) <br/>
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip) <br/>
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip) <br/>
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip) <br/>

### Instruction
Once you download the environment, place the file in the 'p1_navigation/' folder and unzip (or decompress) the file. <br/>
After you have followed the instructions above, open 'Navigation.ipynb' to train the agent and the network weights are saved in the 'checkpoint' folder. You can use them to evaluate the agent's performance.
