from turtle import forward
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self,lr,input_dims,output_dims):
        super(DQN,self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.fc1 = nn.Linear(*self.input_dims,32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,output_dims)
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = "mps" if T.backends.mps.is_available() else "cpu"
        self.to(self.device)


    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = F.softmax(self.fc3(x))
        return action



class player():
    def __init__(self,input_dims,output_dims,lr,max_mem_size = 100):
        self.max_mem_size = max_mem_size
        self.state_memory = np.zeros((max_mem_size, *input_dims),dtype=np.float32)
        self.new_state_memory = np.zeros((max_mem_size, *input_dims),dtype=np.float32)
        self.action_memory = np.zeros(max_mem_size,dtype=np.int32)
        self.reward_memory = np.zeros(max_mem_size,dtype=np.float32)
        self.terminal_memory = np.zeros(max_mem_size,dtype=np.bool)
        self.max_mem_size = 100
        self.action = [i for i in range(output_dims)]
        self.cnt = 0
        self.lr = lr
        self.DQN = DQN(lr,input_dims,output_dims)


    def store_transition(self,state,action,reward,new_state,done):
        index = self.cnt % self.max_mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.cnt += 1


    def choseAction(self, observation):
        state = np.array(observation)
        state = T.tensor([state]).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = T.argmax(actions).item()
        return action


        

        

    
        


