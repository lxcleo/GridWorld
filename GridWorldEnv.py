
from enum import EnumMeta
from inspect import getargvalues
import numpy as np
from random import choice
import copy

class agent():
    def __init__(self,x,y):
        self.row = x
        self.col = y


    def step(self,value):
        if value == 1:
            self.row += 1
        elif value == 2:
            self.row -= 1
        elif value == 3:
            self.col -= 1
        elif value == 4:
            self.col += 1

class env():
    # Action = [0,1,2,3,4] corresponding to [up(0), down(1), left(2), right(3) don't move(4)]
    def __init__(self,map,Pe,action,gama = 1.0):  
        self.w = len(map[0])
        self.h = len(map)
        self.map = map
        self.Pe = Pe
        self.Pa = 1 - Pe
        self.action = action
        self.gama = gama
        # self.policy = np.random.randint(0,len(action),size=(self.w,self.h))
        self.reward = getGridWorld()

    def isBarrier(self,agent): # Check if given index is a barrier
        Bar = self.map[agent.row][agent.col] is None
        # Negative_Reward = agent.col == 4
        # Rs = agent.row == 2 and agent.col == 2
        # Rd = agent.row == 4 and agent.col == 2

        if Bar:
            return True

        else: return False



    def getSurrounding(self,agent):
        # Return vector of 4 number consist of 0 and 1
        # 1 indicates obstacle or boundary and 0 indicates pass
        # In an order of up down left right
        row = agent.row # Y corrdinate corresponding to row index
        col = agent.col # X coordiante corresponding to colum index
        up = not(row == 0 or self.map[row - 1][col] is None)
        down = not(row == self.h - 1 or self.map[row + 1][col] is None)
        left = not(col == 0 or self.map[row][col - 1] is None)
        right = not(col == self.w - 1 or self.map[row][col + 1] is None)


        return[up,down,left,right,True]


    def getUpValue(self,agent,map):
        return map[agent.row - 1][agent.col]

    def getDownValue(self,agent,map):
        return map[agent.row + 1][agent.col]


    def getLeftValue(self,agent,map):
        return map[agent.row][agent.col - 1]

    def getRightValue(self,agent,map):
        return map[agent.row][agent.col + 1]


    def getSelfValue(self,agent,map):
        return map[agent.row][agent.col]

    def getAllValue(self,agent,sur,map): # Get all value according to getSurronding return value
        cnt = []
        value = []
        if sur[0] is True:
            cnt.append(0)
            value.append(self.getUpValue(agent,map))
            

        if sur[1] is True:
            cnt.append(1)
            value.append(self.getDownValue(agent,map))

        if sur[2] is True:
            cnt.append(2)
            value.append(self.getLeftValue(agent,map))

        if sur[3] is True:
            cnt.append(3)
            value.append(self.getRightValue(agent,map))

        cnt.append(4)
        value.append(self.getSelfValue(agent,map))

        return [cnt,value]

    def getValue(self,agent,action):
        if action == 0:
            return self.getUpValue(agent,self.map)

        if action == 1:
            return self.getDownValue(agent,self.map)

        if action == 2:
            return self.getLeftValue(agent,self.map)

        if action == 3:
            return self.getRightValue(agent,self.map)



class VI(env):
    def __init__(self, map, Pe, action, gama=1):
        super().__init__(map, Pe, action, gama)
        
    def calculateValue(self,agent,map,mode = 1):
        value_sum = 0
        sur = self.getSurrounding(agent)
        cnt, value = self.getAllValue(agent,sur,map)
        if mode == 1:
            index = value.index(max(value))
            max_value = value[index]

        else:
            max_action = self.getPolicyAction(agent)
            if not max_action in cnt:
                max_action = choice(cnt)
                self.policy[agent.row][agent.col] = max_action
            index = cnt.index(max_action)
            max_value = value[index]

        # value_sum = self.Pa * (self.getSelfValue(agent) + self.gama * max_value) # Adding self value to value iteration
        P_error = self.Pe / 4
        false_cnt = 0
        for _ in sur:
            if _ is False:
                false_cnt += 1 # Cnt how many barriers are around the agent

        if(index == len(value) - 1): # If staying is the best action
            P_not_move = (false_cnt) * P_error # Probability for not moving
            value_sum = (self.Pa + P_not_move) * (self.getSelfValue(agent,self.reward) + self.gama * max_value)
            value.remove(max_value) # Remove value on the correct direction
            for v in value:
                value_sum += P_error * (self.getSelfValue(agent,self.reward) + self.gama * v)

        else:
            value_sum = (self.Pa) * (self.getSelfValue(agent,self.reward) + self.gama * max_value)
            value.remove(max_value)
            P_not_move = (false_cnt + 1) * P_error # Probability for not moving

            for i, v in enumerate(value):
                if(i == len(value) - 1):
                    # value_sum += P_not_move * (self.getSelfValue(agent) + self.gama * value[-1]) # Probablity of not moving * value of not moving
                    value_sum += P_not_move * (self.getSelfValue(agent,self.reward) + self.gama * value[-1])
                else:
                    # value_sum += P_error * (self.getSelfValue(agent) + self.gama * v)
                    value_sum += P_error * (self.getSelfValue(agent,self.reward) + self.gama * v)


        # if mode != 1:
        #     self.map[agent.row][agent.col] = value_sum
        #     agent.step(max_action)


        return value_sum

    def ValueIteration(self,agent,mp, mode):   
        while(agent.row < self.h):
            while(agent.col < self.w):
                if self.isBarrier(agent) is False:
                    mp[agent.row][agent.col] = (self.calculateValue(agent,self.map,mode))
                    # self.map[agent.row][agent.col] = map_re[agent.row][agent.col]
                    agent.col += 1
                else: 
                    agent.col += 1

            agent.row += 1
            agent.col = 0


        agent.row = 0
        agent.col = 0
        self.map = mp
        return mp
        


class PI(VI):

    def __init__(self, map, Pe, action, gama=1):
        super().__init__(map, Pe, action, gama)
        self.policy = [[np.random.randint(0, len(self.action)) for i in range (self.w)] for j in range(self.h)]
        self.initPolicy()

    def updateValueFunction(self,agent):   
        mp = getGridWorld()
        previous_value = copy.deepcopy(mp)
        cnt = 0
        while True:
            mp = self.ValueIteration(agent,mp,2)
            if previous_value == mp:
                break
            previous_value = copy.deepcopy(mp)
            cnt +=1

        # print(cnt)
        return mp
    

    def updatePolicy(self,agent):
        mp = self.updateValueFunction(agent)
        agent.row = 0
        agent.col = 0
        for i in range(self.w):
            for j in range(self.h):
                if self.isBarrier(agent) is False:
                    sur = self.getSurrounding(agent)
                    cnt, value = self.getAllValue(agent,sur,mp)
                    max_action = cnt[value.index(max(value))]
                    self.policy[agent.row][agent.col] = max_action
                    agent.col += 1
                else: 
                    agent.col += 1

            agent.row += 1
            agent.col = 0


        agent.row = 0
        agent.col = 0



    def initPolicy(self):
        self.policy[1][1] = None
        self.policy[1][2] = None
        self.policy[3][1] = None
        self.policy[3][2] = None

    def policyIteration(self,agent):
        cnt = 0
        old = copy.deepcopy(self.policy)
        while True:
            self.updateValueFunction(agent)
            self.updatePolicy(agent)
            if(old == self.policy):
                break
            old = copy.deepcopy(self.policy)

            cnt += 1

        # print("policy Iteration{}".format(cnt))

        return cnt, self.policy
                
                



    def getPolicyAction(self,agent):
        return self.policy[agent.row][agent.col]

def getGridWorld():
    w, h = (5, 5)
    # Get negative reward done and set all other value to be 0
    map = [[0 for i in range(w)] for j in range(h)]
    for _ in range (w):
        map[_][4] = -1

    # Set barrier at index (1,1) (1,2) (3,1), (3,2)

    map[1][1] = None
    map[1][2] = None
    map[3][1] = None
    map[3][2] = None


    # Set positive rewards
    Rd = 1
    Rs = 10
    map[2][2] = Rd
    map[4][2] = Rs

    return map



def main():

    mp1 = getGridWorld() # k = 1
    mp2 = getGridWorld() # k = 1
    ag = agent(0,0)
    action = [0,1,2,3,4]
    gama = 0.8
    Pe = 0.5
    pi = PI(mp1,Pe,action, gama)
    cnt, policy = pi.policyIteration(ag)
    print(cnt)
    return policy

if __name__ == '__main__':
    a = main()

    
