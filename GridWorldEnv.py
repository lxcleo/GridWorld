
from enum import EnumMeta
import numpy as np

class agent():
    def __init__(self,x,y):
        self.row = x
        self.col = y

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

    def isBarrier(self,agent): # Check if given index is a barrier
        Bar = self.map[agent.row][agent.col] is None
        Negative_Reward = agent.col == 4
        Rs = agent.row == 2 and agent.col == 2
        Rd = agent.row == 4 and agent.col == 2

        if Bar or Negative_Reward or Rs or Rd:
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


    def getUpValue(self,agent):
        return self.map[agent.row - 1][agent.col]

    def getDownValue(self,agent):
        return self.map[agent.row + 1][agent.col]


    def getLeftValue(self,agent):
        return self.map[agent.row][agent.col - 1]

    def getRightValue(self,agent):
        return self.map[agent.row][agent.col + 1]


    def getSelfValue(self,agent):
        return self.map[agent.row][agent.col]

    def getAllValue(self,agent,sur): # Get all value according to getSurronding return value
        cnt = []
        value = []
        if sur[0] is True:
            cnt.append(0)
            value.append(self.getUpValue(agent))
            

        if sur[1] is True:
            cnt.append(1)
            value.append(self.getDownValue(agent))

        if sur[2] is True:
            cnt.append(2)
            value.append(self.getLeftValue(agent))

        if sur[3] is True:
            cnt.append(3)
            value.append(self.getRightValue(agent))

        cnt.append(4)
        value.append(self.getSelfValue(agent))

        return [cnt,value]



    def calculateValue(self,agent):
        value_sum = 0
        sur = self.getSurrounding(agent)
        cnt, value = self.getAllValue(agent,sur)
        index = value.index(max(value))
        max_value = value[index]
        # value_sum = self.Pa * (self.getSelfValue(agent) + self.gama * max_value) # Adding self value to value iteration
        value_sum = self.Pa * (self.gama * max_value)
        P_error = self.Pe / 4
        false_cnt = 0
        for _ in sur:
            if _ is False:
                false_cnt += 1

        value.remove(max_value)
        P_not_move = (false_cnt + 1) * P_error
        for i, v in enumerate(value):
            if(i == len(value) - 1):
                # value_sum += P_not_move * (self.getSelfValue(agent) + self.gama * value[-1]) # Probablity of not moving * value of not moving
                value_sum += P_not_move * (self.gama * value[-1])
            else:
                # value_sum += P_error * (self.getSelfValue(agent) + self.gama * v)
                value_sum += P_error * (self.gama * v)

        return value_sum


    def ValueIteration(self,agent,mp):      
        while(agent.row < self.h):
            while(agent.col < self.w):
                if self.isBarrier(agent) is False:
                    mp[agent.row][agent.col] = self.calculateValue(agent)
                    # self.map[agent.row][agent.col] = map_re[agent.row][agent.col]
                    agent.col += 1
                else: 
                    agent.col += 1

            agent.row += 1
            agent.col = 0


        agent.row = 0
        agent.col = 0
        return mp












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






if __name__ == '__main__':
    mp1 = getGridWorld()
    mp2 = getGridWorld()
    ag = agent(0,0)
    action = [0,1,2,3,4]
    gama = 0.8
    E = env(mp1,0.5,action, gama)
    mp2 = E.ValueIteration(ag,mp2)
    while abs(mp1[4][1] - mp2[4][1]) > 0.01:
        mp1 = E.ValueIteration(ag,mp1)
        mp2 = E.ValueIteration(ag,mp2)


    print(mp1)

    print(222)
