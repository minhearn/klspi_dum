from myAcrobot import myAcrobotEnv
import myAcrobot
from pylevy import levy
import numpy as np
import os
import pandas as pd
from numpy.linalg import inv

def kernel(vec1,vec2,sigma = 10):
    l = len(vec1)
    under = 2*sigma**2
    dif = np.subtract(vec1,vec2)
    s = 0
    for i in range(l):
        s -= dif[i]**2
    return np.exp(s/under)
env = myAcrobotEnv()
env.reset()
dump = env.get_state()
#hyperparameters
ns,na = len(dump), env.action_space.n
mu = 0.001
gamma = 0.9
maxiteration = 10
trials = 1
allowstep = 2000

class Sample(object):
    def __init__(self,state,action,reward,next_state,absorb=False):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
#        self.next_action = next_action
        self.absorb = absorb
    def __repr__(self):
        return 'sample(%s,%s,%s,%s,%s)'%(self.state, self.action, self.reward, self.next_state,self.absorb)

def feature(state,action):
    featureDim = ns*na
    featureVec = np.zeros(featureDim)
    index = action*ns
    for i in range(ns):
        featureVec[i+index] = state[i]
    return featureVec 

random_dic = np.random.uniform(low=-0.1,high=0.1,size=(ns*na,))
def approx(state,action,dic,weights):
    q = 0
    sa = feature(state,action)
    if len(weights)==0:
        random_weights = np.random.rand()
        q = random_weights*kernel(random_dic,sa)
        return q
    k_vec = kRBF(dic,sa)
    for i in range(len(dic)):
        q += weights[i]*k_vec[i]
    return q

def argmin(qs):
    top = float("inf")
    ties = []
    for i in range(len(qs)):
        if qs[i]<top:
            top = qs[i]
            ties = [i]
        elif qs[i]==top:
            ties.append(i)
    return np.random.choice(ties)

def policy(env,state,weights=[],dic=[],epsilon=0):
    
    q_values = []
    for action in range(env.action_space.n):
        q = approx(state,action,dic,weights)
        q_values.append(q)
    if np.random.random()<epsilon:
        return env.action_space.sample()
    else:
        return argmin(q_values)

def collect(env, maxepisodes, maxsteps):
    datasamples = []
    for _ in range(maxepisodes):
        env.reset()
        for i in range(maxsteps):
            state = env.get_state()
            action = policy(env,state)
            obs,reward,done,info = env.step(action)    
            next_state = env.get_state()
            sample = Sample(state,action,reward,next_state)
            if done:
                sample.absorb = True
            datasamples.append(sample)
            if done:
                break
    return datasamples
import time
start = time.perf_counter()
data = collect(env,20,250)
print("collecting time: ", time.perf_counter()-start)
print("size of data: ",len(data))
#data: done

def kRBF(dic,sa):
    res = np.zeros(len(dic))
    for i in range(len(dic)):
        res[i]=kernel(dic[i],sa).item()
    return res.reshape(len(dic),1)

def KRBF(dic):
    l = len(dic)
    res = np.zeros([l,l])
    for i in range(l):
        for j in range(l):
            res[i][j] = kernel(dic[i],dic[j]).item()
    return res

def updateMat(A,a,b,c,row1,col1,row2,col2):
    new_mat = np.zeros([row2,col2])
    for i in range(row1):
        for j in range(col1):
            new_mat[i][j]=A[i][j]
    if (col2>col1):
        for i in range(row1):
            new_mat[i][col2-1]=a[i]
    if (row2>row1):
        for i in range(col1):
            new_mat[row2-1][i]=b[i]
    if row2>row1 and col2>col1:
        new_mat[row2-1][col2-1] = c
def sparsification(data,mu):
    dic = []
    first = feature(data[0].state,data[0].action)
    dic.append(first)
    k_inv = [1.] 
    #ktt = 1
    #K_inv = np.zeros([len(dic),len(dic)])
    #K_inv[0][0] = 1./ktt
    #print(K_inv.shape)
    #c_t = np.zeros(len(dic))
    #c_t[0]=1
    #k_t = np.zeros(len(dic))
    #print(k_t.shape)
    #K_t = np.zeros([1,1])
    for i in range(len(data)):
        sa = feature(data[i].state,data[i].action)
        if len(dic)==0:
            delta=1.
        else:
            k_vec = kRBF(dic,sa)
            c = np.matmul(k_inv,k_vec)
            k_T = np.transpose(k_vec)
            delta = 1. - np.matmul(k_T,c).item()
        if np.abs(delta) > mu:
            dic.append(sa)
            k_mat = KRBF(dic)
            k_inv = inv(k_mat)

        #sa = feature(data[i].state,data[i].action)
        #k_t = kRBF(dic,sa)
        #c_t = np.matmul(K_inv,k_t)
        #ktt = kernel(sa,[sa,]).item()
        #delta = ktt - np.matmul(np.transpose(k_t),c_t).item()
        #if np.abs(delta)>mu:
        #    dic.append(sa)
        #    temp = np.matmul(c_t,np.transpose(c_t))*1/delta
        #    K_inv = np.add(K_inv,temp)
        #    K_inv = updateMat(K_inv, c_t*(0-1/delta), c_t*(0-1/delta), 1/delta,len(dic)-1,len(dic)-1,len(dic),len(dic))
        #   K_t = updateMat(K_t,k_t,k_t,1,len(dic)-1,len(dic)-1,len(dic),len(dic))
    return dic

start = time.perf_counter()
dic = sparsification(data,mu)
#dic: done
print("sparsing time: ",time.perf_counter()-start)
print("dictionary size: ",len(dic));

#KLSTDQ
def KLSTDQ(data,dic,weights):
    A = np.zeros([len(dic),len(dic)])
    b = np.zeros(len(dic)).reshape(len(dic),1)
    for t in range(len(data)):
        sa1 = feature(data[t].state,data[t].action)
        k_cur = kRBF(dic,sa1)
        r = data[t].reward
        if data[t].absorb:
            A = np.add(A,np.matmul(k_cur,np.transpose(k_cur)))
        else:
            sa2 = feature(data[t].next_state,policy(env,data[t].next_state,weights,dic))
            k_nxt = kRBF(dic,sa2)
            A = np.add(A,np.matmul(k_cur,np.transpose(np.subtract(k_cur,gamma*k_nxt))))
        b = np.add(b,r*k_cur)
    new_weights = np.matmul(inv(A),b)
    return new_weights

#weights compare
def compare(a,b):
    if len(a)==len(b):
        diff = np.zeros(len(a))
        for i in range(len(diff)):
            diff[i] = a[i]-b[i]
        return np.linalg.norm(diff)
    else:
        return np.abs(np.linalg.norm(a)-np.linalg.norm(b))

#KLSPI
def KLSPI(env,data,dic,maxiteration):
    weights_list=[]
    weights = []
    i=1
    criterion = 10**-3
    distance = float("inf")
    cost = []
    while((i<=maxiteration)and(distance>criterion)):
        start = time.perf_counter()
        new_weights = KLSTDQ(data,dic,weights)
        print("solving time at iteration ",i,": ",time.perf_counter()-start)
        weights_list.append(new_weights)
        distance = compare(weights,new_weights)
        print("distance between weights and new_weights: ",distance)
        #testing current weights
        cost_per_iter = 0
        numstep = 0
        env.reset()
        while (numstep<allowstep):
            numstep+=1
            curr_state = env.get_state()
            obs,rewards,done,info = env.step(policy(env,curr_state,weights_list[-1],dic))
            cost_per_iter += rewards
            if env._terminal():
                break
        if numstep<allowstep:
            print("succeeded in",numstep)
        else:
            print("failed")
        cost.append(cost_per_iter)
        weights = new_weights
        i+=1
    return weights_list,cost


weights_list,cost = KLSPI(env,data,dic,maxiteration)
##testing
"""
weights_list = KLSPI(env,data,dic,maxiteration)
cost = np.zeros(len(weights_list))
for i in range(len(weights_list)):
    env.reset()
    numstep = 0
    while(numstep<allowstep):
        numstep+=1
        curr_state = env.get_state()
        obs,rewards,done,info = env.step(policy(env,curr_state,weights_list[i],dic))
        cost[i] +=rewards
        if done:
            break
    if numstep<allowstep:
        print("succeed in ",numstep)
    else:
        print("failed")
"""

import matplotlib.pyplot as plt
plt.plot(np.arange(0,len(weights_list)),cost)
plt.savefig("total cost per iteration.pdf")

weights = weights_list[-1]
for trial in range(trials):
    #noise = 0
    env = myAcrobotEnv()
    numstep = 0
    env.reset()
    theta1=[]
    theta2=[]
    dtheta1=[]
    dtheta2=[]
    while(numstep<allowstep):
        numstep+=1
        curr_state = env.get_state()
        theta1.append(curr_state[0])
        theta2.append(curr_state[1])
        dtheta1.append(curr_state[2])
        dtheta2.append(curr_state[3])
        obs,reward,done,info = env.step(policy(env,curr_state,weights,dic))
        if env._terminal():
            break
    last_state = env.get_state()
    theta1.append(last_state[0])
    theta2.append(last_state[1])
    dtheta1.append(last_state[2])
    dtheta2.append(last_state[3])
    outdir = './no-noise'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filename1 = 'position_'+ str(trial) + '.csv'
    fullname1 = os.path.join(outdir,filename1)
    dict1 = {'theta1':theta1,'theta2':theta2,'dtheta1':dtheta1,'dtheta2':dtheta2}
    df1 = pd.DataFrame(dict1)
    df1.to_csv(fullname1)
    env.close()
        
    #noise = gaussian
    env = myAcrobotEnv()
    numstep = 0
    env.reset()
    theta1=[]
    theta2=[]
    dtheta1=[]
    dtheta2=[]
    fake1 = []
    fake2 = []
    fake3 = []
    fake4 = []
    while(numstep<allowstep):
        numstep+=1
        curr_state = env.get_state()
        theta1.append(curr_state[0])
        theta2.append(curr_state[1])
        dtheta1.append(curr_state[2])
        dtheta2.append(curr_state[3])
        #noise_state = curr_state + np.random.normal(0,np.pi/60,4)
        #noise_state[0] = myAcrobot.wrap(noise_state[0],-np.pi,np.pi)
        #noise_state[1] = myAcrobot.wrap(noise_state[1],-np.pi,np.pi)
        #noise_state[2] = myAcrobot.bound(noise_state[2],-env.MAX_VEL_1,env.MAX_VEL_1)
        #noise_state[3] = myAcrobot.bound(noise_state[3],-env.MAX_VEL_2,env.MAX_VEL_2) 
        #fake1.append(noise_state[0])
        #fake2.append(noise_state[1])
        #fake3.append(noise_state[2])
        #fake4.append(noise_state[3])
        #env.set(noise_state)
        obs,reward,done,info = env.step(policy(env,curr_state,weights,dic))
        noise_state = env.get_state() + np.random.normal(0,np.pi/60,4)
        noise_state[0] = myAcrobot.wrap(noise_state[0],-np.pi,np.pi)
        noise_state[1] = myAcrobot.wrap(noise_state[1],-np.pi,np.pi)
        noise_state[2] = myAcrobot.bound(noise_state[2],-env.MAX_VEL_1,env.MAX_VEL_1)
        noise_state[3] = myAcrobot.bound(noise_state[3],-env.MAX_VEL_2,env.MAX_VEL_2) 
        fake1.append(noise_state[0])
        fake2.append(noise_state[1])
        fake3.append(noise_state[2])
        fake4.append(noise_state[3])
        env.set(noise_state)

        if env._terminal():
            break
    last_state = env.get_state()
    theta1.append(last_state[0])
    theta2.append(last_state[1])
    dtheta1.append(last_state[2])
    dtheta2.append(last_state[3])
    outdir = './gaussian-noise'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filename1 = 'position_'+ str(trial) + '.csv'
    fullname1 = os.path.join(outdir,filename1)
    dict1 = {'theta1':theta1,'theta2':theta2,'dtheta1':dtheta1,'dtheta2':dtheta2}
    df1 = pd.DataFrame(dict1)
    df1.to_csv(fullname1)
    filename2 = 'fake_position_'+str(trial) + '.csv'
    fullname2 = os.path.join(outdir,filename2)
    dict2 = {'theta1':fake1,'theta2':fake2,'dtheta1':fake3,'dtheta2':fake4}
    df2 = pd.DataFrame(dict2)
    df2.to_csv(fullname2)
    env.close()

    #noise = uniform
    env = myAcrobotEnv()
    numstep = 0
    env.reset()
    theta1=[]
    theta2=[]
    dtheta1=[]
    dtheta2=[]
    fake1 = []
    fake2 = []
    fake3 = []
    fake4 = []
    while(numstep<allowstep):
        numstep+=1
        curr_state = env.get_state()
        theta1.append(curr_state[0])
        theta2.append(curr_state[1])
        dtheta1.append(curr_state[2])
        dtheta2.append(curr_state[3])
        #noise_state = curr_state + np.random.uniform(-np.pi/60,np.pi/60,4)
        #noise_state[0] = myAcrobot.wrap(noise_state[0],-np.pi,np.pi)
        #noise_state[1] = myAcrobot.wrap(noise_state[1],-np.pi,np.pi)
        #noise_state[2] = myAcrobot.bound(noise_state[2],-env.MAX_VEL_1,env.MAX_VEL_1)
        #noise_state[3] = myAcrobot.bound(noise_state[3],-env.MAX_VEL_2,env.MAX_VEL_2)
        #fake1.append(noise_state[0])
        #fake2.append(noise_state[1])
        #fake3.append(noise_state[2])
        #fake4.append(noise_state[3])
        #env.set(noise_state)
        obs,reward,done,info = env.step(policy(env,curr_state,weights,dic))
        noise_state = env.get_state() + np.random.uniform(-np.pi/60,np.pi/60,4)
        noise_state[0] = myAcrobot.wrap(noise_state[0],-np.pi,np.pi)
        noise_state[1] = myAcrobot.wrap(noise_state[1],-np.pi,np.pi)
        noise_state[2] = myAcrobot.bound(noise_state[2],-env.MAX_VEL_1,env.MAX_VEL_1)
        noise_state[3] = myAcrobot.bound(noise_state[3],-env.MAX_VEL_2,env.MAX_VEL_2)
        fake1.append(noise_state[0])
        fake2.append(noise_state[1])
        fake3.append(noise_state[2])
        fake4.append(noise_state[3])
        env.set(noise_state)

        if env._terminal():
            break
    last_state = env.get_state()
    theta1.append(last_state[0])
    theta2.append(last_state[1])
    dtheta1.append(last_state[2])
    dtheta2.append(last_state[3])
    outdir = './uniform-noise'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filename1 = 'position_'+ str(trial) + '.csv'
    fullname1 = os.path.join(outdir,filename1)
    dict1 = {'theta1':theta1,'theta2':theta2,'dtheta1':dtheta1,'dtheta2':dtheta2}
    df1 = pd.DataFrame(dict1)
    df1.to_csv(fullname1)
    filename2 = 'fake_position_'+str(trial) + '.csv'
    fullname2 = os.path.join(outdir,filename2)
    dict2 = {'theta1':fake1,'theta2':fake2,'dtheta1':fake3,'dtheta2':fake4}
    df2 = pd.DataFrame(dict2)
    df2.to_csv(fullname2)
    env.close()

    #noise = a-stable
    env = myAcrobotEnv()
    numstep = 0
    env.reset()
    theta1=[]
    theta2=[]
    dtheta1=[]
    dtheta2=[]
    fake1 = []
    fake2 = []
    fake3 = []
    fake4 = []
    while(numstep<allowstep):
        numstep+=1
        curr_state = env.get_state()
        theta1.append(curr_state[0])
        theta2.append(curr_state[1])
        dtheta1.append(curr_state[2])
        dtheta2.append(curr_state[3])
        #noise_state = curr_state + levy.random(1.25,0,mu=0.,sigma=np.pi/60,shape=(4,))
        #noise_state[0] = myAcrobot.wrap(noise_state[0],-np.pi,np.pi)
        #noise_state[1] = myAcrobot.wrap(noise_state[1],-np.pi,np.pi)
        #noise_state[2] = myAcrobot.bound(noise_state[2],-env.MAX_VEL_1,env.MAX_VEL_1)
        #noise_state[3] = myAcrobot.bound(noise_state[3],-env.MAX_VEL_2,env.MAX_VEL_2)
        #fake1.append(noise_state[0])
        #fake2.append(noise_state[1])
        #fake3.append(noise_state[2])
        #fake4.append(noise_state[3])
        #env.set(noise_state)
        obs,reward,done,info = env.step(policy(env,curr_state,weights,dic))
        noise_state = env.get_state() + levy.random(1.25,0,mu=0.,sigma=np.pi/60,shape=(4,))
        noise_state[0] = myAcrobot.wrap(noise_state[0],-np.pi,np.pi)
        noise_state[1] = myAcrobot.wrap(noise_state[1],-np.pi,np.pi)
        noise_state[2] = myAcrobot.bound(noise_state[2],-env.MAX_VEL_1,env.MAX_VEL_1)
        noise_state[3] = myAcrobot.bound(noise_state[3],-env.MAX_VEL_2,env.MAX_VEL_2)
        fake1.append(noise_state[0])
        fake2.append(noise_state[1])
        fake3.append(noise_state[2])
        fake4.append(noise_state[3])
        env.set(noise_state)

        #env.set(noise_state)
        if env._terminal():
            break
    last_state = env.get_state()
    theta1.append(last_state[0])
    theta2.append(last_state[1])
    dtheta1.append(last_state[2])
    dtheta2.append(last_state[3])
    outdir = './stable-noise'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filename1 = 'position_'+ str(trial) + '.csv'
    fullname1 = os.path.join(outdir,filename1)
    dict1 = {'theta1':theta1,'theta2':theta2,'dtheta1':dtheta1,'dtheta2':dtheta2}
    df1 = pd.DataFrame(dict1)
    df1.to_csv(fullname1)
    filename2 = 'fake_position_'+str(trial) + '.csv'
    fullname2 = os.path.join(outdir,filename2)
    dict2 = {'theta1':fake1,'theta2':fake2,'dtheta1':fake3,'dtheta2':fake4}
    df2 = pd.DataFrame(dict2)
    df2.to_csv(fullname2)

    env.close()

