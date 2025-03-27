import traci
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import xml.etree.ElementTree as ET
import math
from scipy.interpolate import interp2d,RegularGridInterpolator
import pickle
import torch.nn.init as init
import copy


# 设置matplotlib参数以支持中文显示
plt.rcParams['font.sans-serif']=['SimHei']  # 将字体设置为SimHei，用于显示中文
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' # 设置环境变量以允许KMP库重复加载

######################  各相位的红灯持续时长
def get_waittimeforphase(inter_id,wait_time_for_phase_list):
    #获取各个放行流向的累计等待时间
    current_signal_state=traci.trafficlight.getRedYellowGreenState(inter_id)  #当前灯色状态
    #print(current_signal_state)
    #将`current_signal_state` 列表中的每个状态与其索引配对，形成一个可迭代的元组序列。这允许我们同时访问每个状态的值和它的索引（即相位编号）。
    for current_state,iindex in zip(current_signal_state,range(0,len(current_signal_state))):
        if current_state=="G" or current_state=="g":#变成绿灯
            wait_time_for_phase_list[iindex]=0
        else:
            wait_time_for_phase_list[iindex]=wait_time_for_phase_list[iindex]+1
    #print(wait_time_for_phase_list)
    return wait_time_for_phase_list
######################  交叉口的信号配时状态获取
def get_signal_state(inter_id,wait_time_for_phase_list,max_green_duration_dicts,
                    max_phase_id,min_green_duration,max_green_duration,max_red_wait_t):
    ########################  提取信号灯状态
    signals=traci.trafficlight.getRedYellowGreenState(inter_id)  #当前灯色状态
    ###将灯色状态状态成数字
    signal_state_dicts={"r":0,"y":1,"G":2,"g":3,"O":4}
    signal_state_list=[signal_state_dicts[state1] for state1 in signals]
    current_phase_id=int(traci.trafficlight.getPhase(inter_id))   #当前放行相位
    SpentDuration=int(traci.trafficlight.getSpentDuration(inter_id))   #/max_green_duration  #当前相位已放行时长
    print(current_phase_id,signals,SpentDuration)
    max_green_time=int(max_green_duration_dicts[current_phase_id])    #/max_green_duration            #当前相位最大绿灯时长
    ####各相位的等待时长
    wait_time_for_phase_list1=[waitt for waitt in  wait_time_for_phase_list]
    signal_state=[current_phase_id]+[SpentDuration]+[max_green_time]+signal_state_list+wait_time_for_phase_list1
    #print(signal_state)
    #####加一个安全策略，是否触发了安全策略，超过了极限的最大等待时间等
    is_start_safety_protect_flag=[1 if waitt>max_red_wait_t else 0 for waitt in  wait_time_for_phase_list]
    
    return signal_state+is_start_safety_protect_flag
######################  车辆的位置、速度状态获取
def get_vehstate(lanelist,celllength,lanelengthdicts,maxspeed):
    ############交通状态空间的构建
    #由如下构成：
    #进口道车辆位置、车辆速度

    #####每个进口道车辆的位置和速度信息 指定一个车道list，按顺序组合成状态向量   lanelist，以及lane到交叉口的距离dict lane2interdisdicts
    #####以及lane对应划分的单元格，在第几个单元格上
    ######################提取车辆的运行状态
    vehsate=[]  #车辆运行状态向量
    for laneid in lanelist:
        lanevehpos=[0]*int(np.ceil(lanelengthdicts[laneid]/celllength))  #初始化车道上车辆的位置向量
        lanevehspeed=[0]*int(np.ceil(lanelengthdicts[laneid]/celllength))  #初始化车道上车辆的速度向量
        vehlist=traci.lane.getLastStepVehicleIDs(laneid)
        for vehid in vehlist:   
            cellindex=int(np.floor((lanelengthdicts[laneid]-traci.vehicle.getLanePosition(vehid))/celllength))    #在第几个单元格   #celllength单元格长度
            lanevehpos[cellindex]=1
            lanevehspeed[cellindex]=int(traci.vehicle.getSpeed(vehid)*10)   #/maxspeed
        vehsate=vehsate+lanevehpos+lanevehspeed
    #print(vehsate)
    return vehsate
######################  交叉口停车排队车辆数
def get_stopvehcount(lanelist,stopspeed):
    stopvehcount=0
    for laneid in lanelist:
        vehlist=traci.lane.getLastStepVehicleIDs(laneid)
        for vehid in vehlist:
            if traci.vehicle.getSpeed(vehid)<=stopspeed:
                stopvehcount=stopvehcount+1
    return stopvehcount
######################  交叉口通过车辆数
def get_passvehfort(t,E1detlist,pd_passveh):
    #获取当前时刻通过车检器的车辆数
    for detid in E1detlist:
        passveh=0
        vehdatalist=traci.inductionloop.getVehicleData(detid)
        for vehdata in vehdatalist:
            if vehdata[3]!=-1:
                passveh=passveh+1
        pdtemp=pd.DataFrame.from_dict({"t":t,"detid":[detid],"passveh":[passveh]})
        pd_passveh=pd.concat([pd_passveh,pdtemp])
    return pd_passveh
######################  奖励函数的构建
def get_reward(inter_id,pd_passveh,laststopveh,
               startcountt,endcountt,lanelist,wait_time_for_phase_list,
               max_green_duration_dicts,overwait_factor,overgreen_factor,
               stopspeed,max_red_wait_t,Saturation_headway):
    #单位通过的车辆数   单位时间 才有比较的

    # startcountt  #统计开始时间  去掉黄灯和全红的时间
    # endcountt     #统计结束时间
    sumpassveh=sum(pd_passveh[(pd_passveh["t"]>=startcountt)&(pd_passveh["t"]<=endcountt)]["passveh"])
    #print("通过车辆数：",sumpassveh)
    passvehreward=sumpassveh/(endcountt-startcountt)*30   #折算成30s预计的通过量
    #排队车辆数
    currentstopvehcount=get_stopvehcount(lanelist,stopspeed)
    stopvehreward=currentstopvehcount-laststopveh
    #机动车红灯超限量  max_red_wait_t
    sumoverwaitt=sum([phasewaitt-max_red_wait_t if phasewaitt>max_red_wait_t else 0 for phasewaitt in wait_time_for_phase_list])  #总的超限时长
    overwaitveh=sumoverwaitt/Saturation_headway  #时间转换成车辆数
    overwaitreward=overwaitveh/(endcountt-startcountt)*overwait_factor  #
    ##机动车绿灯是否超过最大绿灯
    current_phase_id=traci.trafficlight.getPhase(inter_id)
    SpentDuration=traci.trafficlight.getSpentDuration(inter_id)
    
    overgreenreward=0
    if current_phase_id in max_green_duration_dicts:
        if max_green_duration_dicts[current_phase_id]<SpentDuration:
            overgreenreward=(SpentDuration-max_green_duration_dicts[current_phase_id])/Saturation_headway*overgreen_factor  #时间转换成车辆数,并有一个系数overgreen_factor


    rewards=passvehreward-stopvehreward-overwaitreward-overgreenreward
    #print(startcountt,endcountt)
    #print("当前相位：{}，当前相位持续的绿灯时长：{},允许的最大绿灯时长：{}".format(current_phase_id,SpentDuration,max_green_duration_dicts[current_phase_id]))
    #print("车辆奖励：{}，停车车辆数惩罚：{}，超长等待惩罚：{}，超长绿灯惩罚：{}，总奖励：{}".format(passvehreward,stopvehreward,overwaitreward,overgreenreward,rewards))

    return rewards
######################  经验回放池
class ReplayBuffer():
    def __init__(self, capacity):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.buffer = collections.deque(maxlen=capacity)
    # 将数据以元组形式添加进经验池 dqn基本构成要素
    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list, len=32
        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state= zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)
    # 目前队列长度
    def size(self):
        return len(self.buffer)
######################  构造深度学习网络
class Net(nn.Module):
    # 构造只有一个隐含层的网络
    def __init__(self, n_states,n_hidden, n_actions):
        super(Net, self).__init__()
        # [b,n_states]-->[b,n_hidden]
        
        self.embedding=nn.Embedding(n_states,1)
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.relu=nn.ReLU()
        # [b,n_hidden]-->[b,n_actions]
        self.fc2 = nn.Linear(n_hidden, n_actions)
    # 前传
    def forward(self, x):  # [b,n_states]
        #print("xxxxxx:{}".format(x))
        x=self.embedding(x)
        x=x.reshape(x.size(0),-1)
        #print("embedding后的输出：",x)
        x = self.fc1(x)
        x=self.relu(x)
        x = self.fc2(x)
        return x
######################  构造深度强化学习DQN智能体
class DQN:
    #（1）初始化
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        # 属性分配
        self.n_states = n_states  # 状态的特征数
        self.n_hidden = n_hidden  # 隐含层个数
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device  # 在GPU计算
        # 计数器，记录迭代次数
        self.count = 0

        # 构建2个神经网络，相同的结构，不同的参数
        # 实例化训练网络  [b,4]-->[b,2]  输出动作对应的奖励
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)

        # 应用Xavier正态分布初始化 权重初始化 避免梯度消失或者爆炸问题 根据层的输入和输出维度自动调整权重的标准差 使得初始时数据的分布能够在网络中稳定传递。
        init.xavier_normal_(self.q_net.fc1.weight, gain=1)
        init.xavier_normal_(self.q_net.fc2.weight, gain=1)
        # 实例化目标网络
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        # 应用Xavier正态分布初始化
        init.xavier_normal_(self.target_q_net.fc1.weight, gain=1)
        init.xavier_normal_(self.target_q_net.fc2.weight, gain=1)
        # 最优的网络
        self.bestq_net=self.q_net
        self.bestq_netmin_loss=100000
        self.bestq_netinwhichepoch=-1#在哪个训练周期中有最好的q网络
        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=self.learning_rate)#lr是 lerning rate简写
        #self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=self.learning_rate, alpha=0.9)
        # 保存训练误差数据
        self.losslist=[]
        #self.criterion = nn.SmoothL1Loss()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer , step_size=20000, gamma=0.1)
  


    #（2）动作选择
    def take_action(self, state,connectdim,connectid2greenphaseiddicts,phaseid2actiondicts):
        #print("是否超限：{}".format(state[-connectdim:]))
        if sum(state[-connectdim:])>0:
            waittlist=state[-2*connectdim:-connectdim]
            # maxwaittime=max(waittlist)
            
            connectid=waittlist.argmax().item() #最大等待的流向（连接器） argmax 最大索引值
            #print(waittlist,type(waittlist),connectid)
            greenphaseid=connectid2greenphaseiddicts[connectid]   #该连接器对应哪个绿灯相位 connect id to green phase
            action=phaseid2actiondicts[greenphaseid]        #该绿灯相位对应是哪个动作 connect id to action
        else:
            # 维度扩充，给行增加一个维度，并转换为张量
            state = torch.LongTensor(state[np.newaxis, :])
            ###### 判断状态向量里面是否有车辆等待时间超过最大的极限等待时间的，如果有 直接切换至对应的相位，并加一个
            # 如果小于该值就取最大的值对应的索引
            if np.random.random() < self.epsilon:  # 0-1
                # 前向传播获取该状态对应的动作的reward
                actions_value = self.q_net(state)
                # print(actions_value)
                # print(lastaction)
                # print(actions_value[:,0:lastaction+2])
                # 获取reward最大值对应的动作索引
                
                # print(actions_valuenew)
                action = actions_value.argmax().item()  # int
                # print(action)
                # if action>lastaction+1:
                #      action=lastaction+1
            # 如果大于该值就随机探索
            else:
                # 随机选择一个动作
                action = np.random.randint(self.n_actions)
        return action

    #（3）网络训练
    def update(self,transition_dict):  # 传入经验池中的batch个样本
        # 获取当前时刻的状态 array_shape=[b,4]
        states = torch.tensor(transition_dict['states'], dtype=torch.long)  #为了在GPU上跑 用tensor
        # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1,1)
        # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1)
        # 下一时刻的状态 array_shape=[b,4]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.long)


        # 输入当前状态，得到采取各运动得到的奖励 [b,4]==>[b,2]==>[b,1]
        # 根据actions索引在训练网络的输出的第1维度上获取对应索引的q值（state_value）
        q_values = self.q_net(states).gather(1, actions)  # [b,1]
        # 下一时刻的状态[b,4]-->目标网络输出下一时刻对应的动作q值[b,2]-->
        # 选出下个状态采取的动作中最大的q值[b]-->维度调整[b,1]
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)
        # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
        q_targets = rewards + self.gamma * max_next_q_values 

        # 目标网络和训练网络之间的均方误差损失
        #dqn_loss=self.criterion(q_values, q_targets.unsqueeze(1))
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        print("训练误差:{}".format(dqn_loss.detach().numpy()),"  ")
        self.losslist.append(dqn_loss.detach().numpy())
        if dqn_loss<self.bestq_netmin_loss:
            self.bestq_net=self.q_net
            self.bestq_netmin_loss=dqn_loss
            self.bestq_netinwhichepoch=self.count
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播参数更新
        dqn_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        # 对训练网络更新
        self.optimizer.step()
        # 更新学习率
        self.lr_scheduler.step()

        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            print("更新目标Q网络")
        
        self.count += 1
######################  初始化填充经验池 
def fixedcontrol_fillreplay_buffer2minsize(signalcontrol_class):
    #先按定时控制的方式 填充经验池，使其满足最小的训练样本数

    for i in range(0,signalcontrol_class.fixedcontrolepoch2fillreplay_bufferminsize):
        currentStep=0.0
        seed=str(random.randint(1,10000))
        pd_passveh=pd.DataFrame()
        traci.start(['sumo','-c',signalcontrol_class.sumocfgxmlfile,'--seed',str(seed)])  #启用gui界面，连接到SUMO的TraCI服务器
        traci.trafficlight.setProgram(signalcontrol_class.inter_id,signalcontrol_class.initprogramid)  
        traci.trafficlight.setPhase(signalcontrol_class.inter_id,signalcontrol_class.initphaseid)  #
        traci.simulationStep(currentStep)
        currentStep=currentStep+1
        startdecisiont=signalcontrol_class.warmupcycle*signalcontrol_class.cyclelength+signalcontrol_class.min_green_durationdicts[signalcontrol_class.initphaseid]
        decisiontdicts={}
        decisiontdicts[startdecisiont]=signalcontrol_class.initphaseid
        lastsignal_state=traci.trafficlight.getRedYellowGreenState(signalcontrol_class.inter_id) 
        wait_time_for_phase_list=[0]*len(lastsignal_state)
        print("初始化wait_time_for_phase_list:{}".format(wait_time_for_phase_list))
        while currentStep <=signalcontrol_class.endtime:
            # 执行一个仿真步骤
            traci.simulationStep(currentStep)
            currentsim_t=int(traci.simulation.getTime())
            #print("当前时刻：{}".format(traci.simulation.getTime()))
            pd_passveh=get_passvehfort(currentsim_t,signalcontrol_class.E1detlist,pd_passveh)
            wait_time_for_phase_list=get_waittimeforphase(signalcontrol_class.inter_id,wait_time_for_phase_list)
            #print(currentsim_t,wait_time_for_phase_list)
            current_phase_id=traci.trafficlight.getPhase(signalcontrol_class.inter_id)
            currentStep=currentStep+1
            #当前时刻的排放数据
            # #考虑预热时间
            if currentsim_t in decisiontdicts:  
                # 提取交叉口运行状态
                vehstate=get_vehstate(signalcontrol_class.lanelist,signalcontrol_class.celllength,signalcontrol_class.lanelengthdicts,signalcontrol_class.maxspeed)
                signal_state=get_signal_state(signalcontrol_class.inter_id,wait_time_for_phase_list,signalcontrol_class.max_green_duration_dicts,signalcontrol_class.max_phase_id,signalcontrol_class.min_green_duration,signalcontrol_class.max_green_duration,signalcontrol_class.max_red_wait_t)
                currentstate=vehstate+signal_state
                print("                ",len(currentstate),"     ")
                currentstate=np.array(currentstate)
                currentstate=torch.tensor(currentstate)

                nextgreenphaseid=signalcontrol_class.current_phase_id2nextgreenphaseiddicts[current_phase_id]
                nextdecisiont=currentsim_t+signalcontrol_class.yt+signalcontrol_class.min_green_durationdicts[nextgreenphaseid]  #下一个决策时间点
                decisiontdicts[nextdecisiont]=nextgreenphaseid



                if currentsim_t<=startdecisiont:
                    last_state=currentstate
                    last_action=signalcontrol_class.phaseid2actiondicts[signalcontrol_class.current_phase_id2nextgreenphaseiddicts[current_phase_id]]
                    last_decisiont=currentsim_t
                    laststopveh=get_stopvehcount(signalcontrol_class.lanelist,signalcontrol_class.stopspeed)

                else:
                    reward=get_reward(signalcontrol_class.inter_id,pd_passveh,laststopveh,last_decisiont,currentsim_t,signalcontrol_class.lanelist,wait_time_for_phase_list,signalcontrol_class.max_green_duration_dicts,signalcontrol_class.overwait_factor,signalcontrol_class.overgreen_factor,signalcontrol_class.stopspeed,signalcontrol_class.max_red_wait_t,signalcontrol_class.Saturation_headway)
                
                    # 添加经验池
                    
                    signalcontrol_class.replay_buffer.add(last_state, last_action, reward, currentstate)
                    
                    # 当经验池超过一定数量后，训练网络
                    if signalcontrol_class.replay_buffer.size() > signalcontrol_class.min_size:
                        if signalcontrol_class.replay_buffer.size()%signalcontrol_class.trainfreq==0:
                            # 从经验池中随机抽样作为训练集
                            s, a, r, ns= signalcontrol_class.replay_buffer.sample(signalcontrol_class.batch_size)
                            # 构造训练集
                            transition_dict = {
                                'states': s,
                                'actions': a,
                                'next_states': ns,
                                'rewards': r,
                            }
                            # 网络更新
                            signalcontrol_class.agent.update(transition_dict)
    
                    #####更新last_state、last_action、last_decisiont
                    last_state=currentstate
                    last_action=signalcontrol_class.phaseid2actiondicts[nextgreenphaseid]        #下一个绿灯相位             #下一个绿灯相位
                    last_decisiont=currentsim_t
                    laststopveh=get_stopvehcount(signalcontrol_class.lanelist,signalcontrol_class.stopspeed)
        traci.close()
######################  DQN智能体训练
def DQN_train(signalcontrol_class):
    #######模型训练
    print("模型正式开始训练")
    return_list = []  # 记录每个回合的回报
    # 训练模型
    for i in range(signalcontrol_class.trainepoch):  
        print("第{}轮训练".format(i))
        # 每个回合开始前重置环境
        # 记录每个回合的回报
        episode_return = 0

        currentStep=0.0
        seed=str(random.randint(1,10000))
        pd_passveh=pd.DataFrame()


        traci.start(['sumo','-c',signalcontrol_class.sumocfgxmlfile,'--seed',str(seed)])  #启用gui界面，连接到SUMO的TraCI服务器
        traci.trafficlight.setProgram(signalcontrol_class.inter_id,signalcontrol_class.initprogramid)  
        traci.trafficlight.setPhase(signalcontrol_class.inter_id,signalcontrol_class.initphaseid)  #
        traci.simulationStep(currentStep)
        currentStep=currentStep+1
        startdecisiont=signalcontrol_class.warmupcycle*signalcontrol_class.cyclelength+signalcontrol_class.min_green_durationdicts[signalcontrol_class.initphaseid]

        

        decisiontdicts={}
        decisiontdicts[startdecisiont]=signalcontrol_class.initphaseid
        changephasetdicts={}


        lastsignal_state=traci.trafficlight.getRedYellowGreenState(signalcontrol_class.inter_id) 
        wait_time_for_phase_list=[0]*len(lastsignal_state)
        while currentStep <=signalcontrol_class.endtime:
            # 执行一个仿真步骤
            traci.simulationStep(currentStep)
            currentsim_t=int(traci.simulation.getTime())
            pd_passveh=get_passvehfort(currentsim_t,signalcontrol_class.E1detlist,pd_passveh)
            #print(pd_passveh)
            wait_time_for_phase_list=get_waittimeforphase(signalcontrol_class.inter_id,wait_time_for_phase_list)
            current_phase_id=traci.trafficlight.getPhase(signalcontrol_class.inter_id)         
            currentStep=currentStep+1



            if currentsim_t in decisiontdicts:  #进行智能体决策
                # 提取交叉口运行状态
                vehstate=get_vehstate(signalcontrol_class.lanelist,signalcontrol_class.celllength,signalcontrol_class.lanelengthdicts,signalcontrol_class.maxspeed)
                signal_state=get_signal_state(signalcontrol_class.inter_id,wait_time_for_phase_list,signalcontrol_class.max_green_duration_dicts,signalcontrol_class.max_phase_id,signalcontrol_class.min_green_duration,signalcontrol_class.max_green_duration,signalcontrol_class.max_red_wait_t)
                currentstate=vehstate+signal_state
                #print("                ",len(currentstate),"     ")
                currentstate=np.array(currentstate)
                #currentstate=torch.tensor(currentstate)
                if currentsim_t<=startdecisiont:
                    last_state=currentstate
                    last_action=signalcontrol_class.phaseid2actiondicts[signalcontrol_class.current_phase_id2nextgreenphaseiddicts[current_phase_id]]
                    last_decisiont=currentsim_t
                    laststopveh=get_stopvehcount(signalcontrol_class.lanelist,signalcontrol_class.stopspeed)
                else:
                    # 计算上一个决策点当当前决策点的奖励
                    #print("当前时刻：{},上一个决策时刻点：{}，当前决策时刻点：{}".format(currentsim_t,last_decisiont,currentsim_t))
                    reward=get_reward(signalcontrol_class.inter_id,pd_passveh,laststopveh,last_decisiont,currentsim_t,signalcontrol_class.lanelist,wait_time_for_phase_list,signalcontrol_class.max_green_duration_dicts,signalcontrol_class.overwait_factor,signalcontrol_class.overgreen_factor,signalcontrol_class.stopspeed,signalcontrol_class.max_red_wait_t,signalcontrol_class.Saturation_headway)
                    # 添加经验池
                    signalcontrol_class.replay_buffer.add(last_state, last_action, reward, currentstate)
                    # 更新回合回报
                    episode_return += reward


                # 当经验池超过一定数量后，训练网络
                if signalcontrol_class.replay_buffer.size() > signalcontrol_class.min_size:
                    if signalcontrol_class.replay_buffer.size()%signalcontrol_class.trainfreq==0:
                        # 从经验池中随机抽样作为训练集
                        s, a, r, ns= signalcontrol_class.replay_buffer.sample(signalcontrol_class.batch_size)
                        # 构造训练集
                        transition_dict = {
                            'states': s,
                            'actions': a,
                            'next_states': ns,
                            'rewards': r,
                        }
                        # 网络更新
                        signalcontrol_class.agent.update(transition_dict)
    

                #通过智能体决策下一步动作
                action = signalcontrol_class.agent.take_action(currentstate,signalcontrol_class.connectdim,signalcontrol_class.connectid2greenphaseiddicts,signalcontrol_class.phaseid2actiondicts)
                #currentgreenphaseid=traci.trafficlight.getPhase(signalcontrol_class.inter_id)    #当前放行相位
                nextgreenphaseid=signalcontrol_class.action2phaseiddicts[action]
                #print("当前时刻：{},当前相位id:{},下一个相位id:{}".format(currentStep,current_phase_id,nextgreenphaseid))

                #更新last_state, last_action,last_decisiont
                last_state=currentstate
                last_action=action
                last_decisiont=currentsim_t
                laststopveh=get_stopvehcount(signalcontrol_class.lanelist,signalcontrol_class.stopspeed)


                if current_phase_id==nextgreenphaseid: #延长当前绿灯相位一个单位绿灯时长

                    
                    #更新决策时间点
                    decisiontdicts[currentsim_t+signalcontrol_class.Unit_Extension_Time]=current_phase_id  #下一个决策时间点
                    ##执行延长绿灯的操作
                    traci.trafficlight.setPhaseDuration(signalcontrol_class.inter_id,signalcontrol_class.Unit_Extension_Time) #当前相位延长一个单位绿灯时间
                    changephasetdicts[currentsim_t+signalcontrol_class.Unit_Extension_Time]={"change2phaseid":"" ,"phasetype":"decision"}   #切换至全红相位的时刻点
                    

                else:
                    #更新决策时间点
                    nextdecisiont=currentsim_t+signalcontrol_class.yt+signalcontrol_class.min_green_durationdicts[nextgreenphaseid]  #下一个决策时间点 delete allr
                    decisiontdicts[nextdecisiont]=nextgreenphaseid
                    #更新相位切换时刻点
                    #执行相位切换的操作
                    traci.trafficlight.setPhase(signalcontrol_class.inter_id,current_phase_id+1)  #切换至黄灯相位
                   #changephasetdicts[currentsim_t+signalcontrol_class.yt]={"change2phaseid":current_phase_id+2 ,"phasetype":"allr"}   #切换至全红相位的时刻点
                    changephasetdicts[currentsim_t+signalcontrol_class.yt]={"change2phaseid":nextgreenphaseid,"phasetype":"G"}    #切换至指定的绿灯相位nextgreenphaseid 的时刻点
    
            if currentsim_t in changephasetdicts:  #进行相位切换  
                if changephasetdicts[currentsim_t]["change2phaseid"]!="":
                    traci.trafficlight.setPhase(signalcontrol_class.inter_id,changephasetdicts[currentsim_t]["change2phaseid"]) 
 

        traci.close()  
        # 记录每个回合的回报
        print(i,episode_return)
        return_list.append(episode_return)

    episodes_list = list(range(len(return_list)))



    ########模型的保存
    # 保存最后一轮训练后的模型权重和偏置
    torch.save(signalcontrol_class.agent.q_net.state_dict(), signalcontrol_class.q_network_savefilename)
    # 保存最优性能的模型权重和偏置
    print("最优模型出现在第{}轮,loss:{}".format(signalcontrol_class.agent.bestq_netinwhichepoch,signalcontrol_class.agent.bestq_netmin_loss))
    torch.save(signalcontrol_class.agent.bestq_net.state_dict(), signalcontrol_class.q_network_best_savefilename)
    # 保存训练误差数据
    pdloss=pd.DataFrame.from_dict({"iteration":list(range(len(signalcontrol_class.agent.losslist))),"loss":signalcontrol_class.agent.losslist})
    pdloss.to_csv(signalcontrol_class.train_loss_savefilename)

    # 保存deque到pkl文件
    with open(signalcontrol_class.deque_savefilename, 'wb') as f:
        pickle.dump(signalcontrol_class.replay_buffer.buffer, f)
    # 保存回合的累计奖励
    pdepisodesreward=pd.DataFrame.from_dict({"episodes":episodes_list,"reward":return_list})
    pdepisodesreward.to_csv(signalcontrol_class.episodesreward_savefilename)

    # 各回合累积奖励绘图
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN Returns')
    plt.savefig( signalcontrol_class.rewardfigname)
    plt.close()
    # 训练误差曲线
    plt.plot(list(range(len(signalcontrol_class.agent.losslist))), signalcontrol_class.agent.losslist)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('DQN trainloss')
    plt.savefig( signalcontrol_class.lossfigname)
######################  DQN智能体推理
def DQNinference(signalcontrol_class):
    #模型推理
    entryexit_result={} 
    currentStep=0.0
    traci.start(['sumo','-c',signalcontrol_class.sumocfgxmlfile,'--seed',str(signalcontrol_class.seed)])  #启用gui界面，连接到SUMO的TraCI服务器
    traci.trafficlight.setProgram(signalcontrol_class.inter_id,signalcontrol_class.initprogramid)  
    traci.trafficlight.setPhase(signalcontrol_class.inter_id,signalcontrol_class.initphaseid)  #
    traci.simulationStep(currentStep)
    currentStep=currentStep+1
    startdecisiont=signalcontrol_class.min_green_durationdicts[signalcontrol_class.initphaseid]
    #print("各相位的最小绿灯时长：{}".format(signalcontrol_class.min_green_durationdicts))
    #print("初始相位：{}，初始决策时刻点：{}".format(signalcontrol_class.initphaseid,startdecisiont))
    decisiontdicts={}
    decisiontdicts[startdecisiont]=signalcontrol_class.initphaseid
    changephasetdicts={}
    
    lastsignal_state=traci.trafficlight.getRedYellowGreenState(signalcontrol_class.inter_id) 
    wait_time_for_phase_list=[0]*len(lastsignal_state)
    while currentStep <=signalcontrol_class.endtime:
        # 执行一个仿真步骤
        traci.simulationStep(currentStep)
        currentsim_t=traci.simulation.getTime()

        wait_time_for_phase_list=get_waittimeforphase(signalcontrol_class.inter_id,wait_time_for_phase_list)

        currentStep=currentStep+1
        if currentsim_t in decisiontdicts:  #进行智能体决策
           # print("进行决策，时刻点：{}".format(currentsim_t))
            # 提取交叉口运行状态
            vehstate=get_vehstate(signalcontrol_class.lanelist,signalcontrol_class.celllength,signalcontrol_class.lanelengthdicts,signalcontrol_class.maxspeed)
            signal_state=get_signal_state(signalcontrol_class.inter_id,wait_time_for_phase_list,signalcontrol_class.max_green_duration_dicts,signalcontrol_class.max_phase_id,signalcontrol_class.min_green_duration,signalcontrol_class.max_green_duration,signalcontrol_class.max_red_wait_t)
            state=vehstate+signal_state
            state=np.array(state)
            #通过智能体决策下一步动作
            action = signalcontrol_class.agent.take_action(state,signalcontrol_class.connectdim,signalcontrol_class.connectid2greenphaseiddicts,signalcontrol_class.phaseid2actiondicts)
            currentgreenphaseid=traci.trafficlight.getPhase(signalcontrol_class.inter_id)    #当前放行相位
            nextgreenphaseid=signalcontrol_class.action2phaseiddicts[action]

            if currentgreenphaseid==nextgreenphaseid: #延长当前绿灯相位一个单位绿灯时长
                #更新决策时间点
                decisiontdicts[currentsim_t+signalcontrol_class.Unit_Extension_Time]=currentgreenphaseid  #下一个决策时间点
                ##执行延长绿灯的操作
                traci.trafficlight.setPhaseDuration(signalcontrol_class.inter_id,signalcontrol_class.Unit_Extension_Time) #当前相位延长一个单位绿灯时间
                
            else:
                #更新决策时间点
                nextdecisiont=currentsim_t+signalcontrol_class.yt+signalcontrol_class.min_green_durationdicts[nextgreenphaseid]  #下一个决策时间点
                decisiontdicts[nextdecisiont]=nextgreenphaseid
                #更新相位切换时刻点
                #执行相位切换的操作
                traci.trafficlight.setPhase(signalcontrol_class.inter_id,currentgreenphaseid+1)  #切换至黄灯相位
               #changephasetdicts[currentsim_t+signalcontrol_class.yt]={"change2phaseid":currentgreenphaseid+2 ,"phasetype":"allr"}   #切换至全红相位的时刻点
                changephasetdicts[currentsim_t+signalcontrol_class.yt]={"change2phaseid":nextgreenphaseid,"phasetype":"G"}    #切换至指定的绿灯相位nextgreenphaseid 的时刻点

            if currentsim_t in changephasetdicts:  #进行相位切换
                traci.trafficlight.setPhase(signalcontrol_class.inter_id,changephasetdicts[currentsim_t]) 
            #print("进行决策，时刻点：{}，决策动作:{},当前的相位：{}，决策后的相位：{}".format(currentsim_t,action,currentgreenphaseid,nextgreenphaseid))
                



    #提取延误数据
    entryexit_result['aveTimeLoss']=[traci.multientryexit.getLastIntervalMeanTimeLoss(signalcontrol_class.entryexitid)]
    entryexit_result['aveTravelTime']=[traci.multientryexit.getLastIntervalMeanTravelTime(signalcontrol_class.entryexitid)]
    entryexit_result['vehsum']=[traci.multientryexit.getLastIntervalVehicleSum(signalcontrol_class.entryexitid)]
    entryexit_result['meanstop']=[traci.multientryexit.getLastIntervalMeanHaltsPerVehicle(signalcontrol_class.entryexitid)]
    pd_entryexit=pd.DataFrame.from_dict(entryexit_result)
    pd_entryexit["seed"]=signalcontrol_class.seed
    pd_entryexit["ecoph"]=signalcontrol_class.currentepoch
    pd_entryexit["type"]="DQN"
    traci.close()
    
    return pd_entryexit
######################  定时控制
def fixedcontrol_simulation(signalcontrol_class):
    #sumocfgxmlfile,seed,currentepoch,endtime,entryexitid,edgelist
    entryexit_result={} 
    currentStep=0.0
    traci.start(['sumo','-c',signalcontrol_class.sumocfgxmlfile,'--seed',str(signalcontrol_class.seed)])  #启用gui界面，连接到SUMO的TraCI服务器
    traci.trafficlight.setProgram(signalcontrol_class.inter_id,signalcontrol_class.initprogramid)  
    traci.trafficlight.setPhase(signalcontrol_class.inter_id,signalcontrol_class.initphaseid)  #
    traci.simulationStep(currentStep)
    currentStep=currentStep+1
    while currentStep <=signalcontrol_class.endtime:
        # 执行一个仿真步骤
        traci.simulationStep(currentStep)
        #当前时刻的排放数据
        # currentsim_t=int(traci.simulation.getTime())
        # print("当前时刻：{},     ".format(currentsim_t))
        currentStep=currentStep+1
        

            

    #提取延误数据
    ##获取平均通行时间、平均车速、平均停车次数、平均延误、通过车辆数 

    entryexit_result['aveTimeLoss']=[traci.multientryexit.getLastIntervalMeanTimeLoss(signalcontrol_class.entryexitid)]
    entryexit_result['aveTravelTime']=[traci.multientryexit.getLastIntervalMeanTravelTime(signalcontrol_class.entryexitid)]
    entryexit_result['vehsum']=[traci.multientryexit.getLastIntervalVehicleSum(signalcontrol_class.entryexitid)]
    entryexit_result['meanstop']=[traci.multientryexit.getLastIntervalMeanHaltsPerVehicle(signalcontrol_class.entryexitid)]

    pd_entryexit=pd.DataFrame.from_dict(entryexit_result)
    pd_entryexit["seed"]=signalcontrol_class.seed
    pd_entryexit["ecoph"]=signalcontrol_class.currentepoch
    pd_entryexit["type"]="fixedtime"
    #print( pd_entryexit)

    traci.close()

    
    return pd_entryexit
######################  感应控制
def actuatedcontrol_simulation(signalcontrol_class):
    entryexit_result={} 
    currentStep=0.0
    traci.start(['sumo','-c',signalcontrol_class.sumocfgxmlfile,'--seed',str(signalcontrol_class.seed)])  #启用gui界面，连接到SUMO的TraCI服务器

    # #仿真的开始时间和结束时间

    # 当前仿真时间
    currentStep = 0
    # 循环直到仿真结束
    # Unit_Extension_Time=3 #单位绿灯延长时间
    # phase_planDurationdicts={0:15,3:15,6:15,9:15}  #初始化为各相位的计划放行绿灯时间  为最小绿
    # phase_maxDurationdicts={0:30,3:25,6:30,9:20}
    #当前相位，持续时间，下一个相位
    # phaseid2detid={0:["n1","n2"],3:["e1","e2"],6:["s1","s2"],9:["w1","w2"]}


    traci.trafficlight.setProgram(signalcontrol_class.inter_id,signalcontrol_class.initprogramid)  
    traci.trafficlight.setPhase(signalcontrol_class.inter_id,signalcontrol_class.initphaseid)  #
    traci.simulationStep(currentStep)
    currentStep=currentStep+1
    phase_planDurationdicts=copy.deepcopy(signalcontrol_class.min_green_durationdicts) #初始化
    while currentStep <= signalcontrol_class.endtime:
        # 执行一个仿真步骤
        traci.simulationStep()
        #获取当前运行相位，已放行的绿灯时间等
        sim_time=traci.simulation.getTime()
        current_phase_id=traci.trafficlight.getPhase(signalcontrol_class.inter_id)  #当前相位id
       
        #print("相位开始时刻：",current_phase_id,currentphase_starttime)
        if current_phase_id in signalcontrol_class.max_green_duration_dicts.keys():
            
            currentphase_SpentDuration=traci.trafficlight.getSpentDuration(signalcontrol_class.inter_id)  #当前相位已放行的绿灯时长   
            
            currentphase_planDuration=phase_planDurationdicts[current_phase_id]
            currentphase_maxgreen=signalcontrol_class.max_green_duration_dicts[current_phase_id]
            if currentphase_SpentDuration>=currentphase_planDuration-3:   #决策是否要延长单位绿灯时间
                #收集车检器的通行数据
                vehsize=sum([traci.inductionloop.getLastStepVehicleNumber(loopid) for loopid in signalcontrol_class.phaseid2detidforactuatecontrol[current_phase_id]])
                if vehsize>0  :#判断该相位上车检器是否有车辆通过，有就延长一个单位绿灯时间
                    traci.trafficlight.setPhaseDuration(signalcontrol_class.inter_id,signalcontrol_class.Unit_Extension_Time) #当前相位延长一个单位绿灯时间
                    phase_planDurationdicts[current_phase_id]=currentphase_SpentDuration+signalcontrol_class.Unit_Extension_Time#更新该相位计划的放行时长
                    currentphase_planDuration=currentphase_SpentDuration+signalcontrol_class.Unit_Extension_Time
                    #print(current_phase_id,sim_time,phase_planDurationdicts[current_phase_id])
            if (currentphase_SpentDuration==currentphase_planDuration)|(currentphase_SpentDuration>=currentphase_maxgreen): #切换至下一个相位
                phase_planDurationdicts=copy.deepcopy(signalcontrol_class.min_green_durationdicts)#更新
                traci.trafficlight.setPhase(signalcontrol_class.inter_id,current_phase_id+1)
            #print(currentStep,signalcontrol_class.min_green_durationdicts)
                #print("相位切换",sim_time,current_phase_id,current_phase_id+1)
      
        # 更新当前仿真时间
        currentStep += 1
    # 仿真结束，关闭与SUMO的连接

    entryexit_result['aveTimeLoss']=[traci.multientryexit.getLastIntervalMeanTimeLoss(signalcontrol_class.entryexitid)]
    entryexit_result['aveTravelTime']=[traci.multientryexit.getLastIntervalMeanTravelTime(signalcontrol_class.entryexitid)]
    entryexit_result['vehsum']=[traci.multientryexit.getLastIntervalVehicleSum(signalcontrol_class.entryexitid)]
    entryexit_result['meanstop']=[traci.multientryexit.getLastIntervalMeanHaltsPerVehicle(signalcontrol_class.entryexitid)]

    pd_entryexit=pd.DataFrame.from_dict(entryexit_result)
    pd_entryexit["seed"]=signalcontrol_class.seed
    pd_entryexit["ecoph"]=signalcontrol_class.currentepoch
    pd_entryexit["type"]="actuate"

    traci.close()
    return pd_entryexit
######################  DQN、定时控制、感应控制效果多轮仿真实验
def  evaluation_DQN(signalcontrolparamter_class):
    #q_networkfilename='q_network_state_dict.pth'
    ####DQN模型的效果评价，与感应控制、定时控制对比
    #########################################加载训练好的DQN模型
    signalcontrolparamter_class=signalcontrolparamter() #初始化模型参数
    q_network =Net(signalcontrolparamter_class.n_states, signalcontrolparamter_class.n_hidden, signalcontrolparamter_class.n_actions)
    q_network.load_state_dict(torch.load(signalcontrolparamter_class.q_network_best_savefilename))
    q_network.eval()  # 确保设置为评估模式filename
    # 实例化DQN
    agent = DQN(n_states=signalcontrolparamter_class.n_states,
                n_hidden=signalcontrolparamter_class.n_hidden,
                n_actions=signalcontrolparamter_class.n_actions,
                learning_rate=signalcontrolparamter_class.lr,
                gamma=signalcontrolparamter_class.gamma,
                epsilon=signalcontrolparamter_class.epsilon,
                target_update=signalcontrolparamter_class.target_update,
                device=signalcontrolparamter_class.device,
            )
    agent.q_net=q_network
    agent.target_q_net=q_network
    signalcontrolparamter_class.agent=agent

    #########################################多轮仿真实验推理计算
    pd_entryexit=pd.DataFrame()

    for i in range(0,signalcontrolparamter_class.simecoph):  #
        seed=str(random.randint(1,10000))
        signalcontrolparamter_class.seed=seed
        signalcontrolparamter_class.currentepoch=i
        pd_entryexit1=DQNinference(signalcontrolparamter_class)
        pd_entryexit2=fixedcontrol_simulation(signalcontrolparamter_class)
        pd_entryexit3=actuatedcontrol_simulation(signalcontrolparamter_class)
   
        pd_entryexit=pd.concat([pd_entryexit,pd_entryexit1,pd_entryexit2,pd_entryexit3])
    

    pd_entryexit.to_csv(signalcontrolparamter_class.compareresult_savefilename)  
    return pd_entryexit
######################  DQN、定时控制、感应控制效果对比绘图（总）
def plot_entryexitresult(pd_entryexit,paralist,ylabellist):  #绘制E3车检器提取的平均通行时间等数据的对比图
    methoddicts={"DQN":"DQN","fixedtime":"定时控制","actuate":"感应控制"}
    mehodcolordicts={"DQN":"r","fixedtime":"b","actuate":"g"}
    # 指定文件夹路径
    path1=os.path.dirname(os.path.realpath(__file__))
    folder_path = path1+"\\"+"entryexit"  #结果数据保存的路径
    # 使用 os.path.exists() 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 使用 os.makedirs() 创建文件夹，mode 可选，表示文件夹权限
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")
    else:
        print(f"Folder already exists at {folder_path}")


    fig=plt.figure(figsize=(32,8))# 初始化图形和坐标轴
    fig.suptitle('不同控制策略下交叉口通行效率对比', fontsize=30)
    axlist=[]
    for i in range(1,len(paralist)+1):
        axlist.append(fig.add_subplot(1, len(paralist), i))

    for para,ylabel,figi in zip(paralist,ylabellist,range(0,len(paralist))):
   
        pdtemp=pd_entryexit.groupby(['type']).apply(lambda x:sum(x[para]*x['vehsum'])/sum(x['vehsum'])).reset_index()
        pdtemp.columns=['type','result']
        methodlist=list(pdtemp['type'])
        mehodlabellist=[methoddicts[method] for method in  methodlist]
        colorlist=[mehodcolordicts[method] for method in  methodlist]
        title=str(ylabel)
        ylist=list(pdtemp['result'])
         #计算下降的比例
        DQNratio=round((ylist[mehodlabellist.index("定时控制")]-ylist[mehodlabellist.index("DQN")])/ylist[mehodlabellist.index("定时控制")]*100,2)
        acutateratio=round((ylist[mehodlabellist.index("感应控制")]-ylist[mehodlabellist.index("DQN")])/ylist[mehodlabellist.index("感应控制")]*100,2)
        #print(ratio)
        x1=[ii*3+3 for ii in range(0,len(mehodlabellist))]
        axlist[figi].bar(x1,ylist,color=colorlist)
        axlist[figi].set_xticks(x1)
        axlist[figi].set_xticklabels(mehodlabellist)
        
        axlist[figi].set_title("DQN相比于定时控制,"+title+"降低了"+str(DQNratio)+"%,"+"\n相比于感应控制降低了"+str(acutateratio)+"%",fontsize=20) #
        
     
        axlist[figi].set_ylabel(ylabel,fontsize=20)
        axlist[figi].set_xlim(0, max(x1)+5)  # 更新x轴的显示范围
        axlist[figi].set_ylim(min(ylist)*0.95, max(ylist)*1.05)  # 更新x轴的显示范围

    plt.subplots_adjust(hspace=0.4,wspace=0.3,top=0.8)
    plt.savefig(folder_path+"\\交叉口通行效率对比图"+'.jpg')
    plt.close()
######################  DQN、定时控制、感应控制效果对比绘图（分）
def plot_entryexitresult_oneparaonefig(pd_entryexit,paralist,ylabellist):  #绘制E3车检器提取的平均通行时间等数据的对比图
    methoddicts={"DQN":"DQN","fixedtime":"定时控制","actuate":"感应控制"}
    mehodcolordicts={"DQN":"r","fixedtime":"b","actuate":"g"}
    # 指定文件夹路径
    path1=os.path.dirname(os.path.realpath(__file__))
    folder_path = path1+"\\"+"entryexit"  #结果数据保存的路径
    # 使用 os.path.exists() 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 使用 os.makedirs() 创建文件夹，mode 可选，表示文件夹权限
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")
    else:
        print(f"Folder already exists at {folder_path}")




    for para,ylabel,figi in zip(paralist,ylabellist,range(0,len(paralist))):
        fig=plt.figure(figsize=(12,8))# 初始化图形和坐标轴
        ax=fig.add_subplot(1,1,1)
        pdtemp=pd_entryexit.groupby(['type']).apply(lambda x:sum(x[para]*x['vehsum'])/sum(x['vehsum'])).reset_index()
        pdtemp.columns=['type','result']
        methodlist=list(pdtemp['type'])
        mehodlabellist=[methoddicts[method] for method in  methodlist]
        colorlist=[mehodcolordicts[method] for method in  methodlist]
        title=str(ylabel)
        ylist=list(pdtemp['result'])
         #计算下降的比例
        DQNratio=round((ylist[mehodlabellist.index("定时控制")]-ylist[mehodlabellist.index("DQN")])/ylist[mehodlabellist.index("定时控制")]*100,2)
        acutateratio=round((ylist[mehodlabellist.index("感应控制")]-ylist[mehodlabellist.index("DQN")])/ylist[mehodlabellist.index("感应控制")]*100,2)
       
        
        #print(ratio)
        x1=[ii*3+3 for ii in range(0,len(mehodlabellist))]
        ax.bar(x1,ylist,color=colorlist)
        ax.set_xticks(x1)
        ax.set_xticklabels(mehodlabellist)
        
        ax.set_title("DQN相比于定时控制,"+title+"降低了"+str(DQNratio)+"%,"+"\n相比于感应控制降低了"+str(acutateratio)+"%",fontsize=20) #
        
        ax.set_ylabel(ylabel,fontsize=20)
        ax.set_xlim(0, max(x1)+5)  # 更新x轴的显示范围
        ax.set_ylim(min(ylist)*0.95, max(ylist)*1.05)  # 更新x轴的显示范围
        plt.savefig(folder_path+"\\对比图_"+str(para)+'.jpg')
        plt.close()
######################  仿真参数的初始化配置
class signalcontrolparamter():#DQN参数配置
    def __init__(self):
        """
        初始化函数，用于设置仿真环境的各种参数和配置。

        该函数主要完成以下任务：
        1. 设置仿真文件的路径和配置。
        2. 初始化仿真环境中的交叉口、车道、信号灯等参数。
        3. 启动SUMO仿真并获取车道长度信息。
        4. 设置信号灯相位、绿灯时长、黄灯时长等交通信号控制相关参数。
        5. 初始化深度学习模型的相关参数，如经验池、学习率、折扣因子等。
        6. 设置训练结果的保存路径和模型效果评价的相关参数。
        """

        # 获取当前文件的路径，并设置SUMO配置文件的路径
        self.path1 = os.path.dirname(os.path.realpath(__file__))
        self.sumocfgxmlfile = self.path1 + "\\model\\longhua.sumocfg"

        # 设置仿真种子和仿真结束时间
        self.seed = 1000
        self.endtime = 3600

        # 设置交叉口ID和车道列表
        self.inter_id = ("J0")  # 交叉口id
        self.lanelist = ["-E0_0", "-E0_1", "-E0_2", "-E0_3",
                         "-E3_0", "-E3_1", "-E3_2", "-E3_3", "-E3_4", "-E3_5",
                         "-E6_0", "-E6_1", "-E6_2", "-E6_3",
                         "-E9_0", "-E9_1", "-E9_2", "-E9_3", "-E9_4",
                         "-E12_0", "-E12_1", "-E12_2", "-E12_3"]
        self.celllength = 6
        self.lanelengthdicts = {}

        # 启动SUMO仿真并获取车道长度信息
        traci.start(['sumo', '-c', self.sumocfgxmlfile, '--seed', str(self.seed)])  # 启用gui界面，连接到SUMO的TraCI服务器
        for laneid in self.lanelist:
            self.lanelengthdicts[laneid] = traci.lane.getLength(laneid)
        traci.close()

        # 设置仿真预热周期数
        self.warmupcycle = 2  # 仿真预热 前多少个周期进行仿真预热

        # 设置相位与动作的映射关系
        self.action2phaseiddicts = {0: 0, 1: 2, 2: 4, 3: 5, 4: 7,5:9}
        self.phaseid2actiondicts = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4,9:5}

        # 设置当前相位到下一个绿灯相位的映射关系
        self.current_phase_id2nextgreenphaseiddicts = {0:2,1:2,2:4,3:4,4:5,5:7,6:7,7:9,8:9,9:0,10:0}  # 当前相位2下一个绿灯相位

        # 设置黄灯时长和全红时长
        self.yt = 3  # 黄灯时长##
       #self.allr = 2  # 全红
        # 设置最小和最大绿灯时长
        self.min_green_durationdicts = {0: 15, 2:10, 4: 5, 5: 15, 7: 15, 9: 20}  # 最小绿灯时长
        self.max_green_duration_dicts = {0: 25, 2:20, 4: 10, 5: 30, 7:30,9:30}

        # 设置仿真开始的信号配时方案和相位
        self.initprogramid = "1"  # 仿真开始放行的信号配时方案
        self.initphaseid =0# 仿真开始放行的相位

        # 设置相位与检测器的映射关系
        self.phaseid2detidforactuatecontrol = {0: ["e1_30", "e1_37", "e1_39","e1_29"], 2: [ "e1_38", "e1_28"], 4: ["e1_23", "e1_34"], 5: ["e1_23", "e1_34", "e1_27","e1_26"],7: ["e1_24", "e1_25", "e1_35","e1_36"],9: ["e1_31", "e1_32", "e1_33"]}

        # 计算信号周期长度
        self.cyclelength = sum(list(self.min_green_durationdicts.values()))

        # 设置连接维度与绿灯相位的映射关系
        self.connectdim = 29
        self.connectid2greenphaseiddicts = {1:0,2:0 ,3:0, 4:2, 6:5, 7:5, 8:7, 9:7,10:4,10:5,
                                            12:0, 13:0, 14:0, 15:2, 16:2,
                                            18:7, 19:7, 20:4,20:5,21:4,21:5,
                                            24:9, 25:9, 26:9, 27 :9,28:9,
                                            0: 0, 0: 2, 0: 4, 0: 5, 0: 7, 0: 9,
                                            5: 0, 5: 2, 5: 4, 5: 5, 5: 7, 5: 9,
                                            17: 0, 17: 2, 17: 4, 17: 5, 17: 7, 17: 9,
                                           22: 0, 22: 2, 22: 4, 22: 5, 22: 7, 22: 9,23: 0, 23: 2, 23: 4, 23: 5, 23: 7, 23:9}
        # 输入数据归一化需要用到的参数
        self.max_phase_id = 10
        self.min_green_duration = 2
        self.max_green_duration = 30

        # 设置最大速度、停车速度阈值、最大等待时间等交通参数
        self.maxspeed = 27
        self.stopspeed = 1  # 判断为停车的速度阈值
        self.max_red_wait_t = 150  # 机动车最大等待时间
        self.Saturation_headway = 2  # 饱和车头时距
        self.Unit_Extension_Time = 3  # 单位绿灯延长时长

        # 设置E1检测器列表和评价效果检测器
        self.E1detlist = ["e1_0", "e1_1", "e1_2", "e1_3",
                          "e1_4", "e1_5", "e1_6", "e1_7", "e1_8", "e1_9",
                          "e1_10", "e1_11", "e1_12", "e1_13",
                          "e1_14", "e1_15", "e1_16", "e1_17", "e1_18",
                          "e1_19", "e1_20", "e1_21", "e1_22"]  # 线圈E1车检器用于检测交叉口通过车辆数
        self.entryexitid = "e3_0"  # 收集评价效果的E3车检器

        # 设置设备类型（GPU或CPU）
        self.device = torch.device("cuda")  # if torch.cuda.is_available()  else torch.device("cpu")

        # 设置经验池容量、学习率、折扣因子、贪心系数等深度学习参数
        self.capacity = 100000  # 经验池容量
        self.lr = 0.1  # 学习率
        self.gamma = 0.7  # 折扣因子
        self.epsilon = 0.90  # 贪心系数
        self.target_update = 40  # 目标网络的参数的更新频率
        self.batch_size = 256
        self.n_hidden = 1024  # 隐含层神经元个数
        self.min_size = 300  # 经验池超过200后再训练
        self.trainfreq = 1  # 经验池超过min_size,每10次之后再训练

        # 设置超过红灯等待时间和最大绿灯时长的惩罚因子
        self.overwait_factor = 5  # 超过红灯等待时间的惩罚因子
        self.overgreen_factor = 2  # 超过最大绿的惩罚因子

        # 设置状态空间和动作空间的维度
        self.n_states = 1023  # 42位向量表示状态
        self.n_actions = 5  # 10个动作值

        # 设置训练回合数和当前回合数
        self.trainepoch = 500  # 训练的回合数
        self.currentepoch = 0  # 当前回合

        # 实例化经验池
        self.fixedcontrolepoch2fillreplay_bufferminsize = 10  # 将经验池按定时控制的方式，进行填充的轮数   加速模型的训练收敛
        self.replay_buffer = ReplayBuffer(self.capacity)

        # 实例化DQN模型
        self.agent = DQN(n_states=self.n_states,
                         n_hidden=self.n_hidden,
                         n_actions=self.n_actions,
                         learning_rate=self.lr,
                         gamma=self.gamma,
                         epsilon=self.epsilon,
                         target_update=self.target_update,
                         device=self.device,
                         )

        # 设置训练结果的保存路径
        self.q_network_savefilename = self.path1 + '\\q_network_state_dict.pth'
        self.q_network_best_savefilename = self.path1 + '\\q_network_state_dict_best.pth'
        self.train_loss_savefilename = self.path1 + '\\pdloss.csv'
        self.deque_savefilename = self.path1 + '\\deque.pkl'
        self.episodesreward_savefilename = self.path1 + '\\pdepisodesreward.csv'
        self.rewardfigname = self.path1 + "\\各回合累积回报曲线_0_500.jpg"
        self.lossfigname = self.path1 + "\\各迭代次数下loss曲线_0_500.jpg"

        # 设置模型效果评价的对比轮数和结果保存路径
        self.simecoph = 20
        self.compareresult_savefilename = self.path1 + "\\pd_entryexit.csv"


######################  主程序  ######################  
signalcontrolparamter_class=signalcontrolparamter() #初始化模型参数
fixedcontrol_fillreplay_buffer2minsize(signalcontrolparamter_class) #先按定时控制填充经验池，减少不好的训练样本
DQN_train(signalcontrolparamter_class)  #模型训练
pd_entryexit=evaluation_DQN(signalcontrolparamter_class) #模型效果对比
paralist=['aveTimeLoss','aveTravelTime','meanstop','vehsum']
ylabellist=["延误(s)","平均通行时间(s)","平均停车次数","平均通过车辆数"]
pd_entryexit=pd.read_csv(signalcontrolparamter_class.compareresult_savefilename)  
plot_entryexitresult(pd_entryexit,paralist,ylabellist)  #模型效果对比图
plot_entryexitresult_oneparaonefig(pd_entryexit,paralist,ylabellist) #模型效果对比图



