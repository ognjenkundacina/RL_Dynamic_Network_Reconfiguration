from collections import namedtuple
from itertools import count
import random
import matplotlib.pyplot as plt
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from config import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        #print(input_size)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc3_bn = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        #self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(512, output_size)
        #print(output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        return self.fc6(x)


class DeepQLearningAgent:

    def __init__(self, environment):
        self.environment = environment
        self.epsilon = 0.2
        self.batch_size = 128
        self.gamma = 0.99
        self.target_update = 10
        self.memory = ReplayMemory(1000000)

        self.state_space_dims = environment.state_space_dims
        self.n_actions = environment.n_actions
        self.loss_list = []

        self.policy_net = DQN(self.state_space_dims, self.n_actions)
        self.target_net = DQN(self.state_space_dims, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.train() #train mode (train vs eval mode)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00001) #todo pokusaj nesto drugo
        #self.optimizer = optim.RMSprop(self.policy_net.parameters())

    #return capacitor index: 0..n_cap - 1
    def get_action(self, state, epsilon):
        #print('Available actions:')
        #print(self.environment.available_actions)
        if random.random() > epsilon:
            self.policy_net.eval()
            with torch.no_grad():
                #self.policy_net(state).sort(-1, descending = True)[1] daje indekse akcija sortiranih po njihovim q vrijednostima
                sorted_actions = self.policy_net(state).sort(-1, descending = True)[1].tolist()[0]
                #print('Sorted actions:')
                #print (sorted_actions)
                for action_candidate in sorted_actions:
                    if (len(self.environment.available_actions) == 0):
                        print('Agent -> get_action: No avaliable actions')
                    if action_candidate in self.environment.available_actions.keys():
                        action = action_candidate
                        break
                #action = self.policy_net(state).max(1)[1].view(1, 1)
                self.policy_net.train()
                #print('Best action: ', action)
                return action
        else:
            action = random.choice(list(self.environment.available_actions.keys()))
            #print('Random action: ', action)
            return action

    def train(self, df_train, n_episodes, df_test):
        #self.policy_net.load_state_dict(torch.load("policy_net"))
        f_loss = open("loss_function.txt", "w")
        f_ter = open("total_episode_reward.txt", "w")
        f_mar = open("moving_average_reward.txt", "w")

        self.epsilon = 0.99
        self.reward_moving_average = 0

        a = 0.99
        b = 0.1
        n_end = (int)(0.8 * n_episodes)
        k = 1.234375E-10
        l = 0.00001975
        delta = (a - b) / n_end
        #print(delta)
        
        #total_episode_rewards1 = []
        #total_episode_rewards2 = []

        ter_list = []
        mar_list = []
        loss_list = []
        for i_episode in range(n_episodes):
            if (i_episode % 100 == 0):
                print("=========Episode: ", i_episode)

            #if i_episode == 500:
                #self.epsilon = 0.2

            #if (i_episode < n_end):
                #self.epsilon = k * (i_episode * i_episode) - l * i_episode + a
            
            #if (i_episode == n_end):
                #self.epsilon = 0.1

            if (i_episode < n_end):
                self.epsilon -= delta

            if (i_episode == n_end):
                self.epsilon = 0.1

            #if (i_episode == 47000):
                #self.epsilon = 0

            #if (i_episode == n_end):
                #self.epsilon = 0.01

            if (i_episode % 2500 == 2499):
                time.sleep(60)

            done = False
            df_row = df_train.sample(n=1)
            row_list = df_row.values.tolist()
            row_list = row_list[0]

            #daily_consumption_percents_per_feeder ima 72 clana. Za svaki od 24 trenutka idu 3 scaling faktora, za svaki od feedera
            #i to prva tri clana liste odgovaraju prvom trenutku, pa sljedeca tri drugom...
            daily_consumption_percents_per_feeder = row_list[1 : 3*NUM_TIMESTEPS + 1]

            #ispod za veliku semu
            #daily_consumption_percents_per_feeder = row_list[1 : 4*NUM_TIMESTEPS + 1]
            
            #x = random.choice((-1, 1))
            #96 zbog 4x24
            #for zz in range(72):
                #daily_consumption_percents_per_feeder[zz] += (-0.6) * random.random() + 0.3  #dodaje random broj u opsegu [-0.15, 0.15]
                #if daily_consumption_percents_per_feeder[zz] < 0.0:
                    #daily_consumption_percents_per_feeder[zz] = 0

            state = self.environment.reset(daily_consumption_percents_per_feeder)
            #print ('Initial losses: ', self.environment.power_flow.get_losses())

            state = torch.tensor([state], dtype=torch.float)
            total_episode_reward = 0

            while not done:
                action = self.get_action(state, epsilon = self.epsilon)
                #print('action', action)
                if (action > self.n_actions):
                    print("agent.train: action > self.n_actions")
                next_state, reward, done = self.environment.step(action)
                #print ('Current losses: ', self.environment.power_flow.get_losses())
                total_episode_reward += reward

                reward = torch.tensor([reward], dtype=torch.float)
                action = torch.tensor([action], dtype=torch.float)
                next_state = torch.tensor([next_state], dtype=torch.float)
                if done:
                    next_state = None

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.optimize_model()
            
            if (i_episode == 0):
                self.reward_moving_average = -1.2
            else:
                self.reward_moving_average = 0.99 * self.reward_moving_average + 0.01 * total_episode_reward
            #total_episode_rewards.append(total_episode_reward)
            ter_list.append(total_episode_reward)
            mar_list.append(self.reward_moving_average)
            #total_episode_rewards1.append(total_episode_reward)
            #total_episode_rewards2.append(self.reward_moving_average)

            #print ("self.epsilon: ", self.epsilon)
            if (i_episode % 100 == 0):
                print ("total_episode_reward: ", total_episode_reward)

            #if (i_episode % 1000 == 999):
                #torch.save(self.policy_net.state_dict(), "policy_net")

            if (i_episode % 10 == 0):
                torch.save(self.policy_net.state_dict(), "policy folder/policy_net" + str(i_episode))

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if (i_episode % 10 == 0 and i_episode != 0):
                self.test(df_test, i_episode, n_episodes)

            #print("=====================================")

        torch.save(self.policy_net.state_dict(), "policy_net")

        #total_episode_reward
        for i in range (len(ter_list)):
            f_ter.write(str(ter_list[i]) + "\n")
        f_ter.close()

        #moving_average_reward
        for i in range (len(mar_list)):
            f_mar.write(str(mar_list[i]) + "\n")
        f_mar.close()

        #ter = []
        #with open('total_episode_reward.txt') as f_ter:
            #for line in f_ter:
                #elems = line.strip()
                #ter.append(float(elems))

        #mar = []
        #with open('moving_average_reward.txt') as f_mar:
            #for line in f_mar:
                #elems = line.strip()
                #mar.append(float(elems))

        #x_axis = [1 + j for j in range(len(ter))]
        #plt.plot(x_axis, ter, color="lightblue")
        #plt.plot(x_axis, mar, color="blue")
        #plt.xlabel('Episode number') 
        #plt.ylabel('Total reward') 
        #plt.savefig("total_rewards.png")
        #plt.show()

        #loss
        for i in range (len(self.loss_list)):
            f_loss.write(str(self.loss_list[i]) + "\n")
        f_loss.close()

        #loss = []
        #with open('loss_function.txt') as fr:
            #for line in fr:
                #elems = line.strip()
                #loss.append(float(elems))

        #x_axis_loss_txt = [1 + j for j in range(len(loss))]
        #plt.plot(x_axis_loss_txt, loss, color="red")
        #plt.xlabel('Iteration') 
        #plt.ylabel('DQN Loss') 
        #plt.savefig("loss_txt.png")
        #plt.show()


    def test(self, df_test, i_episode, n_episodes):
        f1 = open("total_episode_reward_new.txt", "a")
        f2 = open("moving_average_reward_new.txt", "a")
        total_episode_reward_list = []
        #moving_average_reward_list = []

        if (i_episode != n_episodes):
            self.policy_net.load_state_dict(torch.load("policy folder/policy_net" + str(i_episode)))
        else:
            self.policy_net.load_state_dict(torch.load("policy_net"))
        self.policy_net.eval()

        for index, row in df_test.iterrows():
            row_list = row.values.tolist()
            #row_list = row_list[0] nije potrebno jer row nije stra struktura kao df frame u train metodi

            #daily_consumption_percents_per_feeder ima 72 clana. Za svaki od 24 trenutka idu 3 scaling faktora, za svaki od feedera
            #i to prva tri clana liste odgovaraju prvom trenutku, pa sljedeca tri drugom...
            daily_consumption_percents_per_feeder = row_list[1 : 3*NUM_TIMESTEPS + 1]

            #ispod za veliku semu
            #daily_consumption_percents_per_feeder = row_list[1 : 4*NUM_TIMESTEPS + 1]

            state = self.environment.reset(daily_consumption_percents_per_feeder)
            #print ('Initial losses: ', self.environment.power_flow.get_losses())

            state = torch.tensor([state], dtype=torch.float)
            done = False
            total_episode_reward = 0

            while not done:
                action = self.get_action(state, epsilon = 0.0)
                #print("Open switches: ", radial_switch_combinations[action]) 
                
                if (action > self.n_actions):
                    print ("Warning: agent.test: action > self.n_actions")
                
                next_state, reward, done = self.environment.step(action)
                #print("Open switches: ", radial_switch_combinations[action])
                #print("Open switches: ", radial_switch_combinations_reduced_big_scheme[action])
                if(i_episode == n_episodes):
                    print("Open switches: ", radial_switch_combinations_ieee33[action])
                    print ('Current losses: ', self.environment.power_flow.get_losses())
                    
                total_episode_reward += reward
                state = torch.tensor([next_state], dtype=torch.float)

            self.reward_moving_average = 0.9 * self.reward_moving_average + 0.1 * total_episode_reward

            total_episode_reward_list.append(total_episode_reward)
            #moving_average_reward_list.append(self.reward_moving_average)

        f1.write(str(total_episode_reward) + "\n")
        print(total_episode_reward)
        f1.close()

        f2.write(str(self.reward_moving_average) + "\n")
        f2.close()

        if(i_episode == n_episodes):
            print ("Test set reward ", sum(total_episode_reward_list))


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        #converts batch array of transitions to transiton of batch arrays
        batch = Transition(*zip(*transitions))

        #compute a mask of non final states and concatenate the batch elements
        #there will be zero q values for final states later... therefore we need mask
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(-1,1) #reshape into many rows, one column
        reward_batch = torch.cat(batch.reward).view(-1,1)

        # compute Q(s_t, a) - the model computes Q(s_t), then we select
        # the columns of actions taken. These are the actions which would've
        # been taken for each batch state according to policy net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.long())

        #gather radi isto sto i:
        #q_vals = []
        #for qv, ac in zip(Q(obs_batch), act_batch):
        #    q_vals.append(qv[ac])
        #q_vals = torch.cat(q_vals, dim=0)

        # Compute V(s_{t+1}) for all next states
        # q values of actions for non terminal states are computed using
        # the older target network, selecting the best reward with max
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() #nema final stanja
        #za stanja koja su final ce next_state_values biti 0
        #detach znaci da se nad varijablom next_state_values ne vrsi optimizacicja
        next_state_values = next_state_values.view(-1,1)
        # compute the expected Q values
        expected_state_action_values = (next_state_values*self.gamma) + reward_batch

        #Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        loss_ = loss.detach().numpy()
        self.loss_list.append(loss_)

        self.optimizer.zero_grad()
        loss.backward()

        #todo razmisli kasnije o ovome
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()