from collections import namedtuple
from itertools import count
import random
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

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
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc3_bn = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        return self.fc4(x)


class DeepQLearningAgent:

    def __init__(self, environment):
        self.environment = environment
        self.epsilon = 0.2
        self.batch_size = 128
        self.gamma = 1.0
        self.target_update = 10
        self.memory = ReplayMemory(1000000)

        self.state_space_dims = environment.state_space_dims
        self.n_actions = environment.n_actions

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

    def train(self, df_train, n_episodes):
        
        total_episode_rewards = []
        for i_episode in range(n_episodes):
            if (i_episode % 1000 == 0):
                print("=========Episode: ", i_episode)

            #if (i_episode == int(0.1 * n_episodes)):
                #self.epsilon = 0.3
            #if (i_episode == int(0.5 * n_episodes)):
                #self.epsilon = 0.1

            # if (i_episode % 1000 == 999):
            #     time.sleep(60)

            done = False
            df_row = df_train.sample(n=1)
            row_list = df_row.values.tolist()
            row_list = row_list[0]

            consumption_percents = row_list[1:self.environment.n_consumers + 1]
            capacitor_statuses = row_list[self.environment.n_consumers + 1:]

            state = self.environment.reset(consumption_percents, capacitor_statuses)

            state = torch.tensor([state], dtype=torch.float)
            total_episode_reward = 0 

            while not done:
                action = self.get_action(state, epsilon = self.epsilon)
                #print("Toogle capacitor: ", action + 1)    
                if (action > self.n_actions - 1):
                    print("agent.train: action > self.n_actions - 1")
                next_state, reward, done = self.environment.step(action)
                total_episode_reward += reward

                reward = torch.tensor([reward], dtype=torch.float)
                action = torch.tensor([action], dtype=torch.float)
                next_state = torch.tensor([next_state], dtype=torch.float)
                if done:
                    next_state = None

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.optimize_model()
            
            total_episode_rewards.append(total_episode_reward)

            #if (i_episode % 10 == 0):
                #print ("total_episode_reward: ", total_episode_reward)

            if (i_episode % 100 == 0):
                torch.save(self.policy_net.state_dict(), "policy_net")

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            #print("=====================================")

        torch.save(self.policy_net.state_dict(), "policy_net")

        x_axis = [1 + j for j in range(len(total_episode_rewards))]
        plt.plot(x_axis, total_episode_rewards)
        plt.xlabel('Episode number') 
        plt.ylabel('Total episode reward') 
        plt.savefig("total_episode_rewards.png")
        #plt.show()


    def test(self, df_test):
        total_episode_reward_list = [] 

        self.policy_net.load_state_dict(torch.load("policy_net"))
        self.policy_net.eval()

        for index, row in df_test.iterrows():
            row_list = row.values.tolist()
            #row_list = row_list[0] nije potrebno jer row nije stra struktura kao df frame u train metodi

            consumption_percents = row_list[1:self.environment.n_consumers + 1]
            capacitor_statuses = row_list[self.environment.n_consumers + 1:]

            state = self.environment.reset(consumption_percents, capacitor_statuses)
            print ('Initial losses: ', self.environment.power_flow.get_losses())

            state = torch.tensor([state], dtype=torch.float)
            done = False
            total_episode_reward = 0

            while not done:
                action = self.get_action(state, epsilon = 0.0)
                print("Toogle capacitor: ", self.environment.capacitor_names_by_index[action]) 
                if (action > self.n_actions - 1):
                    print ("agent.test: action > self.n_actions - 1")
                
                next_state, reward, done = self.environment.step(action)

                if done: #posljednja akcija nije donosila benefit, pa je ukidamo
                    print ('Last action was reverted')
                    next_state = self.environment.revert_action(action)
                    reward = 0
                    done = False
                    
                if (self.environment.i_step == self.environment.n_actions):
                    print('LOSSES: ', self.environment.current_losses)
                    break
                    
                total_episode_reward += reward
                state = torch.tensor([next_state], dtype=torch.float)

            total_episode_reward_list.append(total_episode_reward)

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

        self.optimizer.zero_grad()
        loss.backward()

        #todo razmisli kasnije o ovome
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()