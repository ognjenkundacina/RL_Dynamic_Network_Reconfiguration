import gym
from gym import spaces
import random
import numpy as np
import matplotlib.pyplot as plt
from power_algorithms.odss_power_flow import ODSSPowerFlow
from power_algorithms.odss_network_management import ODSSNetworkManagement
import power_algorithms.odss_network_management as nm
from config import *
import copy
import json
import matplotlib._color_data as mcd

class Environment(gym.Env):
    
    def __init__(self):
        super(Environment, self).__init__()
        
        self.state = []

        self.network_manager = nm.ODSSNetworkManagement()
        self.power_flow = ODSSPowerFlow()
        self.power_flow.calculate_power_flow() #potrebno zbog odredjivanja state_space_dims

        self.state_space_dims = len(self.power_flow.get_switches_apparent_power()) + 1

        self.radial_switch_combinations = radial_switch_combinations
        #ipak je u config-u, ako hoces izmeni
        #self.radial_switch_combinations = radial_switch_combinations_reduced_big_scheme

        self.n_actions = len(self.radial_switch_combinations)
        self.n_consumers = self.network_manager.get_load_count()
        self.timestep = 0
        self.switching_action_cost = 1.0
        self.base_power = 4000
        self.previous_action = 0
        #self.switching_operation_constraint = 2
        #self.allow_changing_action = False

        #self.used_switches = []
        #self.used_switches = [0 for i in range (14)]

        #ima ih 31, ali im indeksi ne idu od 1 do 31, vec kao sto stoji malo ispod u self.switch_indices
        #self.used_switches = [0 for i in range (1015)] ovo nam ne treba kad nemamo Nsw ogranicenje

        self.switch_names = self.network_manager.get_all_switch_names()
        self.n_switches = len(self.network_manager.get_all_switch_names())
        # indeks prekidaca, pri cemo indeksiranje pocinje od 1
        self.switch_indices = [i for i in range(1, self.n_switches + 1)]

        #ispod za veliku semu
        #self.switch_indices = [1, 50, 97, 144, 191, 238, 253, 302, 349, 396, 443, 490, 505, 554, 601, 648, 695, 742, 757, 806, 853, 900, 947, 994, 1009, 1010, 1011, 1012, 1013, 1014, 1015]

        self.switch_names_by_index = dict(zip(self.switch_indices, self.switch_names))
        
    def _update_state(self, action):
        
        self._update_switch_statuses(action)
        #self._update_switch_statuses_big_scheme(action)

        #self._update_available_actions(action) #todo implement - ovo nam ne treba kad nemamo ogranicenje Nsw
        self.power_flow.calculate_power_flow()

        self.state = []
        switch_s_dict = self.power_flow.get_switches_apparent_power()
        self.state += [val / self.base_power for val in list(switch_s_dict.values())]
        self.state.append(self.timestep / NUM_TIMESTEPS * 1.0)

        if (len(self.state) != self.state_space_dims):
            print('Environment: len(self.state) != self.state_space_dims')

        return self.state


    def _update_available_actions(self, action):
        current_open_switches = self.radial_switch_combinations[action] #ova akcija je vec odradjena
        #print("current open switches", current_open_switches)
        #print("used switches list: ", self.used_switches)
        #zabranjujemo akcije za koje ce se prekoraciti broj switcheva
        #self.switching_operation_constraint

        remove_from_available_actions_list = []
        #remove_from_available_actions_list_both_constraints = []
        #print("remove_from_available_actions_list", remove_from_available_actions_list)
        for potential_action in self.available_actions.keys():
            for switch_index in self.switch_indices:
                if switch_index in self.radial_switch_combinations[potential_action]:
                    if (switch_index != current_open_switches[0] and switch_index != current_open_switches[1] and switch_index != current_open_switches[2]):
                        #print("checking switch number ", switch_index)
                        #print("used: ", self.used_switches[switch_index-1])
                        if self.used_switches[switch_index-1] == self.switching_operation_constraint: #vec je na ogranicenju, ne zelimo da prekoracimo
                            if not potential_action in remove_from_available_actions_list:
                                remove_from_available_actions_list.append(potential_action)
                                #remove_from_available_actions_list_both_constraints.append(potential_action)
                                #print("remove_from_available_actions_list", remove_from_available_actions_list)
                else:
                    if (switch_index == current_open_switches[0] or switch_index == current_open_switches[1] or switch_index == current_open_switches[2]):
                        #print("checking switch number ", switch_index)
                        #print("used: ", self.used_switches[switch_index-1])
                        if self.used_switches[switch_index-1] == self.switching_operation_constraint:
                            if not potential_action in remove_from_available_actions_list:
                                remove_from_available_actions_list.append(potential_action)
                                #remove_from_available_actions_list_both_constraints.append(potential_action)
                                #print("remove_from_available_actions_list", remove_from_available_actions_list)



        #print("remove_from_available_actions_list", remove_from_available_actions_list)
        #print("used switches list: ", self.used_switches)
        #print("----------------------------------------------------------------\n\n")

        #print("remove_from_available_actions_list", remove_from_available_actions_list)
        #print("========================================================================\n\n")

        #print(action)
        #print('self.used_switches: ',self.used_switches)
        #print('remove_from_available_actions_list: ',remove_from_available_actions_list)
        for action_key in remove_from_available_actions_list:
            self.available_actions.pop(action_key)
            #print(self.available_actions)
        #print('self.available_actions: ',self.available_actions)
        #print('====================================================================================================')

    def _update_available_actions_big_scheme(self, action):
        current_open_switches = self.radial_switch_combinations[action] #ova akcija je vec odradjena
        #print("current open switches", current_open_switches)
        #print("used switches list: ", self.used_switches)
        #zabranjujemo akcije za koje ce se prekoraciti broj switcheva
        #self.switching_operation_constraint

        remove_from_available_actions_list = []
        #remove_from_available_actions_list_both_constraints = []
        #print("remove_from_available_actions_list", remove_from_available_actions_list)
        for potential_action in self.available_actions.keys():
            for switch_index in self.switch_indices:
                if switch_index in self.radial_switch_combinations[potential_action]:
                    if (switch_index != current_open_switches[0] and switch_index != current_open_switches[1] and switch_index != current_open_switches[2] and switch_index != current_open_switches[3] and switch_index != current_open_switches[4] and switch_index != current_open_switches[5] and switch_index != current_open_switches[6]):
                        #print("checking switch number ", switch_index)
                        #print("used: ", self.used_switches[switch_index-1])
                        if self.used_switches[switch_index-1] == self.switching_operation_constraint: #vec je na ogranicenju, ne zelimo da prekoracimo
                            if not potential_action in remove_from_available_actions_list:
                                remove_from_available_actions_list.append(potential_action)
                                #remove_from_available_actions_list_both_constraints.append(potential_action)
                                #print("remove_from_available_actions_list", remove_from_available_actions_list)
                else:
                    if (switch_index == current_open_switches[0] or switch_index == current_open_switches[1] or switch_index == current_open_switches[2] or switch_index == current_open_switches[3] or switch_index == current_open_switches[4] or switch_index == current_open_switches[5] or switch_index == current_open_switches[6]):
                        #print("checking switch number ", switch_index)
                        #print("used: ", self.used_switches[switch_index-1])
                        if self.used_switches[switch_index-1] == self.switching_operation_constraint:
                            if not potential_action in remove_from_available_actions_list:
                                remove_from_available_actions_list.append(potential_action)
                                #remove_from_available_actions_list_both_constraints.append(potential_action)
                                #print("remove_from_available_actions_list", remove_from_available_actions_list)



        #print("remove_from_available_actions_list", remove_from_available_actions_list)
        #print("used switches list: ", self.used_switches)
        #print("----------------------------------------------------------------\n\n")

        #print("remove_from_available_actions_list", remove_from_available_actions_list)
        #print("========================================================================\n\n")

        #print(action)
        #print('self.used_switches: ',self.used_switches)
        #print('remove_from_available_actions_list: ',remove_from_available_actions_list)
        for action_key in remove_from_available_actions_list:
            self.available_actions.pop(action_key)
            #print(self.available_actions)
        #print('self.available_actions: ',self.available_actions)
        #print('====================================================================================================')


    def _update_switch_statuses(self, action): 

        #prev_action = self.radial_switch_combinations[self.previous_action]
        for switch_index in self.switch_indices:
            if switch_index in self.radial_switch_combinations[action]:
                self.network_manager.open_switch(self.switch_names_by_index[switch_index])
                #if (switch_index != prev_action[0] and switch_index != prev_action[1] and switch_index != prev_action[2]):
                    #self.used_switches[switch_index - 1] += 1
            else:
                self.network_manager.close_switch(self.switch_names_by_index[switch_index])
                #if (switch_index == prev_action[0] or switch_index == prev_action[1] or switch_index == prev_action[2]):
                    #self.used_switches[switch_index - 1] += 1
        
        #print(prev_action)

    def _update_switch_statuses_big_scheme(self, action): 

        #prev_action = self.radial_switch_combinations[self.previous_action]
        for switch_index in self.switch_indices:
            if switch_index in self.radial_switch_combinations[action]:
                self.network_manager.open_switch(self.switch_names_by_index[switch_index])
                #if (switch_index != prev_action[0] and switch_index != prev_action[1] and switch_index != prev_action[2] and switch_index != prev_action[3] and switch_index != prev_action[4] and switch_index != prev_action[5] and switch_index != prev_action[6]):
                    #self.used_switches[switch_index - 1] += 1
            else:
                self.network_manager.close_switch(self.switch_names_by_index[switch_index])
                #if (switch_index == prev_action[0] or switch_index == prev_action[1] or switch_index == prev_action[2] or switch_index == prev_action[3] or switch_index == prev_action[4] or switch_index == prev_action[5] or switch_index == prev_action[6]):
                    #self.used_switches[switch_index - 1] += 1
        
        #print(prev_action)
    #action: 0..n_actions
    def step(self, action):
        self.timestep += 1
        #self.switch_operations_by_index[toogled_switch_index] += 1
        #if (self.timestep > 1 and self.allow_changing_action == False):
            #action = self.previous_action
            #self.allow_changing_action = True
            
        #elif (self.allow_changing_action == True and action != self.previous_action):
            #self.allow_changing_action = False

        next_state = self._update_state(action)

        reward = self.calculate_reward(action)

        done = (self.timestep == NUM_TIMESTEPS)

        self.set_load_scaling_for_timestep()

        self.previous_action = action

        #if (self.timestep == 24):
            #self.allow_changing_action = False

        return next_state, reward, done

    def calculate_reward(self, action):
        reward = 0
        #self.power_flow.get_losses() daje gubitke u kW, pa odmah imamo i kWh
        reward -= self.power_flow.get_losses() * 0.065625
 
        reward -= self.switching_action_cost * self.get_number_of_switch_manipulations(self.radial_switch_combinations[self.previous_action], self.radial_switch_combinations[action])

        #ispod za veliku semu
        #reward -= self.switching_action_cost * self.get_number_of_switch_manipulations_big_scheme(self.radial_switch_combinations[self.previous_action], self.radial_switch_combinations[action])

        #zbog numerickih pogodnost je potrebno skalirati nagradu tako da moduo total episode reward bude oko 1.0
        reward /= 1000.0
        return reward

    def get_number_of_switch_manipulations(self, previous_action, action):
        num_of_switch_manipulations = 6

        if (previous_action[0] == action[0] or previous_action[0] == action[1] or previous_action[0] == action[2]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[1] == action[0] or previous_action[1] == action[1] or previous_action[1] == action[2]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[2] == action[0] or previous_action[2] == action[1] or previous_action[2] == action[2]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2
        
        return num_of_switch_manipulations


    def get_number_of_switch_manipulations_big_scheme(self, previous_action, action):

        #ima 7 NOS-ova, dakle 7x2
        num_of_switch_manipulations = 14
        #print(previous_action)
        #print(action)

        if (previous_action[0] == action[0] or previous_action[0] == action[1] or previous_action[0] == action[2] or previous_action[0] == action[3] or previous_action[0] == action[4] or previous_action[0] == action[5] or previous_action[0] == action[6]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[1] == action[0] or previous_action[1] == action[1] or previous_action[1] == action[2] or previous_action[1] == action[3] or previous_action[1] == action[4] or previous_action[1] == action[5] or previous_action[1] == action[6]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[2] == action[0] or previous_action[2] == action[1] or previous_action[2] == action[2] or previous_action[2] == action[3] or previous_action[2] == action[4] or previous_action[2] == action[5] or previous_action[2] == action[6]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[3] == action[0] or previous_action[3] == action[1] or previous_action[3] == action[2] or previous_action[3] == action[3] or previous_action[3] == action[4] or previous_action[3] == action[5] or previous_action[3] == action[6]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[4] == action[0] or previous_action[4] == action[1] or previous_action[4] == action[2] or previous_action[4] == action[3] or previous_action[4] == action[4] or previous_action[4] == action[5] or previous_action[4] == action[6]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[5] == action[0] or previous_action[5] == action[1] or previous_action[5] == action[2] or previous_action[5] == action[3] or previous_action[5] == action[4] or previous_action[5] == action[5] or previous_action[5] == action[6]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[6] == action[0] or previous_action[6] == action[1] or previous_action[6] == action[2] or previous_action[6] == action[3] or previous_action[6] == action[4] or previous_action[6] == action[5] or previous_action[6] == action[6]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2
        
        #print(num_of_switch_manipulations)
        return num_of_switch_manipulations


    def reset(self, daily_consumption_percents_per_feeder):
        self.timestep = 0
        self.network_manager = nm.ODSSNetworkManagement()
        self.previous_action = 0
        #self.allow_changing_action = False

        #self.used_switches.clear()
        #for i in range (1015): #1015 indeks poslednjeg switch-a
            #self.used_switches[i] = 0

        #self.consumption_percents_per_feeder je lista koja sadrzi 24 liste koje za trenutka sadrze 3 scaling faktora, po jedan za svaki od feedera
        self.consumption_percents_per_feeder = [daily_consumption_percents_per_feeder[i:i+3] for i in range(0, len(daily_consumption_percents_per_feeder), 3)]

        #za veliku semu ide ovo ispod
        #self.consumption_percents_per_feeder_big_scheme = [daily_consumption_percents_per_feeder[i:i+4] for i in range(0, len(daily_consumption_percents_per_feeder), 4)]

        self.set_load_scaling_for_timestep()
        self.power_flow.calculate_power_flow()
        self.state = []
        switch_s_dict = self.power_flow.get_switches_apparent_power()
        self.state += [val / self.base_power for val in list(switch_s_dict.values())]
        self.state.append(self.timestep / NUM_TIMESTEPS * 1.0)
        
        #inicijalizacija available actions
        self.action_idx_used_in_thisstep = []
        self.available_actions = copy.deepcopy(self.radial_switch_combinations) #deep copy

        initial_switch_operations = [0 for i in range(self.n_switches)]
        self.switch_operations_by_index = dict(zip(self.switch_indices, initial_switch_operations))

        return self.state

    def distribute_feeder_consumptions(self, current_consumption_percents_per_feeder):
        current_consumption_percents_per_node = [0.0 for i in range(self.n_consumers)]
        current_consumption_percents_per_node[0] = current_consumption_percents_per_feeder[0]
        current_consumption_percents_per_node[1] = current_consumption_percents_per_feeder[0]
        current_consumption_percents_per_node[2] = current_consumption_percents_per_feeder[0]
        current_consumption_percents_per_node[3] = current_consumption_percents_per_feeder[0]
        current_consumption_percents_per_node[4] = current_consumption_percents_per_feeder[1]
        current_consumption_percents_per_node[5] = current_consumption_percents_per_feeder[1]
        current_consumption_percents_per_node[6] = current_consumption_percents_per_feeder[1]
        current_consumption_percents_per_node[7] = current_consumption_percents_per_feeder[2]
        current_consumption_percents_per_node[8] = current_consumption_percents_per_feeder[2]
        current_consumption_percents_per_node[9] = current_consumption_percents_per_feeder[2]
        current_consumption_percents_per_node[10] = current_consumption_percents_per_feeder[2]
        return current_consumption_percents_per_node

    def distribute_feeder_consumptions_big_scheme(self, current_consumption_percents_per_feeder):
        current_consumption_percents_per_node = [0.0 for i in range(self.n_consumers)]
        #ima 1008 cvorova, dakle self.n_consumers = 1008, 252 po svakom od 4 fidera
        for i in range (self.n_consumers):
            if (i < 252):
                current_consumption_percents_per_node[i] = current_consumption_percents_per_feeder[0] #prvi fider ima indeks 0
            elif (i >= 252 and i < 504):
                current_consumption_percents_per_node[i] = current_consumption_percents_per_feeder[1]
            elif(i >= 504 and i < 756):
                current_consumption_percents_per_node[i] = current_consumption_percents_per_feeder[2]
            else:
                current_consumption_percents_per_node[i] = current_consumption_percents_per_feeder[3]

        return current_consumption_percents_per_node

    def set_load_scaling_for_timestep(self):
        if (self.timestep == NUM_TIMESTEPS):
            return
        if (self.timestep > NUM_TIMESTEPS):  
            print('WARNING: environment.py; set_load_scaling_for_timestep; self.timestep greater than expected')

        current_consumption_percents_per_feeder = self.consumption_percents_per_feeder[self.timestep]  
        current_consumption_percents_per_node = self.distribute_feeder_consumptions(current_consumption_percents_per_feeder)

        #ispod za veliku semu
        #current_consumption_percents_per_feeder = self.consumption_percents_per_feeder_big_scheme[self.timestep]
        #current_consumption_percents_per_node = self.distribute_feeder_consumptions_big_scheme(current_consumption_percents_per_feeder)

        self.network_manager.set_load_scaling(current_consumption_percents_per_node)

    def test_environment(self):
        print(self.network_manager.is_system_radial())
        print(self.network_manager.are_all_cosumers_fed())
        print('===========Open switch=============')
        self.network_manager.open_switch('Line.Sw4')
        print(self.network_manager.is_system_radial())
        print(self.network_manager.are_all_cosumers_fed())
        print('===========CREATE LOOPS=============')
        self.network_manager.close_switch('Line.Sw4')
        self.network_manager.close_switch('Line.Sw14')
        print(self.network_manager.is_system_radial())
        print(self.network_manager.are_all_cosumers_fed())
        #self.network_manager.close_switch('Line.Sw14')
        #self.power_flow.calculate_power_flow()
        #print(self.power_flow.get_bus_voltages())
        self.network_manager.set_load_scaling()
    
    def find_all_radial_configurations(self):
        Dict = {}
        brojac = 0
        br = 0
        brojac2 = 0
        for a in range (2):
            if (a == 0):
                self.network_manager.close_switch('Line.Sw1')
            else:
                self.network_manager.open_switch('Line.Sw1')
            for b in range(2):
                if (b == 0):
                    self.network_manager.close_switch('Line.Sw2')
                else:
                    self.network_manager.open_switch('Line.Sw2')
                for c in range(2):
                    if (c == 0):
                        self.network_manager.close_switch('Line.Sw3')
                    else:
                        self.network_manager.open_switch('Line.Sw3')
                    for d in range(2):
                        if (d == 0):
                            self.network_manager.close_switch('Line.Sw4')
                        else:
                            self.network_manager.open_switch('Line.Sw4')
                        for e in range(2):
                            if (e == 0):
                                self.network_manager.close_switch('Line.Sw5')
                            else:
                                self.network_manager.open_switch('Line.Sw5')
                            for f in range(2):
                                if (f == 0):
                                    self.network_manager.close_switch('Line.Sw6')
                                else:
                                    self.network_manager.open_switch('Line.Sw6')
                                for g in range(2):
                                    if (g == 0):
                                        self.network_manager.close_switch('Line.Sw7')
                                    else:
                                        self.network_manager.open_switch('Line.Sw7')
                                    for h in range(2):
                                        if (h == 0):
                                            self.network_manager.close_switch('Line.Sw8')
                                        else:
                                            self.network_manager.open_switch('Line.Sw8')
                                        for i in range(2):
                                            if (i == 0):
                                                self.network_manager.close_switch('Line.Sw9')
                                            else:
                                                self.network_manager.open_switch('Line.Sw9')
                                            for j in range(2):
                                                if (j == 0):
                                                    self.network_manager.close_switch('Line.Sw10')
                                                else:
                                                    self.network_manager.open_switch('Line.Sw10')
                                                for k in range(2):
                                                    if (k == 0):
                                                        self.network_manager.close_switch('Line.Sw11')
                                                    else:
                                                        self.network_manager.open_switch('Line.Sw11')
                                                    for l in range(2):
                                                        if (l == 0):
                                                            self.network_manager.close_switch('Line.Sw12')
                                                        else:
                                                            self.network_manager.open_switch('Line.Sw12')
                                                        for m in range(2):
                                                            if (m == 0):
                                                                self.network_manager.close_switch('Line.Sw13')
                                                            else:
                                                                self.network_manager.open_switch('Line.Sw13')
                                                            for n in range(2):
                                                                if (n == 0):
                                                                    self.network_manager.close_switch('Line.Sw14')
                                                                else:
                                                                    self.network_manager.open_switch('Line.Sw14')

                                                                if (a == 1):
                                                                    brojac = brojac + 1

                                                                if (b == 1):
                                                                    brojac = brojac + 1

                                                                if (c == 1):
                                                                    brojac = brojac + 1

                                                                if (d == 1):
                                                                    brojac = brojac + 1

                                                                if (e == 1):
                                                                    brojac = brojac + 1

                                                                if (f == 1):
                                                                    brojac = brojac + 1

                                                                if (g == 1):
                                                                    brojac = brojac + 1

                                                                if (h == 1):
                                                                    brojac = brojac + 1

                                                                if (i == 1):
                                                                    brojac = brojac + 1

                                                                if (j == 1):
                                                                    brojac = brojac + 1

                                                                if (k == 1):
                                                                    brojac = brojac + 1

                                                                if (l == 1):
                                                                    brojac = brojac + 1

                                                                if (m == 1):
                                                                    brojac = brojac + 1

                                                                if (n == 1):
                                                                    brojac = brojac + 1
                                                                
                                                                if (brojac != 3):
                                                                    brojac = 0
                                                                    continue
                                                                else:
                                                                    brojac = 0
                                                                    self.power_flow.calculate_power_flow()
                                                                    if (self.network_manager.is_system_radial() and self.network_manager.are_all_cosumers_fed()):
                                                                        #print(a,b,c,d,e,f,g,h,i,j,k,l,m,n)
                                                                        #br = br + 1
                                                                        key = br
                                                                        Dict.setdefault(key, [])
                                                                        if (a == 1):
                                                                            Dict[key].append(1)

                                                                        if (b == 1):
                                                                            Dict[key].append(2)

                                                                        if (c == 1):
                                                                            Dict[key].append(3)

                                                                        if (d == 1):
                                                                            Dict[key].append(4)

                                                                        if (e == 1):
                                                                            Dict[key].append(5)

                                                                        if (f == 1):
                                                                            Dict[key].append(6)

                                                                        if (g == 1):
                                                                            Dict[key].append(7)

                                                                        if (h == 1):
                                                                            Dict[key].append(8)

                                                                        if (i == 1):
                                                                            Dict[key].append(9)

                                                                        if (j == 1):
                                                                            Dict[key].append(10)

                                                                        if (k == 1):
                                                                            Dict[key].append(11)

                                                                        if (l == 1):
                                                                            Dict[key].append(12)

                                                                        if (m == 1):
                                                                            Dict[key].append(13)

                                                                        if (n == 1):
                                                                            Dict[key].append(14)
                                                                        
                                                                        br = br + 1
                                                                
        self.radial_switch_combinations = Dict                                                             
        print(self.radial_switch_combinations)
        with open('file.txt', 'w') as file:
            file.write(json.dumps(Dict))
        file.close()
        print(len(Dict))

    def closing_all_switches(self):
        self.network_manager.close_switch('Line.Sw1')
        self.network_manager.close_switch('Line.Sw2')
        self.network_manager.close_switch('Line.Sw3')
        self.network_manager.close_switch('Line.Sw4')
        self.network_manager.close_switch('Line.Sw5')
        self.network_manager.close_switch('Line.Sw6')
        self.network_manager.close_switch('Line.Sw7')
        self.network_manager.close_switch('Line.Sw8')
        self.network_manager.close_switch('Line.Sw9')
        self.network_manager.close_switch('Line.Sw10')
        self.network_manager.close_switch('Line.Sw11')
        self.network_manager.close_switch('Line.Sw12')
        self.network_manager.close_switch('Line.Sw13')
        self.network_manager.close_switch('Line.Sw14')

    def closing_all_switches_big_scheme(self):
        self.network_manager.close_switch('Line.Sw1')
        self.network_manager.close_switch('Line.Sw50')
        self.network_manager.close_switch('Line.Sw97')
        self.network_manager.close_switch('Line.Sw144')
        self.network_manager.close_switch('Line.Sw191')
        self.network_manager.close_switch('Line.Sw238')
        self.network_manager.close_switch('Line.Sw253')
        self.network_manager.close_switch('Line.Sw302')
        self.network_manager.close_switch('Line.Sw349')
        self.network_manager.close_switch('Line.Sw396')
        self.network_manager.close_switch('Line.Sw443')
        self.network_manager.close_switch('Line.Sw490')
        self.network_manager.close_switch('Line.Sw505')
        self.network_manager.close_switch('Line.Sw554')
        self.network_manager.close_switch('Line.Sw601')
        self.network_manager.close_switch('Line.Sw648')
        self.network_manager.close_switch('Line.Sw695')
        self.network_manager.close_switch('Line.Sw742')
        self.network_manager.close_switch('Line.Sw757')
        self.network_manager.close_switch('Line.Sw806')
        self.network_manager.close_switch('Line.Sw853')
        self.network_manager.close_switch('Line.Sw900')
        self.network_manager.close_switch('Line.Sw947')
        self.network_manager.close_switch('Line.Sw994')
        self.network_manager.close_switch('Line.Sw1009')
        self.network_manager.close_switch('Line.Sw1010')
        self.network_manager.close_switch('Line.Sw1011')
        self.network_manager.close_switch('Line.Sw1012')
        self.network_manager.close_switch('Line.Sw1013')
        self.network_manager.close_switch('Line.Sw1014')
        self.network_manager.close_switch('Line.Sw1015')
        
    def finding_optimal_states(self):
        self.closing_all_switches_big_scheme()
        minMoneyLossesFinal = 0
        currentMoneyLosses = 0
        minLossesFinal = 0
        currentLosses = 0
        ukupno = 0
        s = 1
        k = 0
        bestResults = {}
        os1 = 0
        os2 = 0
        os3 = 0
        os4 = 0
        os5 = 0
        os6 = 0
        os7 = 0
        os1, os2, os3, os4, os5, os6, os7 = self.radial_switch_combinations[0]
        key = 0
        bestResults.setdefault(key, [])
        
        for v in range(24): 
            file = open("new_loads_4_feeders.txt", "r")
            f2 = open("Optimalno_stanje_velika_sema.txt", "a")
            scaling_factors = [0.0 for i in range(1008)]
            ceo_niz = file.readlines()
            ceo_niz = [float(z) for z in ceo_niz]

            for h in range (1008):
                if (h < 252):
                    scaling_factors[h] = ceo_niz[k] 
                elif (h >= 252 and h < 504):
                    scaling_factors[h] = ceo_niz[k+1] 
                elif(h >= 504 and h < 756):
                    scaling_factors[h] = ceo_niz[k+2] 
                else:
                    scaling_factors[h] = ceo_niz[k+3] 
            #print(scaling_factors)
            file.close()
            self.network_manager.set_load_scaling(scaling_factors)

            ff = open(str(s) + ".trenutak.txt", "a")
            for j in self.radial_switch_combinations:
                a, b, c, d, e, f, g = self.radial_switch_combinations[j]

                if (a == 144 or b == 144 or c == 144 or d == 144 or e == 144 or f == 144 or g == 144):
                    self.network_manager.open_switch('Line.Sw144')

                if (a == 191 or b == 191 or c == 191 or d == 191 or e == 191 or f == 191 or g == 191):
                    self.network_manager.open_switch('Line.Sw191')

                if (a == 238 or b == 238 or c == 238 or d == 238 or e == 238 or f == 238 or g == 238):
                    self.network_manager.open_switch('Line.Sw238')

                if (a == 396 or b == 396 or c == 396 or d == 396 or e == 396 or f == 396 or g == 396):
                    self.network_manager.open_switch('Line.Sw396')

                if (a == 443 or b == 443 or c == 443 or d == 443 or e == 443 or f == 443 or g == 443):
                    self.network_manager.open_switch('Line.Sw443')

                if (a == 490 or b == 490 or c == 490 or d == 490 or e == 490 or f == 490 or g == 490):
                    self.network_manager.open_switch('Line.Sw490')

                if (a == 648 or b == 648 or c == 648 or d == 648 or e == 648 or f == 648 or g == 648):
                    self.network_manager.open_switch('Line.Sw648')

                if (a == 695 or b == 695 or c == 695 or d == 695 or e == 695 or f == 695 or g == 695):
                    self.network_manager.open_switch('Line.Sw695')

                if (a == 742 or b == 742 or c == 742 or d == 742 or e == 742 or f == 742 or g == 742):
                    self.network_manager.open_switch('Line.Sw742')

                if (a == 853 or b == 853 or c == 853 or d == 853 or e == 853 or f == 853 or g == 853):
                    self.network_manager.open_switch('Line.Sw853')

                if (a == 900 or b == 900 or c == 900 or d == 900 or e == 900 or f == 900 or g == 900):
                    self.network_manager.open_switch('Line.Sw900')

                if (a == 947 or b == 947 or c == 947 or d == 947 or e == 947 or f == 947 or g == 947):
                    self.network_manager.open_switch('Line.Sw947')

                if (a == 994 or b == 994 or c == 994 or d == 994 or e == 994 or f == 994 or g == 994):
                    self.network_manager.open_switch('Line.Sw994')

                if (a == 1009 or b == 1009 or c == 1009 or d == 1009 or e == 1009 or f == 1009 or g == 1009):
                    self.network_manager.open_switch('Line.Sw1009')

                if (a == 1010 or b == 1010 or c == 1010 or d == 1010 or e == 1010 or f == 1010 or g == 1010):
                    self.network_manager.open_switch('Line.Sw1010')

                if (a == 1011 or b == 1011 or c == 1011 or d == 1011 or e == 1011 or f == 1011 or g == 1011):
                    self.network_manager.open_switch('Line.Sw1011')

                if (a == 1012 or b == 1012 or c == 1012 or d == 1012 or e == 1012 or f == 1012 or g == 1012):
                    self.network_manager.open_switch('Line.Sw1012')

                if (a == 1013 or b == 1013 or c == 1013 or d == 1013 or e == 1013 or f == 1013 or g == 1013):
                    self.network_manager.open_switch('Line.Sw1013')

                if (a == 1014 or b == 1014 or c == 1014 or d == 1014 or e == 1014 or f == 1014 or g == 1014):
                    self.network_manager.open_switch('Line.Sw1014')

                if (a == 1015 or b == 1015 or c == 1015 or d == 1015 or e == 1015 or f == 1015 or g == 1015):
                    self.network_manager.open_switch('Line.Sw1015')

                
                self.power_flow.calculate_power_flow()
                #print(self.power_flow.get_losses())
                currentMoneyLosses = self.power_flow.get_losses() * 0.065625 + 1 * self.get_number_of_switch_manipulations_big_scheme([os1,os2,os3,os4,os5,os6,os7], [a, b, c, d, e, f, g])
                currentLosses = self.power_flow.get_losses()
                #bestResults[key] = self.radial_switch_combinations[j]
                if (j == 0):
                    minLossesFinal = self.power_flow.get_losses()
                    currentLosses = self.power_flow.get_losses()
                    minMoneyLossesFinal = self.power_flow.get_losses() * 0.065625 + 1 * self.get_number_of_switch_manipulations_big_scheme([os1,os2,os3,os4,os5,os6,os7], [a, b, c, d, e, f, g])
                    currentMoneyLosses = self.power_flow.get_losses() * 0.065625 + 1 * self.get_number_of_switch_manipulations_big_scheme([os1,os2,os3,os4,os5,os6,os7], [a, b, c, d, e, f, g])
                    bestResults[key] = [a, b, c, d, e, f, g]
                    #os1, os2, os3 = a, b, c

                if(currentMoneyLosses < minMoneyLossesFinal):
                    minMoneyLossesFinal = currentMoneyLosses
                    minLossesFinal = currentLosses
                    bestResults[key] = [a, b, c, d, e, f, g]
                    #os1, os2, os3 = a, b, c

                #print(self.radial_switch_combinations[j])
                ff.write(json.dumps(currentLosses))
                ff.write(" kW")
                ff.write("\n")
                ff.write(json.dumps(currentMoneyLosses))
                ff.write(" $")
                ff.write("\n")
                ff.write(json.dumps(self.radial_switch_combinations[j]))
                ff.write("\n")
                ff.write("---------------------------------------------\n\n")
                    
                self.closing_all_switches_big_scheme()
                if(j == 2554):
                    os1, os2, os3, os4, os5, os6, os7 = bestResults[key]
                    ff.write("Minimum money losses for current step: ")
                    f2.write(str(s) + ". trenutak: ")
                    f2.write(json.dumps(minLossesFinal))
                    f2.write(" kW, ")
                    gubiciNovac = minLossesFinal * 0.065625
                    f2.write(json.dumps(gubiciNovac))
                    f2.write(" $, ")
                    gubiciAkcije = minMoneyLossesFinal - gubiciNovac
                    f2.write(json.dumps(gubiciAkcije))
                    f2.write(" $, ")
                    f2.write(json.dumps(minMoneyLossesFinal))
                    f2.write(" $, ")
                    f2.write(json.dumps(bestResults[key]))
                    f2.write("\n")
                    f2.write("\n")
                    ff.write(json.dumps(minLossesFinal))
                    ff.write(" $")
                    ff.write("\n")
                    ukupno += minMoneyLossesFinal
                    
                    key += 1
                    gubiciNovac = 0
                    
                    #f.write("Total load: ")
                    #f.write(json.dumps(aa))
            ff.close()
            a = 0
            b = 0
            c = 0
            s += 1
            k += 4
        f2.write(json.dumps(ukupno))
        f2.write(" $")
    

    def reading_from_load_file(self, k):
        file = open("loads.txt", "r")
        scaling_factors = [0.0 for i in range(self.n_consumers)]
        ceo_niz = file.readlines()
        ceo_niz = [float(z) for z in ceo_niz]
        scaling_factors[0] = ceo_niz[k]
        scaling_factors[1] = ceo_niz[k]
        scaling_factors[2] = ceo_niz[k]
        scaling_factors[3] = ceo_niz[k]
        scaling_factors[4] = ceo_niz[k+1]
        scaling_factors[5] = ceo_niz[k+1]
        scaling_factors[6] = ceo_niz[k+1]
        scaling_factors[7] = ceo_niz[k+2]
        scaling_factors[8] = ceo_niz[k+2]
        scaling_factors[9] = ceo_niz[k+2]
        scaling_factors[10] = ceo_niz[k+2]
        #print(scaling_factors)
        file.close()
        self.network_manager.set_load_scaling(scaling_factors)

    def reading_from_load_file_big_scheme(self, k):
        file = open("new_loads_4_feeders.txt", "r")
        scaling_factors = [0.0 for i in range(self.n_consumers)]
        ceo_niz = file.readlines()
        ceo_niz = [float(z) for z in ceo_niz]
        totalLoad = 0

        for i in range (1008):
            if (i < 252):
                scaling_factors[i] = ceo_niz[k]
            elif (i >= 252 and i < 504):
                scaling_factors[i] = ceo_niz[k+1]
            elif (i >= 504 and i < 756):
                scaling_factors[i] = ceo_niz[k+2]
            else:
                scaling_factors[i] = ceo_niz[k+3]

        #print(totalLoad)
        file.close()
        self.network_manager.set_load_scaling(scaling_factors)

    def opening_switches(self, a, b, c):
        self.network_manager.open_switch('Line.Sw'+str(a))
        self.network_manager.open_switch('Line.Sw'+str(b))
        self.network_manager.open_switch('Line.Sw'+str(c))


    def crtanje_krivih(self):

        fig, axs = plt.subplots(3)
        axs[1].set(ylabel='Load [MW]')
        axs[2].set(xlabel='Hour [h]')

        fig.suptitle('Dayly curves for consumers on feeders')
        f = open("DaylyCurve1.txt", "r")
        ceo_niz1 = f.readlines()
        ceo_niz1g = f.readlines()
        ceo_niz1d = f.readlines()

        ceo_niz1 = [float(z1) for z1 in ceo_niz1]

        x_axis = [1 + j for j in range(24)]
        y_axis1 = ceo_niz1
        axs[0].plot(x_axis, y_axis1, color = 'blue', label = 'Feeder 1')
        axs[0].axis([1, 24 , 0, 1.4])
        axs[0].legend(loc = 'upper left')
        #plt.axis([1, 24 , 0, 1.4])
        axs[0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]) 
        axs[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8 , 1.0, 1.2, 1.4])
        #line_1, = plt.plot(x_axis, y_axis1, label = 'Feeder1', color = 'darkblue')
        
        for j1 in range (24):
            ceo_niz1[j1] += 0.3
        y_axis1g = ceo_niz1
        axs[0].plot(x_axis, y_axis1g, '--', color = 'lightblue')
        #line_1g, = plt.plot(x_axis, y_axis1g, '--', color = 'lightblue')

        for j2 in range (24):
            ceo_niz1[j2] -= 0.6
            if (ceo_niz1[j2] < 0):
                ceo_niz1[j2] = 0

        y_axis1d = ceo_niz1
        axs[0].plot(x_axis, y_axis1d, '--', color = 'lightblue')
        #line_1d, = plt.plot(x_axis, y_axis1d, '--', color = 'lightblue')
        f.close()
        axs[0].grid(True)
        #axs[0].xticks(x_axis)

        ##################################################################
        f = open("DaylyCurve2.txt", "r")
        ceo_niz2 = f.readlines()
        ceo_niz2g = f.readlines()
        ceo_niz2d = f.readlines()

        ceo_niz2 = [float(z2) for z2 in ceo_niz2]

        #x_axis = [1 + j for j in range(24)]
        y_axis2 = ceo_niz2
        axs[1].plot(x_axis, y_axis2, color = 'magenta', label = 'Feeder 2')
        axs[1].legend(loc = 'upper left')
        axs[1].axis([1, 24 , 0, 1.4])
        axs[1].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]) 
        axs[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8 , 1.0, 1.2, 1.4])
        #line_2, = plt.plot(x_axis, y_axis2, label = 'Feeder2', color = 'magenta')
        
        for j3 in range (24):
            ceo_niz2[j3] += 0.3
        y_axis2g = ceo_niz2
        axs[1].plot(x_axis, y_axis2g, '--', color = 'lightpink')
        #line_2g, = plt.plot(x_axis, y_axis2g, '--', color = 'lightpink')

        for j4 in range (24):
            ceo_niz2[j4] -= 0.6
            if (ceo_niz2[j4] < 0):
                ceo_niz2[j4] = 0

        y_axis2d = ceo_niz2
        axs[1].plot(x_axis, y_axis2d, '--', color = 'lightpink')
        #line_2d, = plt.plot(x_axis, y_axis2d, '--', color = 'lightpink')
        f.close()
        axs[1].grid(True)
        #axs[0].xticks(np.arange(1, 25, 1))
        #plt.yticks(np.arange(0, 1.4, 0.2))
        #axs[1].xticks(x_axis)

        ##################################################################
        f = open("DaylyCurve3.txt", "r")
        ceo_niz3 = f.readlines()
        ceo_niz3g = f.readlines()
        ceo_niz3d = f.readlines()

        ceo_niz3 = [float(z3) for z3 in ceo_niz3]

        #x_axis = [1 + j for j in range(24)]
        y_axis3 = ceo_niz3
        axs[2].plot(x_axis, y_axis3, color = 'darkgreen', label = 'Feeder 3')
        axs[2].legend(loc = 'upper left')
        axs[2].axis([1, 24 , 0, 1.4])
        axs[2].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]) 
        axs[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8 , 1.0, 1.2, 1.4])
        #line_3, = plt.plot(x_axis, y_axis3, label = 'Feeder3', color = 'darkgreen')
        
        for j5 in range (24):
            ceo_niz3[j5] += 0.3
        y_axis3g = ceo_niz3
        axs[2].plot(x_axis, y_axis3g, '--', color = 'lightgreen')
        #line_3g, = plt.plot(x_axis, y_axis3g, '--', color = 'lightgreen')

        for j6 in range (24):
            ceo_niz3[j6] -= 0.6
            if (ceo_niz3[j6] < 0):
                ceo_niz3[j6] = 0

        y_axis3d = ceo_niz3
        axs[2].plot(x_axis, y_axis3d, '--', color = 'lightgreen')
        #line_3d, = plt.plot(x_axis, y_axis3d, '--', color = 'lightgreen')
        f.close()
        axs[2].grid(True)
        #axs[2].xticks(x_axis)
        
        #####################################################################
        
        
        
        
        
        
        

        
        
        
        #####################################################################
        #plt.legend(handles = [axs])
        #plt.grid(True)
        #plt.xticks(x_axis)
        
        #plt.title('Dayly curves for consumers on feeders')
        #plt.xlabel('Hour [h]') 
        #plt.ylabel('Load [MW]') 
        plt.savefig("daylycurve_vertical.png")
        plt.show()

    def checking_results(self):

        switch_combinations = {
            0: [191, 490, 695, 1009, 1011, 1012, 1014],
            1: [191, 238, 490, 695, 1009, 1012, 1014],
            2: [191, 238, 490, 695, 1009, 1012, 1014],
            3: [191, 238, 490, 695, 1009, 1012, 1014],
            4: [191, 238, 490, 695, 1009, 1012, 1014],
            5: [191, 947, 1009, 1011, 1012, 1014, 1015],
            6: [191, 1009, 1011, 1012, 1013, 1014, 1015],
            7: [191, 1009, 1011, 1012, 1013, 1014, 1015],
            8: [191, 1009, 1011, 1012, 1013, 1014, 1015],
            9: [191, 238, 490, 1009, 1012, 1013, 1014],
            10: [1009, 1010, 1011, 1012, 1013, 1014, 1015],
            11: [1009, 1010, 1011, 1012, 1013, 1014, 1015],
            12: [1009, 1010, 1011, 1012, 1013, 1014, 1015],
            13: [1009, 1010, 1011, 1012, 1013, 1014, 1015],
            14: [191, 1009, 1011, 1012, 1013, 1014, 1015],
            15: [490, 1009, 1010, 1012, 1013, 1014, 1015],
            16: [238, 490, 947, 994, 1009, 1010, 1012],
            17: [238, 490, 947, 994, 1009, 1010, 1012],
            18: [238, 490, 947, 994, 1009, 1010, 1012],
            19: [238, 947, 994, 1009, 1010, 1012, 1015],
            20: [238, 947, 994, 1009, 1010, 1012, 1015],
            21: [238, 742, 947, 1009, 1010, 1012, 1014],
            22: [238, 742, 947, 1009, 1010, 1012, 1014],
            23: [238, 490, 695, 1009, 1010, 1012, 1014]
        }
            
        self.closing_all_switches_big_scheme()
        moneyLossesTotal = 0
        actionLossesTotal = 0
        totalLosses = 0
        totalMoneyLoss = 0
        sw1, sw2, sw3, sw4, sw5, sw6, sw7 = [1009, 1010, 1011, 1012, 1013, 1014, 1015]
        k = 0
        f = open("Rezultati_velika_sema.txt", "a")

        for i in range(24):

            self.network_manager.close_switch('Line.Sw'+str(sw1))
            self.network_manager.close_switch('Line.Sw'+str(sw2))
            self.network_manager.close_switch('Line.Sw'+str(sw3))
            self.network_manager.close_switch('Line.Sw'+str(sw4))
            self.network_manager.close_switch('Line.Sw'+str(sw5))
            self.network_manager.close_switch('Line.Sw'+str(sw6))
            self.network_manager.close_switch('Line.Sw'+str(sw7))
            sw1, sw2, sw3, sw4, sw5, sw6, sw7 = switch_combinations[i]
            self.network_manager.open_switch('Line.Sw'+str(sw1))
            self.network_manager.open_switch('Line.Sw'+str(sw2))
            self.network_manager.open_switch('Line.Sw'+str(sw3))
            self.network_manager.open_switch('Line.Sw'+str(sw4))
            self.network_manager.open_switch('Line.Sw'+str(sw5))
            self.network_manager.open_switch('Line.Sw'+str(sw6))
            self.network_manager.open_switch('Line.Sw'+str(sw7))
            self.reading_from_load_file_big_scheme(k)
            self.power_flow.calculate_power_flow()
            losses = self.power_flow.get_losses()
            moneyLosses = self.power_flow.get_losses() * 0.065625
            if (i == 0):
                actionLosses = 1 * self.get_number_of_switch_manipulations_big_scheme([1009, 1010, 1011, 1012, 1013, 1014, 1015], switch_combinations[i])
            else:
                actionLosses = 1 * self.get_number_of_switch_manipulations_big_scheme(switch_combinations[i - 1], switch_combinations[i])
            moneyLossesTotal += moneyLosses
            actionLossesTotal += actionLosses
            totalLosses += losses
            totalMoneyLoss += (moneyLosses + actionLosses)

            f.write(str(i + 1) + ". trenutak: ")
            f.write(json.dumps(losses))
            f.write(" kW, ")
            f.write(json.dumps(moneyLosses))
            f.write(" $, ")
            f.write(json.dumps(actionLosses))
            f.write(" $, [")
            f.write(json.dumps(sw1))
            f.write(", ")
            f.write(json.dumps(sw2))
            f.write(", ")
            f.write(json.dumps(sw3))
            f.write(", ")
            f.write(json.dumps(sw4))
            f.write(", ")
            f.write(json.dumps(sw5))
            f.write(", ")
            f.write(json.dumps(sw6))
            f.write(", ")
            f.write(json.dumps(sw7))
            f.write("]")
            f.write("\n\n")
            losses = 0
            moneyLosses = 0
            actionLosses = 0
            k += 4


        f.write("Total losses: ")
        f.write(json.dumps(totalLosses))
        f.write(" kW, ")
        f.write(json.dumps(moneyLossesTotal))
        f.write(" $")
        f.write("\n")
        f.write("Action losses: ")
        f.write(json.dumps(actionLossesTotal))
        f.write(" $")
        f.write("\n")
        f.write("Total money losses: ")
        f.write(json.dumps(totalMoneyLoss))
        f.write("\n")
        f.close()

    

    def checking_voltages(self):

        busVoltages = []
        radial_combinations = [0 for l in range (66)]
        numbOfCustomersWithBadVoltage = 0
        swCombinationsWithBadVoltage = 0
        sw1, sw2, sw3 = self.radial_switch_combinations[0]
        k = 0
        timestep = 0
        for i in range (24):

            k = 0
            f = open(str(i + 1) + ". trenutak.txt", "a")

            for j in range (66):

                f.write(str(j) + ". kombinacija: ")
                self.network_manager.close_switch('Line.Sw'+str(sw1))
                self.network_manager.close_switch('Line.Sw'+str(sw2))
                self.network_manager.close_switch('Line.Sw'+str(sw3))
                sw1, sw2, sw3 = self.radial_switch_combinations[k]
                self.network_manager.open_switch('Line.Sw'+str(sw1))
                self.network_manager.open_switch('Line.Sw'+str(sw2))
                self.network_manager.open_switch('Line.Sw'+str(sw3))

                self.reading_from_load_file(timestep)
                self.power_flow.calculate_power_flow()
                busVoltages = self.power_flow.get_bus_voltages()
                #print(busVoltages)

                for z in range (26):
                    if (busVoltages[z] < 0.95):
                        numbOfCustomersWithBadVoltage += 1

                if (numbOfCustomersWithBadVoltage > 0):
                    swCombinationsWithBadVoltage += 1
                    radial_combinations[j] += 1

                f.write(json.dumps(numbOfCustomersWithBadVoltage))
                f.write("\n")
                numbOfCustomersWithBadVoltage = 0
                k += 1

            timestep += 3
            f2 = open("Checking voltages results new 0.95.txt", "a")
            f2.write(str(i + 1) + ". trenutak: ")
            f2.write(json.dumps(swCombinationsWithBadVoltage))
            f2.write("\n\n")

            f.write("Number of switch combinations with bad voltages: ")
            f.write(json.dumps(swCombinationsWithBadVoltage))
            f.close()
            swCombinationsWithBadVoltage = 0

        counter = 0
        f2.write("[")
        for b in range (66):
            if (radial_combinations[b] > 23):
                f2.write(json.dumps(b))
                counter += 1
                f2.write(", ")
        f2.write("]\n")
        f2.write(json.dumps(counter))
        f2.close()

        f3 = open("Radial_comb2.txt", "w")
        for a in range (66):
            f3.write(str(a) + ". kombinacija: ")
            f3.write(json.dumps(radial_combinations[a]))
            f3.write("\n")
        f3.close()

    def creatingDataset(self):

        f1 = open("DaylyCurve1.txt", "r")
        loads1 = f1.readlines()
        loads1 = [float(z) for z in loads1]
        for i in range (24):
            loads1[i] = loads1[i]/3
        f1.close()

        f2 = open("DaylyCurve11.txt", "w")
        for i in range (24):
            f2.write(json.dumps(loads1[i]))
            f2.write("\n")
        f2.close()


        f3 = open("DaylyCurve2.txt", "r")
        loads2 = f3.readlines()
        loads2 = [float(z) for z in loads2]
        for i in range (24):
            loads2[i] = loads2[i]/3
        f3.close()

        f4 = open("DaylyCurve22.txt", "w")
        for i in range (24):
            f4.write(json.dumps(loads2[i]))
            f4.write("\n")
        f4.close()


        f5 = open("DaylyCurve3.txt", "r")
        loads3 = f5.readlines()
        loads3 = [float(z) for z in loads3]
        for i in range (24):
            loads3[i] = loads3[i]/3
        f5.close()

        f6 = open("DaylyCurve33.txt", "w")
        for i in range (24):
            f6.write(json.dumps(loads3[i]))
            f6.write("\n")
        f6.close()


        f7 = open("DaylyCurve4.txt", "r")
        loads4 = f7.readlines()
        loads4 = [float(z) for z in loads4]
        for i in range (24):
            loads4[i] = loads4[i]/3
        f7.close()

        f8 = open("DaylyCurve44.txt", "w")
        for i in range (24):
            f8.write(json.dumps(loads4[i]))
            f8.write("\n")
        f8.close()




    def checking_voltages_for_explicit_configurations(self):

        switch_combinations = {
            0: [4, 12, 13],
            1: [4, 12, 13],
            2: [4, 12, 13],
            3: [4, 12, 13],
            4: [4, 12, 13],
            5: [4, 12, 13],
            6: [7, 10, 14],
            7: [7, 10, 14],
            8: [10, 12, 14],
            9: [12, 13, 14],
            10: [12, 13, 14],
            11: [12, 13, 14],
            12: [12, 13, 14],
            13: [12, 13, 14],
            14: [12, 13, 14],
            15: [12, 13, 14],
            16: [4, 12, 13],
            17: [4, 12, 13],
            18: [4, 12, 13],
            19: [4, 12, 13],
            20: [4, 12, 13],
            21: [4, 12, 13],
            22: [4, 12, 13],
            23: [4, 12, 13]
        }

        sw1, sw2, sw3 = [12, 13, 14]
        k = 0
        numbOfCustomersWithBadVoltage = 0
        timestep = 0

        f = open("PedjaNeven_0.8.txt", "a")

        for i in range (24):

            f.write(str(i + 1) + ". trenutak: ")
            self.network_manager.close_switch('Line.Sw'+str(sw1))
            self.network_manager.close_switch('Line.Sw'+str(sw2))
            self.network_manager.close_switch('Line.Sw'+str(sw3))
            sw1, sw2, sw3 = self.radial_switch_combinations[k]
            self.network_manager.open_switch('Line.Sw'+str(sw1))
            self.network_manager.open_switch('Line.Sw'+str(sw2))
            self.network_manager.open_switch('Line.Sw'+str(sw3))

            self.reading_from_load_file(timestep)
            self.power_flow.calculate_power_flow()
            busVoltages = self.power_flow.get_bus_voltages()
            #print(busVoltages)

            for z in range (26):
                if (busVoltages[z] < 0.8):
                    numbOfCustomersWithBadVoltage += 1

            timestep += 3
            k += 1

            f.write(json.dumps(numbOfCustomersWithBadVoltage))
            f.write("\n")
            numbOfCustomersWithBadVoltage = 0
        f.close()

    def dat_big_scheme(self):

        f = open("SwitchStatic1.txt", "w")
        for i in range (1016):
            
            if (i == 2 or i == 49 or i == 96 or i == 143 or i == 190 or i == 237):
                #f.write(json.dumps(1)
                f.write("1\n")

            elif (i == 254 or i == 301 or i == 348 or i == 395 or i == 442 or i == 489):
                #f.write(json.dumps(1)
                f.write("1\n")

            elif (i == 506 or i == 553 or i == 600 or i == 647 or i == 694 or i == 741):
                #f.write(json.dumps(1)
                f.write("1\n")

            elif (i == 758 or i == 805 or i == 852 or i == 899 or i == 946 or i == 993):
                #f.write(json.dumps(1)
                f.write("1\n")

            else:
                f.write("0\n")

        f.close()

        f1 = open("SwitchStatic2.txt", "w")
        for j in range (1016):
            if (j < 1008):
                f1.write("0\n")
            else:
                f1.write("1\n")
        f1.close()


        f2 = open("SwitchDinamic1.txt", "w")
        for k in range (1016):
            f2.write("1\n")
        f2.close()


        f3 = open("SwitchDinamic2.txt", "w")
        for l in range (1016):
            if (l < 1008):
                f3.write("1\n")
            else:
                f3.write("0\n")
        f3.close()


        f4 = open("IndDaylyCurve.txt", "w")
        for m in range (1008):
            if (m < 252):
                f4.write("1\n")
            elif (m >= 252 and m < 504):
                f4.write("2\n")
            elif (m >= 504 and m < 756):
                f4.write("3\n")
            else:
                f4.write("4\n")
                
        f4.close()

        f5 = open("Ppick.txt", "w")
        for n in range (1014):
            if (n < 1008):
                f5.write("35\n")
            else:
                f5.write("0\n")
        f5.close()

        f6 = open("SwitchStatic11.txt")

        switches = f6.readlines()
        br = 0
        switches = [float(z) for z in switches]

        for o in range (1020):
            if (switches[o] == 1):
                br += 1
        print(br)
        f6.close()

        
    def generating_big_scheme(self):

        #f = open("Velika_sema.txt", "w")
        #for i in range (1008):   
            #f.write("New Line.Line_" + str(i + 1) + " Phases=3 BaseFreq=50 Bus1=Bus_" + str(i) + " Bus2=Bus_" + str(i + 1) + " LineCode=AB Length=0.01  units=km\n")                                                         
        #f.close()

        f1 = open("Opterecenja.txt", "w")
        for i in range (1008):
            f1.write("New Load.Load_" + str(i + 1) + " Bus1=Bus_" + str(i + 1) + " Phases=3 conn=wye Model=1 kV=11.547 kW=70 kvar=14\n")
        f1.close

    def find_all_radial_configurations_big_scheme(self):
        Dict = {}
        brojac = 0
        br = 0
        brojac2 = 0
        for a in range (2):
            if (a == 0):
                self.network_manager.close_switch('Line.Sw144')
            else:
                self.network_manager.open_switch('Line.Sw144')
            for b in range(2):
                if (b == 0):
                    self.network_manager.close_switch('Line.Sw191')
                else:
                    self.network_manager.open_switch('Line.Sw191')
                for c in range(2):
                    if (c == 0):
                        self.network_manager.close_switch('Line.Sw238')
                    else:
                        self.network_manager.open_switch('Line.Sw238')
                    for d in range(2):
                        if (d == 0):
                            self.network_manager.close_switch('Line.Sw396')
                        else:
                            self.network_manager.open_switch('Line.Sw396')
                        for e in range(2):
                            if (e == 0):
                                self.network_manager.close_switch('Line.Sw443')
                            else:
                                self.network_manager.open_switch('Line.Sw443')
                            for f in range(2):
                                if (f == 0):
                                    self.network_manager.close_switch('Line.Sw490')
                                else:
                                    self.network_manager.open_switch('Line.Sw490')
                                for g in range(2):
                                    if (g == 0):
                                        self.network_manager.close_switch('Line.Sw648')
                                    else:
                                        self.network_manager.open_switch('Line.Sw648')
                                    for h in range(2):
                                        if (h == 0):
                                            self.network_manager.close_switch('Line.Sw695')
                                        else:
                                            self.network_manager.open_switch('Line.Sw695')
                                        for i in range(2):
                                            if (i == 0):
                                                self.network_manager.close_switch('Line.Sw742')
                                            else:
                                                self.network_manager.open_switch('Line.Sw742')
                                            for j in range(2):
                                                if (j == 0):
                                                    self.network_manager.close_switch('Line.Sw853')
                                                else:
                                                    self.network_manager.open_switch('Line.Sw853')
                                                for k in range(2):
                                                    if (k == 0):
                                                        self.network_manager.close_switch('Line.Sw900')
                                                    else:
                                                        self.network_manager.open_switch('Line.Sw900')
                                                    for l in range(2):
                                                        if (l == 0):
                                                            self.network_manager.close_switch('Line.Sw947')
                                                        else:
                                                            self.network_manager.open_switch('Line.Sw947')
                                                        for m in range(2):
                                                            if (m == 0):
                                                                self.network_manager.close_switch('Line.Sw994')
                                                            else:
                                                                self.network_manager.open_switch('Line.Sw994')
                                                            for n in range(2):
                                                                if (n == 0):
                                                                    self.network_manager.close_switch('Line.Sw1009')
                                                                else:
                                                                    self.network_manager.open_switch('Line.Sw1009')
                                                                for o in range (2):
                                                                    if (o == 0):
                                                                        self.network_manager.close_switch('Line.Sw1010')
                                                                    else:
                                                                        self.network_manager.open_switch('Line.Sw1010')
                                                                    for p in range(2):
                                                                        if (p == 0):
                                                                            self.network_manager.close_switch('Line.Sw1011')
                                                                        else:
                                                                            self.network_manager.open_switch('Line.Sw1011')
                                                                        for q in range(2):
                                                                            if (q == 0):
                                                                                self.network_manager.close_switch('Line.Sw1012')
                                                                            else:
                                                                                self.network_manager.open_switch('Line.Sw1012')
                                                                            for r in range(2):
                                                                                if (r == 0):
                                                                                    self.network_manager.close_switch('Line.Sw1013')
                                                                                else:
                                                                                    self.network_manager.open_switch('Line.Sw1013')
                                                                                for s in range(2):
                                                                                    if (s == 0):
                                                                                        self.network_manager.close_switch('Line.Sw1014')
                                                                                    else:
                                                                                        self.network_manager.open_switch('Line.Sw1014')
                                                                                    for t in range(2):
                                                                                        if (t == 0):
                                                                                            self.network_manager.close_switch('Line.Sw1015')
                                                                                        else:
                                                                                            self.network_manager.open_switch('Line.Sw1015')
                                                                                                        
                                                                                        if (a == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (b == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (c == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (d == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (e == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (f == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (g == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (h == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (i == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (j == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (k == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (l == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (m == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (n == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (o == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (p == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (q == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (r == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (s == 1):
                                                                                            brojac = brojac + 1

                                                                                        if (t == 1):
                                                                                            brojac = brojac + 1


                                                                                        if (brojac != 7):
                                                                                            brojac = 0
                                                                                            continue
                                                                                        else:
                                                                                            brojac = 0
                                                                                            self.power_flow.calculate_power_flow()

                                                                                            if (self.network_manager.is_system_radial() and self.network_manager.are_all_cosumers_fed()):
                                                                                                key = br
                                                                                                Dict.setdefault(key, [])

                                                                                                if (a == 1):
                                                                                                    Dict[key].append(144)

                                                                                                if (b == 1):
                                                                                                    Dict[key].append(191)

                                                                                                if (c == 1):
                                                                                                    Dict[key].append(238)

                                                                                                if (d == 1):
                                                                                                    Dict[key].append(396)

                                                                                                if (e == 1):
                                                                                                    Dict[key].append(443)

                                                                                                if (f == 1):
                                                                                                    Dict[key].append(490)

                                                                                                if (g == 1):
                                                                                                    Dict[key].append(648)

                                                                                                if (h == 1):
                                                                                                    Dict[key].append(695)

                                                                                                if (i == 1):
                                                                                                    Dict[key].append(742)

                                                                                                if (j == 1):
                                                                                                    Dict[key].append(853)

                                                                                                if (k == 1):
                                                                                                    Dict[key].append(900)

                                                                                                if (l == 1):
                                                                                                    Dict[key].append(947)

                                                                                                if (m == 1):
                                                                                                    Dict[key].append(994)

                                                                                                if (n == 1):
                                                                                                    Dict[key].append(1009)

                                                                                                if (o == 1):
                                                                                                    Dict[key].append(1010)

                                                                                                if (p == 1):
                                                                                                    Dict[key].append(1011)

                                                                                                if (q == 1):
                                                                                                    Dict[key].append(1012)

                                                                                                if (r == 1):
                                                                                                    Dict[key].append(1013)

                                                                                                if (s == 1):
                                                                                                    Dict[key].append(1014)

                                                                                                if (t == 1):
                                                                                                    Dict[key].append(1015)

                                                                                                br = br + 1                                                                                                                                   
        self.radial_switch_combinations = Dict                                                             
        #print(self.radial_switch_combinations)
        with open('radial_switch_combinations.txt', 'w') as file:
            file.write(json.dumps(Dict))
        file.close()
        print(len(Dict))

    def reading_dict(self):

        dictionary = {}
        f = open("radial_switch_combinations.txt", "r")
        f1 = open("radial_switch_combinations2.txt", "w")
        #dictionary = f.readlines()
        dictionary = json.loads(f.read())
        #print(dictionary)
        for key in dictionary:
            f1.write(json.dumps(key) + ": " + json.dumps(dictionary[key]) + "\n")

        f.close()
        f1.close()

    def creating_new_dataset_all_combined(self):

        f1 = open("DaylyCurve1.txt", "r")
        loads1 = f1.readlines()
        loads1 = [float(z) for z in loads1]
        f1.close()

        f2 = open("DaylyCurve2.txt", "r")
        loads2 = f2.readlines()
        loads2 = [float(z) for z in loads2]
        f2.close()

        f3 = open("DaylyCurve3.txt", "r")
        loads3 = f3.readlines()
        loads3 = [float(z) for z in loads3]
        f3.close()

        f4 = open("DaylyCurve4.txt", "r")
        loads4 = f4.readlines()
        loads4 = [float(z) for z in loads4]
        f4.close()

        f = open("new_loads_4_feeders.txt", "w")
        for i in range (24):
            f.write(json.dumps(loads1[i]))
            f.write("\n")
            f.write(json.dumps(loads2[i]))
            f.write("\n")
            f.write(json.dumps(loads3[i]))
            f.write("\n")
            f.write(json.dumps(loads4[i]))
            f.write("\n")

    def datasets_excel_big_scheme(self):

        f = open("new_loads_4_feeders.txt", "r")
        loads = f.readlines()
        loads = [float(z) for z in loads]
        f.close()

        f1 = open("dataExcel.txt", "w")
        for i in range (96):
            f1.write("df.loc[index, " + str(i) + "] = " + json.dumps(loads[i]) + "\n")
        f1.close()


    def provera(self):

        totalLoad = 0
        self.closing_all_switches_big_scheme()
        self.network_manager.open_switch('Line.Sw191')
        self.network_manager.open_switch('Line.Sw695')
        self.network_manager.open_switch('Line.Sw742')
        self.network_manager.open_switch('Line.Sw490')
        self.network_manager.open_switch('Line.Sw1011')
        self.network_manager.open_switch('Line.Sw1012')
        self.network_manager.open_switch('Line.Sw1009')

        self.reading_from_load_file_big_scheme(0)
        self.power_flow.calculate_power_flow()
        print("losses: ", self.power_flow.get_losses())
        print("\n")
        #self.network_manager.print_loads()

    def provera2(self):

        radial_comb = {}
        brojac = 0
        f = open("radial_switch_combinations.txt", "r")
        radial_comb = json.loads(f.read())
        f.close()

        for key in radial_comb:
            self.closing_all_switches_big_scheme()
            sw1, sw2, sw3, sw4, sw5, sw6, sw7 = radial_comb[key]
            self.network_manager.open_switch('Line.Sw' + str(sw1))
            self.network_manager.open_switch('Line.Sw' + str(sw2))
            self.network_manager.open_switch('Line.Sw' + str(sw3))
            self.network_manager.open_switch('Line.Sw' + str(sw4))
            self.network_manager.open_switch('Line.Sw' + str(sw5))
            self.network_manager.open_switch('Line.Sw' + str(sw6))
            self.network_manager.open_switch('Line.Sw' + str(sw7))

            self.reading_from_load_file_big_scheme(0)
            self.power_flow.calculate_power_flow()

            if (self.network_manager.is_system_radial() and self.network_manager.are_all_cosumers_fed()):
                brojac += 1

        print(brojac)

    def checking_voltages_big_scheme(self):

        radial_switch_combinations = {}
        f6 = open("radial_switch_combinations.txt", "r")
        radial_switch_combinations = json.loads(f6.read())
        f6.close()

        busVoltages = []
        radial_combinations = [0 for l in range (3567)]
        numbOfCustomersWithBadVoltage = 0
        swCombinationsWithBadVoltage = 0
        sw1, sw2, sw3, sw4, sw5, sw6, sw7 = [1009, 1010, 1011, 1012, 1013, 1014, 1015]
        k = 0
        timestep = 0
        for i in range (24):

            k = 0
            #f = open(str(i + 1) + ". trenutak.txt", "a")

            for key in radial_switch_combinations:

                #f.write(str(key) + ". kombinacija: ")
                self.network_manager.close_switch('Line.Sw'+str(sw1))
                self.network_manager.close_switch('Line.Sw'+str(sw2))
                self.network_manager.close_switch('Line.Sw'+str(sw3))
                self.network_manager.close_switch('Line.Sw'+str(sw4))
                self.network_manager.close_switch('Line.Sw'+str(sw5))
                self.network_manager.close_switch('Line.Sw'+str(sw6))
                self.network_manager.close_switch('Line.Sw'+str(sw7))
                sw1, sw2, sw3, sw4, sw5, sw6, sw7 = radial_switch_combinations[key]
                self.network_manager.open_switch('Line.Sw'+str(sw1))
                self.network_manager.open_switch('Line.Sw'+str(sw2))
                self.network_manager.open_switch('Line.Sw'+str(sw3))
                self.network_manager.open_switch('Line.Sw'+str(sw4))
                self.network_manager.open_switch('Line.Sw'+str(sw5))
                self.network_manager.open_switch('Line.Sw'+str(sw6))
                self.network_manager.open_switch('Line.Sw'+str(sw7))

                self.reading_from_load_file_big_scheme(timestep)
                self.power_flow.calculate_power_flow()
                busVoltages = self.power_flow.get_bus_voltages()
                #print(busVoltages)

                for z in range (1040):
                    if (busVoltages[z] < 0.85):
                        numbOfCustomersWithBadVoltage += 1
                        break

                if (numbOfCustomersWithBadVoltage > 0):
                    swCombinationsWithBadVoltage += 1
                    radial_combinations[k] += 1

                #f.write(json.dumps(numbOfCustomersWithBadVoltage))
                #f.write("\n")
                numbOfCustomersWithBadVoltage = 0
                k += 1

            timestep += 4
            #f2 = open("Checking voltages results big scheme.txt", "a")
            #f2.write(str(i + 1) + ". trenutak: ")
            #f2.write(json.dumps(swCombinationsWithBadVoltage))
            #f2.write("\n\n")

            #f.write("Number of switch combinations with bad voltages: ")
            #f.write(json.dumps(swCombinationsWithBadVoltage))
            #f.close()
            swCombinationsWithBadVoltage = 0

        counter = 0
        #f2.write("[")
        #for b in range (3567):
            #if (radial_combinations[b] > 23):
                #f2.write(json.dumps(b))
                #counter += 1
                #f2.write(", ")
        #f2.write("]\n")
        #f2.write(json.dumps(counter))
        #f2.close()

        f3 = open("Radial_comb2_85.txt", "w")
        for a in range (3567):
            f3.write(str(a) + ": [")
            f3.write(json.dumps(radial_combinations[a]))
            f3.write("],\n")
        f3.close()

    def redukovanje_broja_kombinacija_velika_sema(self):

        brojac = 0
        my_list = []
        radial_comb_reduced = {}
        f = open("sw_comb.txt", "r")
        my_list = json.loads(f.read())
        f.close()
        for i in range (1012):
            brojac += 1
            self.radial_switch_combinations.pop(my_list[i])

        #print(self.radial_switch_combinations)

        j = 0
        radial_switch_combinations = {}
        f1 = open("sw_comb_reduced.txt", "w")
        for key in self.radial_switch_combinations:
            radial_switch_combinations[j] = self.radial_switch_combinations[key]
            f1.write(json.dumps(j))
            f1.write(": ")
            f1.write(json.dumps(radial_switch_combinations[j]))
            f1.write(",\n")
            j += 1

        f1.close()






