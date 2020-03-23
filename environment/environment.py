import gym
from gym import spaces
import random
import numpy as np
from power_algorithms.odss_power_flow import ODSSPowerFlow
from power_algorithms.odss_network_management import ODSSNetworkManagement
import power_algorithms.odss_network_management as nm
from config import *
import copy
import json

class Environment(gym.Env):
    
    def __init__(self):
        super(Environment, self).__init__()
        
        self.state = []

        self.network_manager = nm.ODSSNetworkManagement()
        self.power_flow = ODSSPowerFlow()
        self.power_flow.calculate_power_flow() #potrebno zbog odredjivanja state_space_dims

        self.state_space_dims = len(self.power_flow.get_switches_apparent_power()) + 1

        self.radial_switch_combinations = radial_switch_combinations

        self.n_actions = len(self.radial_switch_combinations)
        self.n_consumers = self.network_manager.get_load_count()
        self.timestep = 0
        self.switching_action_cost = 1.0
        self.base_power = 4000

        self.switch_names = self.network_manager.get_all_switch_names()
        self.n_switches = len(self.network_manager.get_all_switch_names())
        # indeks prekidaca, pri cemo indeksiranje pocinje od 1
        self.switch_indices = [i for i in range(1, self.n_switches + 1)]
        self.switch_names_by_index = dict(zip(self.switch_indices, self.switch_names))
        
    def _update_state(self, action):
        
        self._update_switch_statuses(action)
        #self._update_available_actions() #todo implement
        self.power_flow.calculate_power_flow()

        self.state = []
        switch_s_dict = self.power_flow.get_switches_apparent_power()
        self.state += [val / self.base_power for val in list(switch_s_dict.values())]
        self.state.append(self.timestep / NUM_TIMESTEPS * 1.0)

        if (len(self.state) != self.state_space_dims):
            print('Environment: len(self.state) != self.state_space_dims')

        return self.state


    def _update_available_actions(self):
        #mozda kasnije da dodamo da ne dozvoljavamo akcije koje ce neki prekidac angazovati vise 
        #od tri puta
        #koristiti self.switch_operations_by_index
        #self.available_actions.pop(switch_index)
        pass

    def _update_switch_statuses(self, action):
        global previous_action
        global new_action
        previous_action = [0 for i in range(3)]
        new_action = [0 for j in range(3)]
        ii = 0
        jj = 0

        for switch_index in self.switch_indices:
            if switch_index in self.radial_switch_combinations[action]:
                if (self.network_manager.is_opened(self.switch_names_by_index[switch_index])):
                    previous_action[ii] = switch_index
                    ii += 1
                self.network_manager.open_switch(self.switch_names_by_index[switch_index])
                new_action[jj] = switch_index
                jj += 1
            else:
                if (self.network_manager.is_opened(self.switch_names_by_index[switch_index])):
                    previous_action[ii] = switch_index
                    ii += 1
                self.network_manager.close_switch(self.switch_names_by_index[switch_index])
        
        #print(previous_action)
        #print(new_action)

    #action: 0..n_actions
    def step(self, action):
        self.timestep += 1
        #self.switch_operations_by_index[toogled_switch_index] += 1
        next_state = self._update_state(action)

        reward = self.calculate_reward(action)

        done = (self.timestep == NUM_TIMESTEPS)

        self.set_load_scaling_for_timestep()

        return next_state, reward, done

    def calculate_reward(self, action):
        reward = 0
        #self.power_flow.get_losses() daje gubitke u kW, pa odmah imamo i kWh
        reward -= self.power_flow.get_losses() * 0.065625

        #reward -= 5 ** (self.power_flow.get_losses() * 0.065625 / 20.0)
 
        reward -= self.switching_action_cost * self.get_number_of_switch_manipulations()

        #zbog numerickih pogodnost je potrebno skalirati nagradu tako da moduo total episode reward bude oko 1.0
        reward /= 20.0
        return reward

    def get_number_of_switch_manipulations(self):
        num_of_switch_manipulations = 6

        if (previous_action[0] == new_action[0] or previous_action[0] == new_action[1] or previous_action[0] == new_action[2]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[1] == new_action[0] or previous_action[1] == new_action[1] or previous_action[1] == new_action[2]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[2] == new_action[0] or previous_action[2] == new_action[1] or previous_action[2] == new_action[2]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2
        
        return num_of_switch_manipulations


    def reset(self, daily_consumption_percents_per_feeder):
        self.timestep = 0
        self.network_manager = nm.ODSSNetworkManagement()

        #self.consumption_percents_per_feeder je lista koja sadrzi 24 liste koje za trenutka sadrze 3 scaling faktora, po jedan za svaki do feedera
        self.consumption_percents_per_feeder = [daily_consumption_percents_per_feeder[i:i+3] for i in range(0, len(daily_consumption_percents_per_feeder), 3)]
        self.set_load_scaling_for_timestep()
        self.power_flow.calculate_power_flow()
        self.state = []
        switch_s_dict = self.power_flow.get_switches_apparent_power()
        self.state += [val / self.base_power for val in list(switch_s_dict.values())]
        self.state.append(self.timestep / NUM_TIMESTEPS * 1.0)
        
        #inicijalizacija available actions
        self.action_idx_used_in_thisstep = []
        self.available_actions = copy.deepcopy(self.radial_switch_combinations) #deep copy

        initial_switch_operations = [0 for i in range(self.n_switches )]
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
        current_consumption_percents_per_node[11] = current_consumption_percents_per_feeder[0]
        current_consumption_percents_per_node[12] = current_consumption_percents_per_feeder[1]
        current_consumption_percents_per_node[13] = current_consumption_percents_per_feeder[0]
        return current_consumption_percents_per_node

    def set_load_scaling_for_timestep(self):
        if (self.timestep == NUM_TIMESTEPS):
            return
        if (self.timestep > NUM_TIMESTEPS):  
            print('WARNING: environment.py; set_load_scaling_for_timestep; self.timestep greater than expected')  
        current_consumption_percents_per_feeder = self.consumption_percents_per_feeder[self.timestep]
        current_consumption_percents_per_node = self.distribute_feeder_consumptions(current_consumption_percents_per_feeder)
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
        
    def finding_optimal_states(self):
        self.closing_all_switches()
        minLossesFinal = 0
        currentLosses = 0
        s = 1
        k = 0

        for v in range(24): 
            file = open("loads.txt", "r")
            f2 = open("Optimalni gubici.txt", "a")
            scaling_factors = [0.0 for i in range(self.n_consumers)]
            ceo_niz = file.readlines()
            ceo_niz = [int(z) for z in ceo_niz]
            scaling_factors[0] = ceo_niz[k] * 0.7 /1000
            scaling_factors[1] = ceo_niz[k] * 0.7 /1000
            scaling_factors[2] = ceo_niz[k] * 0.7 /1000
            scaling_factors[3] = ceo_niz[k] * 0.7 /1000
            scaling_factors[4] = ceo_niz[k+1] * 0.7 /1000
            scaling_factors[5] = ceo_niz[k+1] * 0.7 /1000
            scaling_factors[6] = ceo_niz[k+1] * 0.7 /1000
            scaling_factors[7] = ceo_niz[k+2] * 0.7 /1000
            scaling_factors[8] = ceo_niz[k+2] * 0.7 /1000
            scaling_factors[9] = ceo_niz[k+2] * 0.7 /1000
            scaling_factors[10] = ceo_niz[k+2] * 0.7 /1000
            scaling_factors[11] = ceo_niz[k] * 0.7 /1000
            scaling_factors[12] = ceo_niz[k+1] * 0.7 /1000
            scaling_factors[13] = ceo_niz[k] * 0.7 /1000
            #print(scaling_factors)
            file.close()
            self.network_manager.set_load_scaling(scaling_factors)

            f = open(str(s) + ".trenutak.txt", "a")
            for j in self.radial_switch_combinations:
                a, b, c = self.radial_switch_combinations[j]

                if (a == 1 or b == 1 or c == 1):
                    self.network_manager.open_switch('Line.Sw1')

                if (a == 2 or b == 2 or c == 2):
                    self.network_manager.open_switch('Line.Sw2')
                                                                            
                if (a == 3 or b == 3 or c == 3):
                    self.network_manager.open_switch('Line.Sw3')
                                                                            
                if (a == 4 or b == 4 or c == 4):
                    self.network_manager.open_switch('Line.Sw4')
                                                                            
                if (a == 5 or b == 5 or c == 5):
                    self.network_manager.open_switch('Line.Sw5')
                                                                            
                if (a == 6 or b == 6 or c == 6):
                    self.network_manager.open_switch('Line.Sw6')
                                                                            
                if (a == 7 or b == 7 or c == 7):
                    self.network_manager.open_switch('Line.Sw7')
                                                                                
                if (a == 8 or b == 8 or c == 8):
                    self.network_manager.open_switch('Line.Sw8')
                                                                                
                if (a == 9 or b == 9 or c == 9):
                    self.network_manager.open_switch('Line.Sw9')
                                                                                
                if (a == 10 or b == 10 or c == 10):
                    self.network_manager.open_switch('Line.Sw10')
                                                                                
                if (a == 11 or b == 11 or c == 11):
                    self.network_manager.open_switch('Line.Sw11')
                                                                                
                if (a == 12 or b == 12 or c == 12):
                    self.network_manager.open_switch('Line.Sw12')
                                                                                
                if (a == 13 or b == 13 or c == 13):
                    self.network_manager.open_switch('Line.Sw13')
                                                                            
                if (a == 14 or b == 14 or c == 14):
                    self.network_manager.open_switch('Line.Sw14')

                a = 0
                b = 0
                c = 0
                self.power_flow.calculate_power_flow()
                #print(self.power_flow.get_losses())
                currentLosses = self.power_flow.get_losses()
                if (j == 0):
                    aa = 0
                    minLossesFinal = self.power_flow.get_losses()
                    currentLosses = self.power_flow.get_losses()
                    aa = (0 - 1) * self.power_flow.get_network_injected_p() - self.power_flow.get_losses()

                if(currentLosses < minLossesFinal):
                    minLossesFinal = currentLosses

                #print(self.radial_switch_combinations[j])
                f.write(json.dumps(self.power_flow.get_losses()))
                f.write("\n")
                f.write(json.dumps(self.radial_switch_combinations[j]))
                f.write("\n")
                f.write("---------------------------------------------\n\n")
                    
                self.closing_all_switches()
                if(j == 185):
                    f.write("Minimum losses for current step: ")
                    f2.write(str(s) + ". trenutak: ")
                    f2.write(json.dumps(minLossesFinal))
                    f2.write("\n")
                    f2.write("\n")
                    f.write(json.dumps(minLossesFinal))
                    f.write("\n")
                    f.write("Total load: ")
                    f.write(json.dumps(aa))
            f.close()
            s += 1
            k += 3
        #print(len(Dict))
        f2.close()
        #print(j)
                                                                    