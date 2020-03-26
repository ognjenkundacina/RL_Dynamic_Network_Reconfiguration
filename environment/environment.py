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
        self.previous_action = 0

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

        for switch_index in self.switch_indices:
            if switch_index in self.radial_switch_combinations[action]:
                self.network_manager.open_switch(self.switch_names_by_index[switch_index]) 
            else:
                self.network_manager.close_switch(self.switch_names_by_index[switch_index])
        

    #action: 0..n_actions
    def step(self, action):
        self.timestep += 1
        #self.switch_operations_by_index[toogled_switch_index] += 1
        next_state = self._update_state(action)

        reward = self.calculate_reward(action)

        done = (self.timestep == NUM_TIMESTEPS)

        self.set_load_scaling_for_timestep()

        self.previous_action = action
        return next_state, reward, done

    def calculate_reward(self, action):
        reward = 0
        #self.power_flow.get_losses() daje gubitke u kW, pa odmah imamo i kWh
        reward -= self.power_flow.get_losses() * 0.065625

        #reward -= 5 ** (self.power_flow.get_losses() * 0.065625 / 20.0)
 
        reward -= self.switching_action_cost * self.get_number_of_switch_manipulations(self.radial_switch_combinations[self.previous_action], self.radial_switch_combinations[action])

        #zbog numerickih pogodnost je potrebno skalirati nagradu tako da moduo total episode reward bude oko 1.0
        reward /= 400.0
        return reward

    def get_number_of_switch_manipulations(self, previous_action, action):
        num_of_switch_manipulations = 6

        if (previous_action[0] == action[0] or previous_action[0] == action[1] or previous_action[0] == action[2]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[1] == action[0] or previous_action[1] == action[1] or previous_action[1] == action[2]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2

        if (previous_action[2] == action[0] or previous_action[2] == action[1] or previous_action[2] == action[2]):
            num_of_switch_manipulations = num_of_switch_manipulations - 2
        
        #print(previous_action)
        #print(action)
        #print(num_of_switch_manipulations)
        return num_of_switch_manipulations


    def reset(self, daily_consumption_percents_per_feeder):
        self.timestep = 0
        self.network_manager = nm.ODSSNetworkManagement()
        self.previous_action = 0

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
        minMoneyLossesFinal = 0
        currentMoneyLosses = 0
        minLossesFinal = 0
        currentLosses = 0
        s = 1
        k = 0
        bestResults = {}
        os1 = 0
        os2 = 0
        os3 = 0
        os1, os2, os3 = self.radial_switch_combinations[0]
        key = 0
        bestResults.setdefault(key, [])
        
        for v in range(24): 
            file = open("loads.txt", "r")
            f2 = open("Optimalno stanje_gubici_cena 1.txt", "a")
            scaling_factors = [0.0 for i in range(11)]
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

                self.power_flow.calculate_power_flow()
                #print(self.power_flow.get_losses())
                currentMoneyLosses = self.power_flow.get_losses() * 0.065625 + 1 * self.get_number_of_switch_manipulations([os1,os2,os3], [a, b, c]) * 1
                currentLosses = self.power_flow.get_losses()
                #bestResults[key] = self.radial_switch_combinations[j]
                if (j == 0):
                    #aa = 0
                    if(v == 0):
                        bestResults[key] = self.radial_switch_combinations[j]
                    else: 
                        bestResults[key] = [os1, os2, os3]

                    minLossesFinal = self.power_flow.get_losses()
                    currentLosses = self.power_flow.get_losses()
                    minMoneyLossesFinal = self.power_flow.get_losses() * 0.065625 + 1 * self.get_number_of_switch_manipulations([os1,os2,os3], [a, b, c]) * 1
                    currentMoneyLosses = self.power_flow.get_losses() * 0.065625 + 1 * self.get_number_of_switch_manipulations([os1,os2,os3], [a, b, c]) * 1
                    
                    #aa = (0 - 1) * self.power_flow.get_network_injected_p() - self.power_flow.get_losses()

                if(currentLosses < minLossesFinal):
                    minMoneyLossesFinal = currentMoneyLosses
                    minLossesFinal = currentLosses
                    bestResults[key] = self.radial_switch_combinations[j]
                    os1, os2, os3 = self.radial_switch_combinations[j]

                #print(self.radial_switch_combinations[j])
                f.write(json.dumps(currentLosses))
                f.write(" kW")
                f.write("\n")
                f.write(json.dumps(currentMoneyLosses))
                f.write(" $")
                f.write("\n")
                f.write(json.dumps(self.radial_switch_combinations[j]))
                f.write("\n")
                f.write("---------------------------------------------\n\n")
                    
                self.closing_all_switches()
                if(j == 185):
                    f.write("Minimum losses for current step: ")
                    f2.write(str(s) + ". trenutak: ")
                    f2.write(json.dumps(minLossesFinal))
                    f2.write(" kW, ")
                    f2.write(json.dumps(minMoneyLossesFinal))
                    f2.write(" $, ")
                    f2.write(json.dumps(bestResults[key]))
                    f2.write("\n")
                    f2.write("\n")
                    f.write(json.dumps(minLossesFinal))
                    f.write(" $")
                    f.write("\n")
                    key += 1
                    
                    #f.write("Total load: ")
                    #f.write(json.dumps(aa))
            f.close()
            a = 0
            b = 0
            c = 0
            s += 1
            k += 3
    

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

    def opening_switches(self, a, b, c):
        self.network_manager.open_switch('Line.Sw'+str(a))
        self.network_manager.open_switch('Line.Sw'+str(b))
        self.network_manager.open_switch('Line.Sw'+str(c))


    def finding_optimal_states_4(self):

        self.closing_all_switches()
        totalMoneyLoss = 0
        totalLossFinal = 0
        totalLoss = 0
        totalMoneyLossFinal = 10000
        sw1 = 0
        sw2 = 0
        sw3 = 0

        swa1 = 0
        swa2 = 0
        swa3 = 0

        swb1 = 0
        swb2 = 0
        swb3 = 0

        swc1 = 0
        swc2 = 0
        swc3 = 0

        swd1 = 0
        swd2 = 0
        swd3 = 0
        brojac = 0

        for a in range(18):
            for b in range(18):
                for c in range(18):
                    for d in range(18):
                        self.closing_all_switches()
                        totalLoss = 0
                        totalMoneyLoss = 0
                        sw1, sw2, sw3 = self.radial_switch_combinations[a]
                        self.network_manager.open_switch('Line.Sw'+str(sw1))
                        self.network_manager.open_switch('Line.Sw'+str(sw2))
                        self.network_manager.open_switch('Line.Sw'+str(sw3))
                        self.reading_from_load_file(0)
                        self.power_flow.calculate_power_flow()
                        totalLoss = self.power_flow.get_losses()
                        totalMoneyLoss = self.power_flow.get_losses() * 0.065625 + 1 * self.get_number_of_switch_manipulations(self.radial_switch_combinations[a], [12, 13, 14])

                        self.network_manager.close_switch('Line.Sw'+str(sw1))
                        self.network_manager.close_switch('Line.Sw'+str(sw2))
                        self.network_manager.close_switch('Line.Sw'+str(sw3))
                        sw1, sw2, sw3 = self.radial_switch_combinations[b]
                        self.network_manager.open_switch('Line.Sw'+str(sw1))
                        self.network_manager.open_switch('Line.Sw'+str(sw2))
                        self.network_manager.open_switch('Line.Sw'+str(sw3))
                        self.reading_from_load_file(3)
                        self.power_flow.calculate_power_flow()
                        totalLoss += self.power_flow.get_losses()
                        totalMoneyLoss += (self.power_flow.get_losses() * 0.065625 + 1 * self.get_number_of_switch_manipulations(self.radial_switch_combinations[a], self.radial_switch_combinations[b]))

                        self.network_manager.close_switch('Line.Sw'+str(sw1))
                        self.network_manager.close_switch('Line.Sw'+str(sw2))
                        self.network_manager.close_switch('Line.Sw'+str(sw3))
                        sw1, sw2, sw3 = self.radial_switch_combinations[c]
                        self.network_manager.open_switch('Line.Sw'+str(sw1))
                        self.network_manager.open_switch('Line.Sw'+str(sw2))
                        self.network_manager.open_switch('Line.Sw'+str(sw3))
                        self.reading_from_load_file(6)
                        self.power_flow.calculate_power_flow()
                        totalLoss += self.power_flow.get_losses()
                        totalMoneyLoss += (self.power_flow.get_losses() * 0.065625 + 1 * self.get_number_of_switch_manipulations(self.radial_switch_combinations[b], self.radial_switch_combinations[c]))

                        self.network_manager.close_switch('Line.Sw'+str(sw1))
                        self.network_manager.close_switch('Line.Sw'+str(sw2))
                        self.network_manager.close_switch('Line.Sw'+str(sw3))
                        sw1, sw2, sw3 = self.radial_switch_combinations[d]
                        self.network_manager.open_switch('Line.Sw'+str(sw1))
                        self.network_manager.open_switch('Line.Sw'+str(sw2))
                        self.network_manager.open_switch('Line.Sw'+str(sw3))
                        self.reading_from_load_file(9)
                        self.power_flow.calculate_power_flow()
                        totalLoss += self.power_flow.get_losses()
                        totalMoneyLoss += (self.power_flow.get_losses() * 0.065625 + 1 * self.get_number_of_switch_manipulations(self.radial_switch_combinations[c], self.radial_switch_combinations[d]))

                        if(totalMoneyLoss < totalMoneyLossFinal):

                            totalMoneyLossFinal = totalMoneyLoss
                            totalLossFinal = totalLoss
                            swa1, swa2, swa3 = self.radial_switch_combinations[a]
                            swb1, swb2, swb3 = self.radial_switch_combinations[b]
                            swc1, swc2, swc3 = self.radial_switch_combinations[c]
                            swd1, swd2, swd3 = self.radial_switch_combinations[d]
                        
                #if(brojac == 0):
                    #print(totalMoneyLoss)
                    #print(self.radial_switch_combinations[a])
                    #print(self.radial_switch_combinations[b])
                    #print(self.radial_switch_combinations[c])
                    #print(self.radial_switch_combinations[d])
                        brojac += 1

        #print(brojac)
        f = open("Optimalno stanje_4 trenutka_cena 1.txt", "a")
        f.write(json.dumps(totalMoneyLossFinal))
        f.write(" $")
        f.write("\n")

        f.write(json.dumps(totalLossFinal))
        f.write(" kW")
        f.write("\n")

        f.write("1. trenutak: [")
        f.write(json.dumps(swa1))
        f.write(", ")
        f.write(json.dumps(swa2))
        f.write(", ")
        f.write(json.dumps(swa3))
        f.write("]")
        f.write("\n")

        f.write("2. trenutak: [")
        f.write(json.dumps(swb1))
        f.write(", ")
        f.write(json.dumps(swb2))
        f.write(", ")
        f.write(json.dumps(swb3))
        f.write("]")
        f.write("\n")

        f.write("3. trenutak: [")
        f.write(json.dumps(swc1))
        f.write(", ")
        f.write(json.dumps(swc2))
        f.write(", ")
        f.write(json.dumps(swc3))
        f.write("]")
        f.write("\n")

        f.write("4. trenutak: [")
        f.write(json.dumps(swd1))
        f.write(", ")
        f.write(json.dumps(swd2))
        f.write(", ")
        f.write(json.dumps(swd3))
        f.write("]")
        f.write("\n")

                        

                                                                    