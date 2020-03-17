import gym
from gym import spaces
import random
import numpy as np
from power_algorithms.odss_power_flow import ODSSPowerFlow
import power_algorithms.odss_network_management as nm
from config import *
import copy

class Environment(gym.Env):
    
    def __init__(self):
        super(Environment, self).__init__()
        
        self.state = []

        self.network_manager = nm.ODSSNetworkManagement()
        self.power_flow = ODSSPowerFlow()
        self.power_flow.calculate_power_flow() #potrebno zbog odredjivanja state_space_dims

        self.state_space_dims = len(self.power_flow.get_switches_apparent_power()) + 1

        self.radial_switch_combinations = {
            0 : [12, 13, 14],
            1 : [11, 13, 14],
            2 : [10, 13, 14]
        }

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
        
    def _update_state(self):
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

    def _update_switch_statuses(action):
        #koristiti:
        #self.network_manager.toogle_switch_status(self.switch_index))
        pass


    #action: 0..n_actions
    def step(self, action):
        self.timestep += 1
        self.set_load_scaling_for_timestep()
            
        #self.switch_operations_by_index[toogled_switch_index] += 1
        
        next_state = self._update_state()

        reward = self.calculate_reward(action)

        done = (self.timestep == NUM_TIMESTEPS)

        return next_state, reward, done

    def calculate_reward(self, action):
        reward = 0
        #self.power_flow.get_losses() daje gubitke u kW, pa odmah imamo i kWh
        #reward -= self.power_flow.get_losses() * 0.065625
            
        reward -= 5 ** (self.power_flow.get_losses() * 0.065625 / 20.0)
            
        #reward -= self.switching_action_cost * self.num_of_switching_actions

        #zbog numerickih pogodnost je potrebno skalirati nagradu tako da moduo total episode reward bude oko 1.0
        #reward /= 20.0
        return reward

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
                                                                
                                                                        
        print(Dict)
        
    