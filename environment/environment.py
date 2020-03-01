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

        self.state_space_dims = len(self.power_flow.get_bus_voltages()) + 1
        ####self.n_actions = 1 + len(self.network_manager.get_all_switch_names())
        self.n_actions = 1 + len(self.network_manager.get_all_capacitors())
        self.n_consumers = self.network_manager.get_load_count()
        self.timestep = 0
        self.switching_action_cost = 0.1
        self.zero_action_name = 'Zero action index - go in the next timestep'

        ####self.switch_names = self.network_manager.get_all_switch_names()
        ##### action_index = 0 = kraj sekvence za aktuelni interva
        ##### action_index != 0 = indeks prekidaca, pri cemo indeksiranje pocinje od 1
        ####self.action_indices = [i for i in range(self.n_actions + 1)]
        ####self.switch_indices = [i for i in range(1, self.n_actions + 1)]
        ####self.switch_names_by_index = dict(zip(self.switch_indices, self.switch_names))

        self.switch_names = self.network_manager.get_all_capacitor_switch_names()
        # action_index = 0 = kraj sekvence za aktuelni interva
        # action_index != 0 = indeks prekidaca, pri cemo indeksiranje pocinje od 1
        self.action_indices = [i for i in range(self.n_actions + 1)]
        self.switch_indices = [i for i in range(1, self.n_actions + 1)]
        self.switch_names_by_index = dict(zip(self.switch_indices, self.switch_names))
        
    def _update_state(self):
        self.power_flow.calculate_power_flow()

        bus_voltages_dict = self.power_flow.get_bus_voltages()
        self.state = list(bus_voltages_dict.values())
        self.state.append(self.timestep / NUM_TIMESTEPS * 1.0)
        #line_rated_powers_dict = self.power_flow.get_line_rated_powers()
        #self.state = list(line_rated_powers_dict.values())

        if (len(self.state) != self.state_space_dims):
            print('Environment: len(self.state) != self.state_space_dims')

        return self.state

    def _reset_available_actions_for_next_timestep(self):
        #ovdje je potrebno postaviti available actions na sljedecu vriejdnost:
        #all_actions - forbidden_actions
        #forbidden_actions - prekidaci koji su vec tri puta iskoristeni u okviru epizode
        self.available_actions = copy.deepcopy(self.switch_names_by_index) #deep copy
        self.available_actions[0] = self.zero_action_name #dodaje key value pair u dictionary
        for switch_index in self.switch_operations_by_index:
            if self.switch_operations_by_index[switch_index] > 3:
                #print('Forbidden action: ', switch_index)
                self.available_actions.pop(switch_index)

    def _add_or_remove_zero_from_available_actions(self):
        if self._is_configuration_radial():
            if not (0 in self.available_actions): #provjerava da key nije u dictionariju
                self.available_actions[0] = self.zero_action_name
                #cak je dobro staviti lazni switch name za ovu akciju, to ce nam biti dobar test
                #ako nismo dobro rukvoali ovom akcijom poslacemo je u openDSS koji ce dati gresku
        else:
            if 0 in self.available_actions():
                self.available_actions.pop(0)

    def _is_configuration_radial(self):
        #todo nekako preko opendss skontati
        #ako ne moze, onda napraviti neki dictionary sa svim kombinacjama switcheva koje daju radijalnu konfiguraciju
        #i iscupati iz njega je li trenutna radijalna
        return True

    #action: 0..n_actions
    def step(self, action):
        if action==0:
            self.timestep += 1
            self._reset_available_actions_for_next_timestep()
        else:
            self.network_manager.toogle_switch_status(self.available_actions[action])
            self.switch_operations_by_index[action] += 1
            self.available_actions.pop(action)
            self._add_or_remove_zero_from_available_actions()
        
        next_state = self._update_state()

        reward = self.calculate_reward(action)

        done = (self.timestep == NUM_TIMESTEPS)

        return next_state, reward, done

    def calculate_reward(self, action):
        reward = 0
        if action == 0:
            reward -= self.power_flow.get_losses() / 1000.0
        else:
            reward -= self.switching_action_cost

        return reward

    def reset(self, consumption_percents, capacitor_statuses):
        self.network_manager.set_load_scaling(consumption_percents)
        self.network_manager.set_capacitors_initial_status(capacitor_statuses)
        
        self.power_flow.calculate_power_flow()
        bus_voltages_dict = self.power_flow.get_bus_voltages()
        self.state = list(bus_voltages_dict.values())
        self.state.append(self.timestep / NUM_TIMESTEPS * 1.0)
        #line_rated_powers_dict = self.power_flow.get_line_rated_powers()
        #self.state = list(line_rated_powers_dict.values())
        self.timestep = 0
        self.available_actions = copy.deepcopy(self.switch_names_by_index) #deep copy
        self.available_actions[0] = self.zero_action_name

        initial_switch_operations = [0 for i in range(self.n_actions)]
        self.switch_operations_by_index = dict(zip(self.switch_indices, initial_switch_operations))

        return self.state