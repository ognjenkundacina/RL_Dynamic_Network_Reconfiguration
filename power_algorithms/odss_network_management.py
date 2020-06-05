import opendssdirect as dss
from opendssdirect.utils import Iterator

class ODSSNetworkManagement:
    def __init__(self):
        dss.run_command('Redirect power_algorithms/NRsema.dss')
        self.nominal_load_kW = {}
        self.nominal_load_kVAr = {}
        self.__save_nominal_load_powers() #remember nominal load values so that load scaling feature can use them

    def __save_nominal_load_powers(self):
        for loadName in Iterator(dss.Loads, 'Name'):
            #dss.Loads.Name(loadName())
            kW = dss.Loads.kW()
            kVAr = dss.Loads.kvar()
            self.nominal_load_kW.update( {loadName() : kW} )
            self.nominal_load_kVAr.update( {loadName() : kVAr} )


    def get_all_switch_names(self):
        switch_names = []
        for line_name in dss.Lines.AllNames():
            if ('sw' in line_name):
                switch_names.append('Line.' + line_name)
        return switch_names

    def get_all_switch_statuses_as_double(self):
        sw_statuses = []
        for line_name in dss.Lines.AllNames():
            if ('sw' in line_name):
                switch_name = 'Line.' + line_name
                dss.Circuit.SetActiveElement(switch_name)
                if (dss.CktElement.IsOpen(0, 0)):
                    sw_statuses.append(0.0)
                else:
                    sw_statuses.append(1.0)
        return sw_statuses

    def close_switch(self, switch_name):
        dss.Circuit.SetActiveElement(switch_name)
        #prvi argument: terminal = 0
        #drugi argument: phases = 0 #all phases
        dss.CktElement.Close(0, 0)

    def open_switch(self, switch_name):
        dss.Circuit.SetActiveElement(switch_name)
        dss.CktElement.Open(0, 0)
    
    def is_opened(self, switch_name):
        dss.Circuit.SetActiveElement(switch_name)
        return dss.CktElement.IsOpen(0, 0)

    def toogle_switch_status(self, switch_name):
        dss.Circuit.SetActiveElement(switch_name)
        if dss.CktElement.IsOpen(0, 0):
            dss.CktElement.Close(0, 0)
        else:
            dss.CktElement.Open(0, 0)

    def toogle_capacitor_status(self, capSwitchName):
        dss.Capacitors.Name(capSwitchName)
        currentState = dss.Capacitors.States()
        if (currentState[0] == 0):
            dss.Capacitors.Close()
        else:
            dss.Capacitors.Open()

    def get_all_capacitor_switch_names(self):
        return dss.Capacitors.AllNames()

    def get_all_capacitors(self):
        capacitorStatuses = {}
        for capName in Iterator(dss.Capacitors, 'Name'):
            dss.Capacitors.Name(capName())
            currentState = dss.Capacitors.States()
            capacitorStatuses.update( {capName() : currentState[0]} )

        return capacitorStatuses
        
    def set_capacitor_status(self, cap_name, on):
        dss.Capacitors.Name(cap_name)
        if on:
            dss.Capacitors.Close()
        else:
            dss.Capacitors.Open()

    def set_capacitors_initial_status(self, capacitors_statuses):
        if (len(capacitors_statuses) != dss.Capacitors.Count()):
            print("(ERROR) Input list of capacitor statuses {} is not the same length as number of capacitors {}".format(len(capacitors_statuses), dss.Capacitors.Count()))
            return
        
        index = 0
        for capName in Iterator(dss.Capacitors, 'Name'):
            dss.Capacitors.Name(capName())
            if (capacitors_statuses[index] == 0):
                dss.Capacitors.Open()
            else:
                dss.Capacitors.Close()
            index = index + 1

    def set_load_scaling(self, scaling_factors):
        if (len(scaling_factors) != dss.Loads.Count()):
            print("(ERROR) Input list of scaling factors {} is not the same length as number of loads {}".format(len(scaling_factors), dss.Loads.Count()))
            return

        index = 0
        for loadName in Iterator(dss.Loads, 'Name'):
            dss.Loads.Name(loadName())
            dss.Loads.kW(self.nominal_load_kW[loadName()] * scaling_factors[index])
            dss.Loads.kvar(self.nominal_load_kVAr[loadName()] * scaling_factors[index])
            index = index + 1
            
    def get_load_count(self):
        return dss.Loads.Count()

    def is_system_radial(self):
        dss.Text.Command('CalcVoltageBases')
        return dss.Topology.NumLoops() == 0

    def are_all_cosumers_fed(self):
        dss.Text.Command('CalcVoltageBases')
        return dss.Topology.NumIsolatedLoads() == 0

    def print_loads(self):
        for loadName in Iterator(dss.Loads, 'Name'):
            kW = dss.Loads.kW()
            kVAr = dss.Loads.kvar()
            print(loadName())
            print(kW)
            print(kVAr) 
