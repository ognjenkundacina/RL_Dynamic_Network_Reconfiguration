import opendssdirect as dss
from opendssdirect.utils import Iterator

class ODSSNetworkManagement:
    def __init__(self):
        dss.run_command('Redirect power_algorithms/IEEE123_scheme/Run_IEEE123Bus.DSS')
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

    def print_loads(self):
        for loadName in Iterator(dss.Loads, 'Name'):
            kW = dss.Loads.kW()
            kVAr = dss.Loads.kvar()
            print(loadName())
            print(kW)
            print(kVAr)
