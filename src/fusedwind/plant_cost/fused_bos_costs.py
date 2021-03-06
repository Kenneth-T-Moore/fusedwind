# KLD: _cost model for FUSED_Wind - most basic structure

from openmdao.lib.datatypes.api import Float, Int, Array, VarTree
from openmdao.main.api import Component, Assembly, VariableTree

from fusedwind.plant_cost.fused_tcc import BaseTurbineCostModel
from fusedwind.interface import base, implement_base

##########################################################
# Plant Balance of Station and CAPEX Models

# Balance of Station Cost Model
# Base Variable Trees, Components and Assemblies
@base
class BaseBOSCostAggregator(Component):
    """ Base balance of station cost aggregator for doing some auxiliary cost calculations needed to get a full wind plant balance of station cost estimate.    """

    # Outputs
    bos_costs = Float(iotype='out', desc='Overall wind plant balance of station/system costs up to point of comissioning')

@base
class BaseBOSCostModel(Assembly):
    """ Base balance of station cost assembly for coupling models to get a full wind plant balance of station cost estimate.    """

    # Outputs
    bos_costs = Float(iotype='out', desc='Overall wind plant balance of station/system costs up to point of comissioning')
    
def configure_base_bos(assembly):
    """ Base configure method for a balance of station cost assembly for coupling models to get a full wind plant balance of station cost estimate.  It adds a default balance of station cost aggregator component.    """
      
    assembly.add('bos', BaseBOSCostAggregator())
    
    assembly.driver.workflow.add('bos')

    assembly.connect('bos.bos_costs','bos_costs')

#Extended Variable Trees, Components and Assemblies
@base
class BOSVarTree(VariableTree):
    """ Base balance of station cost variable tree based on the DOE/NREL system cost breakdown structure.    """

    development_costs = Float(desc='Overall wind plant balance of station/system costs up to point of comissioning')
    preparation_and_staging_costs = Float(desc='Site preparation and staging')
    transportation_costs = Float(desc='Any transportation costs to site / staging site') #BOS or turbine cost?
    foundation_and_substructure_costs = Float(desc='Foundation and substructure costs')
    electrical_costs = Float(desc='Collection system, substation, transmission and interconnect costs')
    assembly_and_installation_costs = Float(desc='Assembly and installation costs')
    soft_costs = Float(desc='Contingencies, bonds, reserves, decommissioning, profits, and construction financing costs')
    other_costs = Float(desc='Bucket for any other costs not captured above')

@implement_base(BaseBOSCostAggregator)
class ExtendedBOSCostAggregator(Component):
    """ Extended balance of station cost aggregator for doing some auxiliary cost calculations needed to get a full wind plant balance of station cost estimate as well as a detailed cost breakdown.    """

    # Outputs
    bos_breakdown = VarTree(BOSVarTree(), iotype='out', desc='BOS cost breakdown')
    bos_costs = Float(iotype='out', desc='Overall wind plant balance of station/system costs up to point of comissioning')

@implement_base(BaseBOSCostModel)
class ExtendedBOSCostModel(Assembly):
    """ Extended balance of station cost assembly for coupling models to get a full wind plant balance of station cost estimate as well as a detailed cost breakdown.    """

    # Outputs
    bos_breakdown = VarTree(BOSVarTree(), iotype='out', desc='BOS cost breakdown')
    bos_costs = Float(iotype='out', desc='Overall wind plant balance of station/system costs up to point of comissioning')

def configure_extended_bos(assembly):
    """ Extended configure method for a balance of station cost assembly for coupling models to get a full wind plant balance of station cost estimate.  It replaces the base balance of station cost aggregator component with an extended version which contains the full balance of station variable tree.    """

    configure_base_bos(assembly)
    
    assembly.replace('bos', ExtendedBOSCostAggregator())

    assembly.connect('bos.bos_breakdown','bos_breakdown')

# Full Variable Trees, Components and Assemblies
@base
class FullBOSVarTree(VariableTree):
    """ Full balance of station cost variable tree based on the DOE/NREL system cost breakdown structure.    """

    management_costs = Float(desc='Project management costs')
    development_costs = Float(desc='Overall wind plant balance of station/system costs up to point of comissioning')
    preparation_and_staging_costs = Float(desc='Site preparation and staging')
    transportation_costs = Float(desc='Any transportation costs to site / staging site')
    foundation_and_substructure_costs = Float(desc='Foundation and substructure costs')
    collection_and_substation_costs = Float(desc='Collection system and onsite substation costs')
    transmission_and_interconnection_costs = Float(desc='Transmission and grid interconnection costs')
    assembly_and_installation_costs = Float(desc='Assembly and installation costs')
    contingencies_and_insurance_costs = Float(desc='Contingencies, bonds, reserves for project')
    decommissioning_costs = Float(desc='_costs associated with plant decommissioning at end of life')
    construction_financing_costs = Float(desc='Construction financing costs')
    other_costs = Float(desc='Bucket for any other costs not captured above')
    developer_profits = Float(desc='Developer profits')

@implement_base(BaseBOSCostAggregator)
class FullBOSCostAggregator(Component):
    """ Full balance of station cost aggregator for doing some auxiliary cost calculations needed to get a full wind plant balance of station cost estimate as well as a detailed cost breakdown.    """

    # outputs
    bos_breakdown = VarTree(FullBOSVarTree(), iotype='out', desc='BOS cost breakdown')
    bos_costs = Float(iotype='out', desc='Output BOS cost elements')

@implement_base(BaseBOSCostModel)
class FullBOSCostModel(Assembly):
	  
    # outputs
    bos_breakdown = VarTree(FullBOSVarTree(), iotype='out', desc='BOS cost breakdown')
    bos_costs = Float(iotype='out', desc='Output BOS cost elements')	  

def configure_full_bos(assembly):
    """ Full configure method for a balance of station cost assembly for coupling models to get a full wind plant balance of station cost estimate.  It replaces the extended or base balance of station cost aggregator component with an full version which contains the full balance of station variable tree.    """

    configure_base_bos(assembly)
    
    assembly.replace('bos', FullBOSCostAggregator())

    assembly.connect('bos.bos_breakdown','bos_breakdown')


# CAPEX Variable Trees, Components and Assemblies
@base
class BaseCAPEXAggregator(Component):

    # Inputs
    turbine_cost = Float(iotype='in', desc = 'A Wind Turbine Capital _cost')
    bos_costs = Float(iotype='in', desc='A Wind Plant Balance of Station _cost Model')
    
    # Parameters
    turbine_number = Int(iotype = 'in', desc = 'number of turbines at plant')

    # Outputs
    capex = Float(iotype='out', desc='Overall wind plant capital expenditures including turbine and balance of station costs') 

def configure_capex(assembly):

    # To be replaced by actual models        
    assembly.add('tcc',BaseTurbineCostModel())
    assembly.add('bos',BaseBOSCostModel())
    assembly.add('cap_ex',BaseCAPEXAggregator())

    assembly.driver.workflow.add(['tcc','bos','capex'])

    assembly.connect('turbine_number', ['bos.turbine_number', 'cap_ex.turbine_number'])     
    assembly.connect('tcc.turbine_cost', 'cap_ex.turbine_cost')
    assembly.connect('bos.bos_costs', 'cap_ex.bos_costs')
    
    assembly.connect('cap_ex.capex','capex')