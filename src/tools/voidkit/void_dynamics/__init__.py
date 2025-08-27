# Expose void_dynamics API for simple imports
from .FUM_Void_Equations import universal_void_dynamics
from .FUM_Void_Debt_Modulation import VoidDebtModulation

__all__ = ["universal_void_dynamics", "VoidDebtModulation"]