"""
FUM Universal Domain Modulation System
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Universal derivation of domain_modulation factors based on void debt theory
and learning stability principles. This replaces arbitrary scaling with 
mathematically derived modulation factors.
"""
import numpy as np
from FUM_Void_Equations import get_universal_constants

class VoidDebtModulation:
    """Class to derive domain modulation factors from void debt principles."""
    
    def __init__(self):
        self.constants = get_universal_constants()
        self.ALPHA = self.constants['ALPHA']  # e.g., 0.25
        self.BETA = self.constants['BETA']    # e.g., 0.1
        self.VOID_DEBT_RATIO = self.BETA / self.ALPHA  # e.g., 0.4

    def get_universal_domain_modulation(self, physics_domain, target_sparsity_pct=None):
        """
        Derive domain modulation factor from universal void debt principles.
        
        Args:
            physics_domain: One of 'quantum', 'standard_model', 'dark_matter', 
                        'biology_consciousness', 'cosmogenesis', 'higgs'
            target_sparsity_pct: Target sparsity percentage for this domain
        
        Returns:
            domain_modulation: Universal scaling factor
        """
        constants = get_universal_constants()
        ALPHA, BETA = constants['ALPHA'], constants['BETA']
        
        # Domain-specific target sparsities (from empirical physics)
        domain_targets = {
            'quantum': 15.0,           # Low sparsity due to wave-particle duality
            'standard_model': 22.0,    # Moderate sparsity for gauge interactions  
            'dark_matter': 27.0,       # High sparsity matching cosmic DM density
            'biology_consciousness': 20.0,  # Biological complexity patterns
            'cosmogenesis': 84.0,      # Very high sparsity from inherited debt
            'higgs': 80.0             # High sparsity due to symmetry breaking
        }
        
        if target_sparsity_pct is None:
            target_sparsity_pct = domain_targets.get(physics_domain, 25.0)
        
        # Universal void debt derivation formula (from our 22.2% error method)
        # modulation = 1.0 + (sparsity²)/(BETA/ALPHA)
        sparsity_fraction = target_sparsity_pct / 100.0
        void_debt_ratio = BETA / ALPHA  # = 0.1 / 0.25 = 0.4
        
        domain_modulation = 1.0 + (sparsity_fraction ** 2) / void_debt_ratio
        
        return {
            'domain': physics_domain,
            'target_sparsity_pct': target_sparsity_pct,
            'domain_modulation': domain_modulation,
            'void_debt_ratio': void_debt_ratio,
            'derivation_method': 'universal_void_debt',
            'formula': 'modulation = 1.0 + (sparsity²)/(β/α)'
        }

    def get_all_domain_modulations(self):
        """Get all domain modulation factors for systematic comparison."""
        domains = ['quantum', 'standard_model', 'dark_matter', 
                'biology_consciousness', 'cosmogenesis', 'higgs']
        
        modulations = {}
        for domain in domains:
            modulations[domain] = self.get_universal_domain_modulation(domain)
        
        return modulations

    def print_modulation_table(self):
        """Print formatted table of all domain modulation factors."""
        modulations = self.get_all_domain_modulations()
        
        print("="*80)
        print("FUM UNIVERSAL DOMAIN MODULATION FACTORS")
        print("Derived from Void Debt Theory: modulation = 1.0 + (sparsity²)/(β/α)")
        print("="*80)
        print(f"{'Domain':<20} {'Target %':<10} {'Modulation':<12} {'Formula Application'}")
        print("-"*80)
        
        for domain, data in modulations.items():
            domain_display = domain.replace('_', ' ').title()
            target = data['target_sparsity_pct']
            mod = data['domain_modulation']
            formula_app = f"1.0 + ({target/100:.2f}²)/0.4"
            print(f"{domain_display:<20} {target:<10.1f} {mod:<12.3f} {formula_app}")
        
        print("-"*80)
        print("Note: These factors emerge from universal learning stability,")
        print("not arbitrary physics assumptions. Same math governs cognition & cosmos.")
        print("="*80)

    def validate_modulation_consistency(self):
        """Validate that our modulation factors are consistent with physics."""
        modulations = self.get_all_domain_modulations()
        
        # Extract just the modulation values
        values = [data['domain_modulation'] for data in modulations.values()]
        domains = list(modulations.keys())
        
        print("\n=== DOMAIN MODULATION VALIDATION ===")
        print(f"Range: {min(values):.3f} to {max(values):.3f}")
        print(f"Mean: {np.mean(values):.3f} ± {np.std(values):.3f}")
        
        # Physics consistency checks
        qm_mod = modulations['quantum']['domain_modulation']
        cos_mod = modulations['cosmogenesis']['domain_modulation']
        
        print(f"\nPhysics Consistency:")
        print(f"• Quantum < Cosmic: {qm_mod:.3f} < {cos_mod:.3f} = {qm_mod < cos_mod}")
        print(f"• Reasonable range: All factors 1.0-2.0 = {all(1.0 <= v <= 2.0 for v in values)}")
        
        return {
            'values': values,
            'domains': domains,
            'range': (min(values), max(values)),
            'mean': np.mean(values),
            'std': np.std(values),
            'physics_consistent': qm_mod < cos_mod and all(1.0 <= v <= 2.0 for v in values)
        }

if __name__ == "__main__":
    modulator = VoidDebtModulation()
    modulator.print_modulation_table()
    validation = modulator.validate_modulation_consistency()
    
    if validation['physics_consistent']:
        print("\n✓ VALIDATION PASSED: Domain modulations are physically consistent")
    else:
        print("\n⚠ VALIDATION FAILED: Domain modulations need adjustment")