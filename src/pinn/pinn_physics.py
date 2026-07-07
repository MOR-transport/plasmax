#phase 4 : Creation of the Physical Residue

import jax
import jax.numpy as jnp

from scimba_jax.physical_models.abstract_physical_model import AbstractPhysicalModel
from scimba_jax.physical_models.abstract_residuals import InteriorResidual

class VlasovResidual(InteriorResidual):
    """Physical residual for the 1D Vlasov equation"""
    def __init__(self, domain, time_domain):
        super().__init__(
            domain=domain,
            time_domain=time_domain,
            size=1,
            model_type="t_x_v",
        )
        
    def construct_residual(self, f, precomputed: dict | None = None):
        """Construct the equation : df/dt + df/dx - E * df/dv = 0"""
        df_dt = f.d_t()
        df_dx = f.partial_derivative_x(0) #0 for the 1st spatial dimension
        df_dv = f.partial_derivative_v(0) #0 for the 1st velocity dimension
        
        if precomputed is None:
            return df_dt + df_dx - df_dv
        
        #Recovery from precomputation_without_diff
        E = precomputed["E"]
        v = precomputed["v"] #We will also pass the v coordinate here to simplify
        
        return df_dt + v * df_dx - E * df_dv
    

class VlasovPoissonModel(AbstractPhysicalModel):
    """Global physical model orchestrating the Vlasov residue and electric field integration"""
    def __init__(self, main_domain, time_domain, E_interpolator):
        super().__init__(main_domain=main_domain, time_domain=time_domain)
        self.E_interpolator = E_interpolator 
        
        #declaration of the physical residual to be evaluated at collocation points within the domain
        self.physical_residuals = {
            "interior": VlasovResidual(domain=main_domain, time_domain=time_domain)
        }
        
        # assimilation data (L_data) will be added dynamically via add_data_residual
        self.data_residuals = {}
        
    def pre_computation_without_diff(self, space, sample_dict: dict) -> dict:
        """
        Precomputes E(x,t) on collocation points before network evaluation. 
        Anything returned here is protected by jax.lax.stop_gradient
        """
        precomputed = {}
        #"interior" corresponds to points drawn randomly in t x \Omega_x x \Omega_v" 
        if "interior" in sample_dict: 
            #the order corresponds to the model_type="t_x_v" in the VlasovResidual
            t_colloc, x_colloc, v_colloc = sample_dict["interior"]
        
            #call to the interpolation function from PlasmaX (phase 2)
            E_val = self.E_interpolator(x_colloc, t_colloc)
            
            precomputed["E"] = E_val
            precomputed["v"] = v_colloc
            
        return precomputed
            
        