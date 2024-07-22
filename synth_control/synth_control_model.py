from typing import Optional, Union
import numpy as np
from scipy.optimize import minimize

# plot function
# cross-validation
# conformal prediction
# augmented method
# unit tests
# documentation

class SynthControlModel:
    '''
    A class to optimize the weights of control units to construct a synthetic control.
    
    Attributes:
        control_units (np.ndarray): The control units used to construct the synthetic control.
        treated_units (np.ndarray): The treated unit which we want to compare against the synthetic control.
        pre_intervention_indices (Union[np.ndarray, slice]): Indices or slice for the pre-intervention period.
        post_intervention_indices (Union[np.ndarray, slice]): Indices or slice for the post-intervention period.        
        optimal_weights (np.ndarray): The optimal weights for the control units.  
        final_loss (float): The final mse loss using the optimal weights.
        final_synth_control (np.ndarray): The final synthetic control for the full series (pre and post-intervention)
    '''
    
    def __init__(self, 
                 control_units: np.ndarray, 
                 treated_unit: np.ndarray, 
                 pre_intervention_indices: Union[np.ndarray, slice],
                 post_intervention_indices: Union[np.ndarray, slice]):
        '''
        Initializes the optimizer.

        Args:
            control_units (np.ndarray): The control units used to construct the synthetic control.
            treated_units (np.ndarray): The treated unit which we want to compare against the synthetic control.
            pre_intervention_indices (Union[np.ndarray, slice]): Indices for the pre-intervention period.
        '''
        self.control_units = control_units
        self.treated_unit = treated_unit
        self.pre_intervention_indices = pre_intervention_indices
        self.post_intervention_indices = post_intervention_indices
        self.optimal_weights: Optional[np.ndarray] = None
        self.final_loss: Optional[float] = None
        self.final_synth_control: Optional[np.ndarray] = None

    def calculate_synth_control(self, weights: np.ndarray, control_units: np.ndarray) -> float:
        '''
        Calculates the synthetic control.

        Args:
            weights (np.ndarray): The weights applied to the control units.
            control_units (np.ndarray): The control units used to construct the synthetic control.

        Returns:
            float: The calculated synthetic control.
        '''
        synthetic_control = np.dot(control_units, weights)
        return synthetic_control

    def calculate_loss(self, weights: np.ndarray, control_units: np.ndarray, treated_unit: np.ndarray) -> float:
        '''
        Calculates the mean squared error (MSE) between the treated unit and the synthetic control.

        Args:
            weights (np.ndarray): The weights applied to the control units.
            control_units (np.ndarray): The control units used to construct the synthetic control.
            treated_unit (np.ndarray): The treated unit which we want to compare against the synthetic control.

        Returns:
            float: The MSE between the treated unit and the synthetic control.
        '''
        synthetic_control = self.calculate_synth_control(weights, control_units)
        mse = np.sqrt(np.sum((treated_unit - synthetic_control)**2))
        return mse

    def optimise_weights(self) -> np.ndarray:
        '''
        Optimizes the weights of the control units to minimize the loss function.

        Returns:
            np.ndarray: The optimal weights for the control units.
        '''
        num_control_units = self.control_units.shape[1]
        initial_weights = np.ones(num_control_units) / num_control_units
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(num_control_units)]

        result = minimize(self.calculate_loss,
                          initial_weights,
                          args=(self.control_units[self.pre_intervention_indices], self.treated_unit[self.pre_intervention_indices]),
                          method='SLSQP', 
                          bounds=bounds, 
                          constraints=constraints,
                          options={'disp': False, 'maxiter': 1000, 'ftol': 1e-9},
        )
        
        if not result.success:
            raise ValueError("Optimization did not converge. Please check the input data and constraints.")
    
        self.optimal_weights = result.x
        return self.optimal_weights

    def calcuate_final_synth_control(self) -> float:
        '''
        Calculates the final synthetic control for the full series (pre and post-intervention)

        Returns:
            float: The synthetic control
        '''
        if self.optimal_weights is None:
            raise ValueError("Optimal weights have not been calculated. Call optimise_weights() first.")
        
        self.final_synth_control = self.calculate_synth_control(self.optimal_weights, self.control_units)     
        return self.final_synth_control
    
    def calculate_final_loss(self) -> float:
        '''
        Calculates the final mse loss using the optimal weights.

        Returns:
            float: The final mse loss.
        '''
        if self.optimal_weights is None:
            raise ValueError("Optimal weights have not been calculated. Call optimise_weights() first.")
        
        self.final_loss = self.calculate_loss(self.optimal_weights, self.control_units[self.pre_intervention_indices], self.treated_unit[self.pre_intervention_indices])
        return self.final_loss

    def causal_impact(self) -> float:
        '''
        Calculate the causal impact in the post-intervention period.

        Returns:
            float: The causal impact in the post-intervention period.
        '''
        if self.final_synth_control is None:
            raise ValueError("Synthetic control has not been calculated. Call calcuate_final_synth_control() first.")       

        return np.sum(self.treated_unit[self.post_intervention_indices] - self.final_synth_control[self.post_intervention_indices])