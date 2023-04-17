import torch
from torch import nn

class Density(torch.jit.ScriptModule):
    
    def __init__(self, t, s):
        """Linear spline for density approximation.
        
        Args:
            t -- spline knots, tensor of shape (..., n_knots)
            s -- knot density values, tensor of shape (..., n_knots)
        """
        super().__init__()
        self.t = t
        self.s = s
        self.integrals = self._get_partial_integrals()

            
    def update_spline_parameters(self, t, s):
        """Replace grid t and density s. Allows avoiding jit recompilation."""
        self.t = t
        self.s = s
        self.integrals = self._get_partial_integrals()
        
        
    @staticmethod
    def _get_bin_values(values, indices):
        """Auxilliary function used in piecewise functions."""
        return torch.gather(values, -1, indices)

    
    def _get_partial_integrals(self,):
        """Integrate the spline from t_n to the right boundary of a each bin."""
        bin_integral = 0.5 * (self.s[..., 1:] + self.s[..., :-1]) * torch.diff(self.t)
        return torch.cumsum(bin_integral, dim=-1)
    
    
    def _get_bin_indices(self, t):
        """Return a indices of bins containing t."""
        assert(self.t.shape[:-1] == t.shape[:-1])
        indices = torch.searchsorted(self.t, t)
        indices = torch.clamp(indices, min=1)
        return indices

    
    def forward(self, t):
        """Compute linear spline at $t$
        
        Args:
            t -- spline arguments of shape (..., m)
        """
        indices = self._get_bin_indices(t)
        t_left = self._get_bin_values(self.t, indices - 1)
        t_right = self._get_bin_values(self.t, indices)
        alpha = (t_right - t) / (t_right - t_left)
        s_left = self._get_bin_values(self.s, indices - 1)
        s_right = self._get_bin_values(self.s, indices)  
        return alpha * (s_left - s_right) + s_right
        
    
    def _sigma_int(self, t):
        """ Compute spline integral $\int_{t_0}^t \sigma(s) \mathrm d s$.
        
        Args:
            t -- spline arguments of shape (..., m)
        """
        indices = self._get_bin_indices(t)
        full_part = self._get_bin_values(self.integrals, indices - 1)
        
        t_right = self._get_bin_values(self.t, indices)
        s_right = self._get_bin_values(self.s, indices)
        s = self.forward(t)
        curr_part = 0.5 * (s + s_right) * (t - t_right)
        return full_part + curr_part
    
    @torch.jit.script_method
    def _sigma_int_bin_inv(self, t_left, t_right, s_left, s_right, full_part, y):
        """Find inverse of $\int_{t_0}^t \sigma(s) \mathrm d s$ inside a bin"""
        # equation to solve:
        # a = (t_r - t) / (t_r - t_l)
        # y = 0.5 (a * s_l + (1 - a) * s_r + s_r) * (t - t_r) + full_part
        # 0 = 0.5 * (s_r - s_l) * (t_r - t_l) * a ** 2 - s_r * (t_r - t_l) * a + (full_part - y)
        # 0 = 0.5 * (s_r - s_l) * dt ** 2 - s_r * (t_r - t_l) * dt + (full_part - y) * (t_r - t_l)
        a = (s_right - s_left)
        b = s_right * (t_right - t_left)
        c = 2 * (full_part - y) * (t_right - t_left)
        root = -c / (b + torch.sqrt((b ** 2 - a * c).abs() + 1e-12))
        return root
        
    
    def _sigma_int_inv(self, y):
        """Compute the inverse of sigma integral."""
        # find the bin containing the inverse
        indices = torch.searchsorted(self.integrals, y)
        indices = torch.clamp(indices, max=self.integrals.shape[-1] - 1)
        # find the inverse within the bin
        t_left = self._get_bin_values(self.t, indices)
        t_right = self._get_bin_values(self.t, indices + 1)
        s_left = self._get_bin_values(self.s, indices)
        s_right = self._get_bin_values(self.s, indices + 1)
        full_part = self._get_bin_values(self.integrals, indices)
        bin_root = self._sigma_int_bin_inv(t_left,
                                           t_right,
                                           s_left,
                                           s_right,
                                           full_part,
                                           y)
        result = t_right + bin_root
        #return torch.minimum(torch.maximum(result, t_left), t_right)
        return result

    
    @torch.jit.script_method
    def _prepare_inv_int_input(self, u):
        """Compute -(log(1 + y_f u)) used in the inverse"""
        sigma_int_far = self.integrals[..., -1:]
        # truncated sample
        lse_input = torch.stack([-sigma_int_far + u.log(), (-u).log1p()], dim=-1)
        inv_args = -(torch.logsumexp(lse_input, dim=-1))
        inv_args = torch.where(sigma_int_far < 1e-6, u * sigma_int_far, inv_args)
        return inv_args
    
    def inv_cdf(self, u):
        """Compute the inverse cdf at $u$.
        
        We compute the inverse of the spline integral _sigma_int(t) rather
        than the opacity F(t) = 1 - exp(-_sigma_int(t)) to achieve better
        numerical stability.
        
        Args:
            y -- inverse function arguments of shape (..., m)
            n_iter -- number of iterations used in binary search
            eps -- hyperparameter used for gradient numerical stability
        """
        inv_args = self._prepare_inv_int_input(u)
        t = self._sigma_int_inv(inv_args)
        return t
        
    
    def transparency(self, t):
        """Compute the transparency function at $t$."""
        sigma_int = self._sigma_int(t)
        return torch.exp(-sigma_int)

    
    def opacity(self, t):
        """Compute the opacity function at $t$."""
        sigma_int = self._sigma_int(t)
        return -torch.expm1(-sigma_int)


    def pdf(self, t):
        """Return the probability density function induced by spline"""
        return self.forward(t) * self.transparency(t)
            
        
    def t_min(self):
        """Return the nearest ray point."""
        return self.t[..., :1]

    
    def t_max(self):
        """Return the farthest ray point."""
        return self.t[..., -1:]
    
    
    def y_min(self):
        """Return the infimum of opacity"""
        return torch.zeros_like(self.t[..., :1])

    @torch.jit.script_method
    def y_max(self):
        """Return the supremum of opacity"""
        return -torch.expm1(-self.integrals[..., -1:])
