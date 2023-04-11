function X = normNucMinimization(H,N,mask)
% NORMNUCMINIMIZATION CVX optimization algorithm to minimize the nuclear
% norm of X given the samples of H specified by the sampling mask 
% This function will require the CVX package to work.
%
% Inputs:
%   - H     {Matrix}:   contains the true values of H of which measurements
%                       will be taken
%   - N     {Number}:   The size of H and X where both are complex matrices
%                       of N X N
%   - mask  {Matrix}:   sampling mask
%
% Output:
%   - X     {Matrix}:   Estimation of H based on nuclear norm minimization
%                       with the knowledge of a few samples



cvx_quiet('True')
cvx_begin
    variable X(N, N) complex;
    minimize norm_nuc(X);
    subject to
        X.*mask == H.*mask;
cvx_end
cvx_quiet('False')

end