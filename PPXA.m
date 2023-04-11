function X_n = PPXA(y,A,gama,initMatrix)
% PPXA This function will calculate the results given by the Parallel
% proximal algorithm. 
%
% Inputs:
%   - y {vector}:           Measurement vector
%   - A {Matrix}:           CS matrix
%   - gama {Number}:        Regularization term for prox_f1
%   - initMatrix {Matrix}:  Matrix used for initial value of the PPXA
%                           algorithm
%
%
% Outputs:
%   - X_n

% (y,A,U,lambda,gamma,initMatrix,maxIter) % These could also be input
% arguments of the function.
    
    % (Hyper)parameter declaration
    maxIter = 1000;
    lambda = 0.8;
    rowNum = 32;
    N = 32;

    n = 0; % iterator for the PPXA loop, this will determine convergence
    
    % Initialization of variables for PPXA
    X_n(:,:) = initMatrix;
    Gamma(:,:,1) = initMatrix;
    Gamma(:,:,2) = initMatrix;
    Gamma(:,:,3) = initMatrix;
    P1 = zeros(32);
    P2 = zeros(32);
    P3 = zeros(32);
    % arg min nuc_norm(X) + l2,1 norm(X) + (y = Avec(X))
    % Turns into
    % arg min f1 + f2 + f3
    
    while n < maxIter
        % perform prox for each f_i

        % Prox f1
        [U, S, V] = svd(Gamma(:,:,1),"econ");
        Sbar = max(S-gama,0);
        P1 = U*Sbar*V';

        % Prox f2
        % each row j of Gamma is scaled with 0 or some other value
        for row = 1:rowNum
            P2(row,:) = (max(norm(Gamma(row,:,2))-lambda,0)/norm(Gamma(row,:,2))) * Gamma(row,:,2);
        end
        
        % Prox f3 = prox(y = Avec(X))
        % prox f3 = Projection onto that space
        % https://ieeexplore-ieee-org.tudelft.idm.oclc.org/abstract/document/5414555?casa_token=-jY5asgFqV4AAAAA:4n6ClfNTWXtxKH89h0pO6ydy-S4Hsm-HO2w52Rs4xgiOq-lsjRboaPf5v7v7ZYnU0uBxBxSzZQ
        % = vec(X) + pseudo_inv(A)(y-Ax)
        temp = reshape(Gamma(:,:,3),[N^2,1]) + A'*inv(A*A')*(y-A*reshape(Gamma(:,:,3),[N^2,1]));
        % temp is still a vector so turn it back into a matrix
        P3 = reshape(temp,[N,N]);
        
        % Caclulate new X
        X_n1 = (P1 + P2 + P3)./3;
        
        % Calculate new Gammas
        Gamma(:,:,1) = Gamma(:,:,1) + 2.*X_n1 - X_n - P1;
        Gamma(:,:,2) = Gamma(:,:,2) + 2.*X_n1 - X_n - P2;
        Gamma(:,:,3) = Gamma(:,:,3) + 2.*X_n1 - X_n - P3;
        
        % Update for next iteration
        X_n = X_n1;
        n = n + 1;

    end
end

