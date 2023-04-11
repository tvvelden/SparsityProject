function z = OMP(A,y,error)
%OMP Perform the Orthogonal Matching Pursuit Algorithm to recover sparse
%vector z
% Inputs:
%        A (Matrix): The compressed sensing matrix
%        y (vector): Vector containing m measurements
%        error (scalar): Error threshold for the OMP loop
% Outputs:
%        z (vector): Recovered sparse vector z

    % Prepare data
    % Create empty z vector of size (32*32)x1
    % Create empty set Lambda
    z = zeros(length(A),1);
    lambda = [];

    % Make the residual equal to y
    residual = y;

    % Create iteration threshold to make sure loop does not run endlessly
    MaxIter = length(y);
    currIter = 1;
    
    % Create empty h beforehand
    h = zeros(length(y),1);

    % Transpose A beforehand so it does not need to be calculated every
    % loop
    herA = A';
    
    while currIter < MaxIter && norm(residual) > error
        
        % Calculate new h
        h = herA*residual;

        % Find largest row such that z will now also have nonzero value in
        % that row
        [~,k] = max(abs(h));

        % Add the index k of the kth row where z is allowed to be nonzero
        lambda = [lambda, k];

        % Find z values through pseudo inverse
        z(lambda) = (A(:,lambda)'*A(:,lambda))^-1*A(:,lambda)'*y;

        % Find new residual
        residual = y - A*z;

        % Increase iterator
        currIter = currIter + 1;
    end
end

