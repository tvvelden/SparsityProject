% This script solves the problem using nuclear norm minimization in CVX.
% This script will do it for a fixed amount of samples specified by the
% number 'SampleAmount'
%
% This script will require the CVX package to work.
%% Initialization
clear all

load("Sparse_Low_Rank_dataset.mat")

N = size(H,1); % Every matrix in H is of shape N x N
numMatrices = size(H,3); % Number of matrices in H
% numMatrices = 100;

U = dftmtx(N); % Create sparsifying 2D-DFT matrix of size N x N

diff = zeros(numMatrices,1);

f = waitbar(0,'Start with solving');

SampleAmount = 800;

%% 
% Create the CS matrix
sampleIndices = transpose(randperm(1024,SampleAmount));
m = length(sampleIndices);
A = zeros(1, N*N);
for index = 1:length(sampleIndices)
    A(1, sampleIndices(index)) = 1;
end

% Transform the CS matrix so it can be used in the hadamard product
mask = reshape(A,[32,32]);

%% Start minimization
tic()
for i = 1:numMatrices
    % Retrieve the current H from the large dataset
    matH = reshape(H(:,:,i),N,N);
    
    % Only keep the measurements, set all other values to 0
    sparseM = matH.*mask;
    
    % Solve using CVX
    phi = normNucMinimization(sparseM,32,mask);
    
    % Store the error
    diff(i) = norm(phi-matH,'fro')/norm(matH,'fro');
    
    % Update waitbar to see how far the code is
    if mod(i,numMatrices/100) == 0
        waitbar(i/numMatrices,f,"Solving matrix("+i+"/"++")")
    end
end

% store the amount of time it took for all matrices to be solved
t = toc();
close(f)