% This script will solve the problem using projected gradient descent.
% The projection is achieved by projecting the solution onto a low rank
% matrix subspace of rank rankR = 16
%% Initialization
clear all
clc

rankR = 16;
load("Sparse_Low_Rank_dataset.mat")

N = size(H,1); % Every matrix in H is of shape N x N
numMatrices = size(H,3); % Number of matrices in H

U = dftmtx(N); % Create sparsifying 2D-DFT matrix of size N x N
maxGDIter = 100; % maximum amount of iteration for gradient descent 
eta = 0.001; % stepsize for gradient descent

maxSamples = 300;
sampleList = ceil(linspace(1, maxSamples, 15));


numMatrices = 50; % As this method is rather slow, less matrices are used to increase testing speed
diffList = zeros(numel(sampleList),numMatrices);
diffListBad = zeros(numel(sampleList),numMatrices);
timeList = zeros(numel(sampleList),1);
mList = zeros(numel(sampleList),1);

%% Start testing
for samplingIndex = 1:numel(sampleList)
    % Generate CS matrix
    sampledAmount = sampleList(samplingIndex);
    sampleIndices = randperm(1024,sampledAmount);
    m = length(sampleIndices);
    mList(samplingIndex) = m;

    A = zeros(m, N*N);
    for index = 1:length(sampleIndices)
        A(index, sampleIndices(index)) = 1;
    end
    ATilde = A*kron(transpose(U), U');
    
    tic();
    for Hiterator = 1:numMatrices
        % Generate measurements
        trueH = H(:,:,Hiterator);
        y = trueH(sampleIndices);
        y = y(:);
        

        % Start gradient descent
        Xhat(:,:,1) = zeros(N,N);
        for GDIter = 1 : maxGDIter
            n = GDIter;
            %f1 = ||y-Ax||^2
            %d/dx f1 = -2A^H y + A^H A x
            f0temp = (y-ATilde*reshape(Xhat(:,:,n), [N*N,1]));
            f0(n) = f0temp'*f0temp;
            
            % Find the gradient (in this special case there is an analystic solution)
            gradF1 = reshape(2*ATilde' * (ATilde * reshape(Xhat(:,:,n), [N*N,1]) - y), [N,N]);
            
            % Gradient descent step
            Xhat(:,:,n+1) = Xhat(:,:,n) - eta*gradF1;
        
            % Project on the subspace with low rank matrices
            [svdU, svdS, svdV] = svd(Xhat(:,:,n+1), "vector");
            svdS(rankR:end) = 0;
            svdS = diag(svdS);
            Xhat(:,:,n+1) = svdU * svdS * svdV';
        end
        finalX = Xhat(:,:,end);
        Hhat = U' *finalX * U;
        diffListBad(samplingIndex, Hiterator) = norm(Hhat - trueH,'fro')/norm(trueH,'fro');
        % As the solution of this gradient descent often gave the support
        % of the solution, but wrongly scaled; this solution will be used
        % to find a better solution using the now (often) overdetermined 
        % system of equations.

        % find the support of X
        sortedX = sort(reshape(abs(finalX), [N*N,1]), "descend");
        maxId = findElbow(sortedX);
        support = find(abs(finalX) >= sortedX(maxId));
        
        % Create the overdetermined system of equations and solve using
        % pseudo-inverse
        truncA = ATilde(:, support);
        xHat = zeros(N*N,1);
        xHat(support) = pinv(truncA)*y;
        xHat = reshape(xHat, [N,N]);
        
        % transform to the non-sparse domain
        Hhat = U' *xHat * U;

        % store the error
        diffList(samplingIndex, Hiterator) = norm(Hhat - trueH,'fro')/norm(trueH,'fro');
    end
    finalTime = toc();
    timeList(samplingIndex) = finalTime / numMatrices;
    disp("PGD: Final time (avg) for " +m+ " samples is " + finalTime)
end

%% Plot the results
%%
figure(10)
clf;
yyaxis left
barplot = bar(sampleList, mean(diffList,2));
ylabel("$\frac{||\hat{H} - H||_F}{||H||_F}$", 'interpreter','latex', 'FontSize',20)

yyaxis right
timeplot = plot(sampleList, timeList, 'LineWidth', 2);
ylabel("Averaged computation time [s]")
xlabel("Amount of samples taken")

title("Reconstruction error using PGD")

figure(11)
clf;
yyaxis left
barplot = bar(sampleList, mean(diffListBad,2));
ylabel("$\frac{||\hat{H} - H||_F}{||H||_F}$", 'interpreter','latex', 'FontSize',20)

yyaxis right
timeplot = plot(sampleList, timeList, 'LineWidth', 2);
ylabel("Averaged computation time [s]")
xlabel("Amount of samples taken")

title("Reconstruction error using PGD without finding its support and solving for that support")