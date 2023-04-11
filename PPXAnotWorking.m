% This script uses the PPXA algorithm described in the report.
%
%

clear all
clc
load("Sparse_Low_Rank_dataset.mat")


N = size(H, 2);
U = dftmtx(N);

amountOfMatrices = size(H, 3); % The amount of matrices over which we will average
amountOfMatrices = 10;

sampledAmount = 150;

rankR = 10;


%
maxSamples = 200;
sampleList = ceil(linspace(1,maxSamples, 10));

% Init results matrices
diffList = zeros(numel(sampleList),amountOfMatrices);
timeList = zeros(numel(sampleList),1);
mList = zeros(numel(sampleList),1);


maxPPAIteration = 100; % maximum amount of iteration allowed for PPXA algorithm

% Regularization variables for the proximal operators
omega = 0;
lambda = 0.01;
epsilon = 0.1;

for samplingIndex = 1 : length(sampleList)
    % Take measurements
    sampleIndices = randperm(1024,sampleList(samplingIndex));
    m = length(sampleIndices);
    samplingMatrix = zeros(N,N);
    samplingMatrix(sampleIndices) = 1;
    
    % Create the A matrix
    AMatrix = zeros(m, N*N);
    for index = 1:length(sampleIndices)
        AMatrix(index, sampleIndices(index)) = 1;
    end
    
    ATilde = AMatrix*kron(transpose(U),U');
    
    tic();
    for matrixIndex = 1 : amountOfMatrices
        % Initialize the matrix to take measurements from
        trueH = H(:,:,matrixIndex);
        
        y = trueH(sampleIndices);
        y = y(:);
        
    
        yMatrix = trueH .* samplingMatrix;
        
        noConvergence = true;
        PPAIteration = 0;
        X(:,:,1) = U*yMatrix*U';
        Gamma1(:,:,1) = X(:,:,1);
        Gamma2(:,:,1) = X(:,:,1);
        Gamma3(:,:,1) = X(:,:,1);
        while noConvergence
            PPAIteration = PPAIteration + 1;
            n = PPAIteration; %n is better readible for the next code
            
            % prox for f_1 = i_B_*
            [U1, S1, V1] = svd(Gamma1(:, :, n), 'vector');
            S1(rankR:end) = 0;
            S1 = diag(S1);
            SigmaBar1 = exp(1i*angle(S1)).*max(0, abs(S1)-omega); % Unsure about this operator
            P1(:,:,n) = U1 * SigmaBar1 * V1';
            
            % prox for f_2 = = ||X||_2,1
            P2(:,:,n) = max(norm(Gamma2(:,:,n),2)-lambda, 0)/norm(Gamma2(:,:,n),2) * Gamma2(:,:,n);
    
            % prox for f_3 = i_B_l2
            Astar = @(x) reshape(ATilde' * y, [N, N]);
            r = y - ATilde * reshape(Gamma3(:,:,n), [N*N, 1]);
            nu = 1024; % Derived by taking A^H A and seeing what value remains
            P3(:,:,n) = Gamma3(:,:,n) + nu^-1 * Astar(r) * max(0, 1-epsilon*norm(r, 2)^-1);
            
            X(:,:,n+1) = (P1(:,:,n)+P2(:,:,n)+P3(:,:,n))./3;
            
            Gamma1(:,:,n+1) = Gamma1(:,:,n) + 2*X(:,:,n+1) - X(:,:,n) - P1(:,:,n);
            Gamma2(:,:,n+1) = Gamma2(:,:,n) + 2*X(:,:,n+1) - X(:,:,n) - P2(:,:,n);
            Gamma3(:,:,n+1) = Gamma3(:,:,n) + 2*X(:,:,n+1) - X(:,:,n) - P3(:,:,n);
            
            % Stopping criterion, this could be further expanded to include
            % convergences
            if PPAIteration > maxPPAIteration
                break;
            end
        end
        
        finalX = X(:,:,end);
        % As the final solution often is wrongly scaled, only the support of this
        % solution will be used. HhatBad shows the 'unoptimized' solution
        HhatBad = U' *finalX * U;
        diffListBad(samplingIndex, matrixIndex) = norm(abs(trueH-HhatBad), "fro");
        %find the solution using an elbow method, which is inspired by PCA
        %analysis for clustering
        sortedX = sort(reshape(abs(finalX), [N*N,1]), "descend");
        maxId = findElbow(sortedX);
        support = find(abs(finalX) >= sortedX(ceil(maxId/2)));
        
        %most often the cardinality of the support is smaller than the amount
        %of measurements, thus one can use the pseudo inverse, since now the
        %system of equations is overdetermined (instead of underdetermined)
        truncA = ATilde(:, support);
        xHat = zeros(N*N,1);
        xHat(support) = pinv(truncA)*y;
        xHat = reshape(xHat, [N,N]);
        
        % Transform to the non-sparse domain
        Hhat = U' *xHat * U;
        
        % store the error
        diffList(samplingIndex, matrixIndex) = norm(abs(trueH-Hhat), "fro");
    end
    finalTime = toc();
    timeList(samplingIndex) = finalTime / amountOfMatrices;
    disp("PPXA: Final time (avg) for " +m+ " samples is " + finalTime)
end

%% Usefull plots for debugging, and showing results
figure(3)
clf()
subplot(2,3,1)
heatmap(abs(trueH))
title(" trueH")

subplot(2,3,2)
heatmap(abs(Hhat))
title("Estimated matrix")

subplot(2,3,3)
heatmap(abs(HhatBad))
title("HHAT BAD")

subplot(2,3,4)
heatmap(abs(U * trueH * U'))
title("True H in sparse domain")

subplot(2,3,5)
heatmap(abs(xHat))
title("Estimated H in sparse domain")

subplot(2,3,6)
heatmap(abs(U*HhatBad*U'))
title("XHAT BAD")

%%
figure(4)
heatmap(abs(U * trueH * U'))
title("Sparse representation of H")

figure(5)
heatmap(abs(U*HhatBad*U'))
title("Sparse representation of estimated H")

figure(6)
heatmap(abs(U*Hhat*U'))
title("Sparse representation of new estimation of H")

%%
% %
% figure(10)
% clf;
% sortedX = sort(reshape(abs(finalX), [N*N,1]), "descend");
% yyaxis left
% plot(sortedX(1:100), LineWidth=2)
% ylabel('$|\hat{\mathbf{X}}|$', Interpreter='latex')
% 
% yyaxis right
% sortedTrueX = sort(reshape(abs(U * trueH * U'), [N*N,1]), "descend");
% plot(sortedTrueX(1:100), LineWidth=2)
% ylabel('$|\mathbf{X}|$', Interpreter='latex')
% 
% xlabel("Index in the ordering")
% title("X values ordered by size")

%
figure(11)
clf;
yyaxis left
barplot = bar(sampleList, mean(diffList,2));
ylabel("$\frac{||\hat{H} - H||_F}{||H||_F}$", 'interpreter','latex', 'FontSize',20)

yyaxis right
timeplot = plot(sampleList, timeList, 'LineWidth', 2);
ylabel("Averaged computation time [s]")
xlabel("Amount of samples taken")

title("Reconstruction error using PPXA")

%
figure(12)
clf;
yyaxis left
barplot = bar(sampleList, mean(diffListBad,2));
ylabel("$\frac{||\hat{H} - H||_F}{||H||_F}$", 'interpreter','latex', 'FontSize',20)

yyaxis right
timeplot = plot(sampleList, timeList, 'LineWidth', 2);
ylabel("Averaged computation time [s]")
xlabel("Amount of samples taken")

title("Reconstruction error using PPXA without finding its support and solving for that support")

%%
sortedX = sort(reshape(abs(finalX), [N*N,1]), "descend");
maxId = findElbow(sortedX);