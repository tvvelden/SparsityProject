% This script generates the results of solving the problem using the greedy
% strategy Orthogonal Matching Pursuit (OMP
%

%% Initialize variables
clear all

load("Sparse_Low_Rank_dataset.mat")

N = size(H,1); % Every matrix in H is of shape N x N
numMatrices = size(H,3); % Number of matrices in H

U = dftmtx(N); % Create sparsifying 2D-DFT matrix of size N x N

% As testing showed that OMP can reconstruct the matrix with very few
% measurements, the larger amounts will not be shown
maxSamples = 100;
sampleList = 1:maxSamples;

% Init results matrices
diffList = zeros(numel(sampleList),numMatrices);
mList = zeros(numel(sampleList),1);


% As the results will be shown in a plot directly, this plot has to be
% created on forehand.
figure(10)
clf;
yyaxis left
barplot = bar(sampleList,zeros(length(sampleList), 1));
ylabel("$\frac{||\hat{H} - H||_F}{||H||_F}$", 'interpreter','latex', 'FontSize',20)

yyaxis right
timeplot = plot(sampleList, ones(length(sampleList), 1)*nan, 'LineWidth', 2);
ylabel("Averaged computation time [s]")
xlabel("Amount of samples taken")

title("Reconstruction error using OMP")
pause(0.000001) % This gives matlab the time to show the plot

% This is used so one can plot intermediate results when using parfor
D = parallel.pool.DataQueue;
D.afterEach(@(x) updateBarplot(x(1), x(2), x(3), barplot, timeplot));

%% Start testing
parfor samplingIndex = 1:numel(sampleList)
    diffListForCurrentSampling = zeros(1, numMatrices);                 
    
    % Initialize the CS matrix
    samplingPercentage = sampleList(samplingIndex);
    sampleIndices = transpose(randperm(1024,sampleList(samplingIndex)));
    m = length(sampleIndices);
    A = zeros(m, N*N);
    for index = 1:length(sampleIndices)
        A(index, sampleIndices(index)) = 1;
    end
    
    OMP_A = A*kron(transpose(U),U');

    mList(samplingIndex) = m; % used for plotting
    
    tic()
    for Hiterator = 1:numMatrices
        % Create measurements
        trueH = H(:,:,Hiterator);
        y = trueH(sampleIndices);
        
        % Use OMP to solve
        theta = OMP(OMP_A,y,1e-2);
        
        % Go back to the non-sparse domain
        OMP_x = U' * reshape(theta, [32,32]) * U;

        % Calculate error
        diffListForCurrentSampling(Hiterator) = norm(OMP_x - trueH,'fro')/norm(trueH,'fro');
    end

    % Collect all answers in one matrix
    diffList(samplingIndex, :) = diffListForCurrentSampling;
    finalTime = toc()/numMatrices; % calculate averaged end time

    % Send back to host for plotting when finished
    send(D, [samplingIndex, mean(diffListForCurrentSampling), finalTime])
    
    disp(['Finished sample num ', num2str(m), ' in ', num2str(finalTime), ' seconds'])
end

%% Used to update the plot in real time
function updateBarplot(samplingIndex, errorMean, finalTime, barPlot, timePlot)
    barPlot.YData(samplingIndex) = errorMean;
    timePlot.YData(samplingIndex) = finalTime;
    drawnow('limitrate')
end