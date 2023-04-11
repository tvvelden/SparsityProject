% This script will generate the results of the sparse matrix reconstruction
% using the Parallel ProXimal Algorithm PPXA

%% Initialization of matrices
clear all

load("Sparse_Low_Rank_dataset.mat")

H = gpuArray(H);

N = size(H,1); % Every matrix in H is of shape N x N
numMatrices = size(H,3); % Number of matrices in H
% numMatrices = 10; % This can be used for speeding up testing

U = dftmtx(N); % Create sparsifying 2D-DFT matrix of size N x N

% Define how many samples will be used
maxSamples = 800;
sampleList = 100:100:maxSamples;

% initialize the eventual storage for the metrics
diffList = zeros(numel(sampleList),numMatrices);
mList = zeros(numel(sampleList),1);

% Start the plot in which the parfor workers will plot their results
% This plot will also be the final result shown in the report
figure(12)
clf;
yyaxis left
barplot = bar(sampleList,zeros(length(sampleList), 1));
ylabel("$||\hat{H} - H||_F$", 'interpreter','latex')

yyaxis right
timeplot = plot(sampleList, ones(length(sampleList), 1)*nan, 'LineWidth', 2);
ylabel("Averaged computation time [s]")
xlabel("Amount of samples taken")

title("Reconstruction error using OMP")
pause(0.000001)

% This is used so parfor can update the plot concurrently
D = parallel.pool.DataQueue;
D.afterEach(@(x) updateBarplot(x(1), x(2), x(3), barplot, timeplot));


parfor samplingIndex = 1:numel(sampleList)
    % The errors must be stored for parfor workers seperately before they
    % can be combined
    diffListForCurrentSampling = zeros(1, numMatrices);                 
    
    % Generate CS
    sampleIndices = transpose(randperm(1024,sampleList(samplingIndex)));
    m = length(sampleIndices);
    A = zeros(m, N*N);
    for index = 1:length(sampleIndices)
        A(index, sampleIndices(index)) = 1;
    end
    
    CS_A = A*kron(transpose(U),U');

    mList(samplingIndex) = m;

    
    tic() % Start timer, this timer will calculate the time it takes for all matrices to be solved
    for Hiterator = 1:numMatrices
        % Generate the measurements
        trueH = H(:,:,Hiterator);
        y = trueH(sampleIndices);
        
        % Use PPXA
        Xhat = PPXA(y,CS_A,0.3,rand(32));
        
        % Transform back to non-sparse domain
        Hhat = U' * reshape(Xhat, [32,32]) * U;

        % Calculate the error
        diffListForCurrentSampling(Hiterator) = norm(Hhat - trueH,'fro')/norm(trueH,'fro');
    end
    % Store the errors of the parfor workers in the larger datastructure
    diffList(samplingIndex, :) = diffListForCurrentSampling;

    finalTime = toc()/numMatrices; % Stop the timer and average it
    
    % Send intermediate result to the host, allowing for direct plotting
    send(D, [samplingIndex, mean(diffListForCurrentSampling), finalTime])
    
    disp(['Finished sample num ', num2str(m), ' in ', num2str(finalTime), ' seconds'])
end

%% Function used to update the plot in real-time
function updateBarplot(samplingIndex, errorMean, finalTime, barPlot, timePlot)
    barPlot.YData(samplingIndex) = errorMean;
    timePlot.YData(samplingIndex) = finalTime;
    drawnow('limitrate')
end