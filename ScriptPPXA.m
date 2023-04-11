%% OMP
clear all

load("Sparse_Low_Rank_dataset.mat")

H = gpuArray(H);

N = size(H,1); % Every matrix in H is of shape N x N
numMatrices = size(H,3); % Number of matrices in H
numMatrices = 10;

U = dftmtx(N); % Create sparsifying 2D-DFT matrix of size N x N

norm21 = @(A) sum(sqrt(sum(A .^ 2)));
nucnorm = @(A) norm(svd(A),1);

maxIteration = 1000;
maxSamples = 800;


sampleList = 100:100:maxSamples;

diffList = zeros(numel(sampleList),numMatrices);
mList = zeros(numel(sampleList),1);

% f = waitbar(0,'je_vader');

figure(12)
clf;
yyaxis left
barplot = bar(sampleList,zeros(length(sampleList), 1));
ylabel("$||\hat{H} - H||_F$", 'interpreter','latex')

yyaxis right
timeplot = plot(sampleList, ones(length(sampleList), 1)*nan, 'LineWidth', 2);
ylabel("Time spent on all")
xlabel("Amount of samples taken")

title("Reconstruction error using OMP")
pause(0.000001)

D = parallel.pool.DataQueue;
D.afterEach(@(x) updateBarplot(x(1), x(2), x(3), barplot, timeplot));


for samplingIndex = 1:numel(sampleList)
    diffListForCurrentSampling = zeros(1, numMatrices);                 
    samplingPercentage = sampleList(samplingIndex);

    sampleIndices = transpose(randperm(1024,sampleList(samplingIndex)));
    m = length(sampleIndices);
    A = zeros(m, N*N);
    for index = 1:length(sampleIndices)
        A(index, sampleIndices(index)) = 1;
    end
    
    OMP_A = A*kron(transpose(U),U');

    mList(samplingIndex) = m;
    tic()
    for Hiterator = 1:numMatrices
        
        trueH = H(:,:,Hiterator);
        y = trueH(sampleIndices);
        
        theta = PPXA(y,OMP_A,0.3,rand(32));
        
        OMP_x = U' * reshape(theta, [32,32]) * U;
        diffListForCurrentSampling(Hiterator) = norm(OMP_x - trueH,'fro')/norm(trueH,'fro');
    end
    diffList(samplingIndex, :) = diffListForCurrentSampling;

    finalTime = toc();
    send(D, [samplingIndex, mean(diffListForCurrentSampling), finalTime])
    
    disp(['Finished sample num ', num2str(m), ' in ', num2str(finalTime), ' seconds'])
end

%%
function updateBarplot(samplingIndex, errorMean, finalTime, barPlot, timePlot)
    barPlot.YData(samplingIndex) = errorMean;
    timePlot.YData(samplingIndex) = finalTime;
    drawnow('limitrate')
end