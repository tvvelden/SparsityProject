function indexOfElbowPoint = findElbow(sortedX,showPlot)
% Find the elbow-point of a sorted vector. This is a similar method as used
% for PCA in cluster analysis. This method is a variation on the triangle
% threshold method.
% This function will most often return a (too) large value, which will lead
% to a rather large support of X.
%
% Inputs:
%   - sortedX   {Vector}:   Data vector in which an elbow point must be
%                           found
%   - showPlot  {Boolean}:  Toggle for showing a plot with the elbow point.
%                           The default value is false
%   
% Outputs:
%   - indexOfElbowPoint {Number}:   The index of the elbow point in the
%                                   vector sortedX

arguments
    sortedX {mustBeNumeric}
    showPlot logical = false % default value is false
end

    N = 32;

    % This method works by making a triangle between the first and last
    % value of the sortedX values. Then we see which point has the longest
    % distance to this line
    
    % Create the line y = ax + b through the first and last point
    startPoint = [1, sortedX(1)];
    endPoint = [N*N, sortedX(N*N)];

    a = (endPoint(2)-startPoint(2)) / (endPoint(1)-startPoint(1));
    b = startPoint(2) - a * startPoint(1);
    
    % find the distance from all points to that line (orthogonally)
    for xIndex = 1 : 1024
        % Find a line y = ax + b which is perpendicular to the triangle
        % line and crosses the current point
        cur_x = xIndex;
        cur_y = sortedX(xIndex);

        cur_a = -1/a; % 
        cur_b = cur_y - cur_a * cur_x;

        % Intersect the two lines
        cross_x = (cur_b - b)/(a-cur_a);
        cross_y = a * cross_x + b;
        % Debug print statement
%         disp("cross_x:" + cross_x +", cross_y:"+ cross_y+", cur_x:"+cur_x + ", cur_y:" + cur_y + ", cur_a:" + cur_a + ", cur_b:"+cur_b+", a:"+a+", b:"+b)
        
        % store the line length to each of the points
        diffList(xIndex) = norm([cur_x, cur_y] - [cross_x, cross_y],2);
    end

    % maximum distance should be returned
    [~, maxId] = max(diffList);
%     disp(maxId)
    
    if showPlot
        % plot the lines including the elbow point to show how large the
        % support of X should be according to this method
        figure(11)
        clf;
        % Plot the elbow line
%         subplot(1,2,1)
        plot(sortedX(1:end), LineWidth=2, Color="#0072BD")
        hold on
        plot([1, N*N],[sortedX(1), sortedX(end)], '--', LineWidth=2, Color="#0072BD")
        plot([1, maxId, 1024],[a*1+b, sortedX(maxId) a*1024+b], LineWidth=2)
        ylabel("$|\hat{\mathbf{X}}|$", Interpreter="latex")
        xlabel("Index of ordering")
        title("Solution of solving elbow method using triangle threshold method")

%         % Plot the distances of each point
%         subplot(1,2,2)
%         plot(diffList)
    end

    % return the value
    indexOfElbowPoint = maxId;
end