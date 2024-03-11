%% starplot.m
% This function will draw a simple star plot for visualizing a feature
% vector.
function starplot(X)
    clf; % Clear figure

    % Each vector to be plotted should be a ROW in X
    [N,D] = size(X);
    angles = 2*pi*(0:D)/D;
    hold on;
    C = colororder;
    % Lines
    for n = 1:N
        x = [X(n,:),X(n,1)];
        currentlinecol = C(1 + mod(n-1,size(C,1)),:);
        plot(x.*cos(angles),x.*sin(angles),'-o',...
            'MarkerFaceColor',currentlinecol,...
            'MarkerEdgeColor',currentlinecol,...
            'Color',currentlinecol,...
            'LineWidth',1);
    end

    % Patches
    for n = 1:N
        x = [X(n,:),X(n,1)];
        currentlinecol = C(1 + mod(n-1,size(C,1)),:);
        patch(x.*cos(angles),x.*sin(angles),currentlinecol,...
            'FaceColor',currentlinecol,...
            'EdgeColor','none',...
            'FaceAlpha',0.1);
    end

    % Add the grid
    for fraction = 0.2:0.2:1
        plot(fraction*cos(angles),fraction*sin(angles),'k')
    end
    for d = 1:D
        plot(cos(angles(d))*[1 0],sin(angles(d))*[1 0],'k')
    end
    hold off;
end