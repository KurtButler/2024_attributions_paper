function gaussshapbarplot(ylist,yvars,xlist)
%GAUSSBARPLOT
% Creates a cute plot
if nargin < 2
    yvars = zeros(size(ylist));
end
if nargin < 3
    xlist = 1:numel(ylist);
end


for i = 1:numel(ylist)
    if ylist(i) > 0
        COLORKEY = 'blue';
    elseif  ylist(i) < 0
        COLORKEY = 'red';
    else
        COLORKEY = 'black';
    end
    plot(xlist(i) + [-0.4 0.4]', ylist(i) + [0 0]',COLORKEY,'LineWidth',1.5)
    hold on;
    plot(xlist(i) + [-0.2 0.2]', ylist(i)+sqrt(yvars(i))+ [0 0]',COLORKEY,'LineWidth',1.5)
    plot(xlist(i) + [-0.2 0.2]', ylist(i)-sqrt(yvars(i)) + [0 0]',COLORKEY,'LineWidth',1.5)


    patch(xlist(i) + [-0.4 0.4 0.4 -0.4]', + ylist(i)*[1 1 0 0]',COLORKEY,...
        'FaceAlpha',0.1,'EdgeColor','none')
    plot(xlist(i)+[0;0], ylist(i) + sqrt(yvars(i))*[1, -1]',COLORKEY,'LineWidth',1.5)

% 
%     patch(xlist(i) + [-0.4 0.4 0.4 -0.4]',ylist(i) + sqrt(yvars(i))*[1 1 -1 -1]',COLORKEY,...
%         'FaceAlpha',0.1,'EdgeColor','none')
%     plot(xlist(i)+[0;0], ylist(i)*[1, 0]',COLORKEY,'LineWidth',1.5)
end
hold off;

end

