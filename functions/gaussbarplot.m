function gaussbarplot(ylist,yvars,xlist)
%GAUSSBARPLOT
% Creates a cute plot
if nargin < 2
    yvars = zeros(size(ylist));
end
if nargin < 3
    xlist = 1:numel(ylist);
end

COLORKEY = 'black';
% COLORCUSTOM = [58, 158, 117]/256;
for i = 1:numel(ylist)
    plot(xlist(i) + [-0.4 0.4]', ylist(i) + [0 0]',COLORKEY,'LineWidth',1.5)
%     plot(xlist(i) + [-0.4 0.4]', ylist(i) + [0 0]','Color',COLORCUSTOM,'LineWidth',1.5)
    hold on;
patch(xlist(i) + [-0.4 0.4 0.4 -0.4]',ylist(i) + sqrt(yvars(i))*[1 1 -1 -1]',COLORKEY,...
    'FaceAlpha',0.1,'EdgeColor','none')
end
hold off;

end

