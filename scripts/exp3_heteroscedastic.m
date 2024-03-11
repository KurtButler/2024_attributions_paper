%% Random seed
rng(0)

%% Configure and fit the GP model 
F = @(X) sin(X(:,1)).*sin(1.5*X(:,2));
Xt = 10*rand(500,2);
yt = F(Xt);
yt = yt + 0.5*randn(size(yt));

[N,D] = size(Xt);
featurenames = {'x_1','x_2'};

gp = fitrgp(Xt,yt,'KernelFunction','ardsquaredexponential',...
    'BasisFunction','none');

%% Basic attribution test
x0 = [0 0];
xp = [1 0];
[Eattr,Varattr] = get_attr(gp,Xt,xp,x0);


%% Compute attributions along the path
t = (10*(0:N)/N);

M = sum(yt);
Egp = zeros(N+1,1);
Sgp = zeros(N+1,1);
EATTR = zeros(N+1,D);
VATTR = zeros(N+1,D);
m = 0;
for n = 1:N+1
    % Compute the point beta(t)
    xp = t(n)*[1 1];

    % Predict from the GPR model
    [meanpred,stdpred] = predict(gp,xp);
    Egp(n,:) = meanpred;
    Sgp(n,:) = stdpred;
    
    % Compute attributions for this prediction
    [Eattr,Varattr] = get_attr(gp,Xt,xp,x0);
    EATTR(n,:) = Eattr(:);
    VATTR(n,:) = Varattr(:);
end



%% Figure 2
i0 = 1;
j0 = 2;


linedist =  t;

figure(102)
tiledlayout(2,3,'Padding','tight','TileSpacing','compact')

nexttile
[X1,X2] = meshgrid(linspace(0,10,30));
Yhat = zeros(size(X1));
Yhat(:) = predict(gp,[X1(:),X2(:)]);
surf(X1,X2,Yhat,'EdgeAlpha',0.5)
xlabel('$x_1$','interpreter','latex','FontSize',12)
ylabel('$x_2$','interpreter','latex','FontSize',12)
zlabel('$F(x_1,x_2)$','interpreter','latex','FontSize',12)
title('Function learned via GPR','interpreter','latex','FontSize',15)
view([1 -1 3])

nexttile([1 2])
plot(linedist', Egp,'k', 'LineWidth',1.5);
patch([linedist,fliplr(linedist)],[(Egp'+Sgp'), fliplr(Egp'-Sgp')],'black','FaceAlpha',0.1,'EdgeColor','none') 
title('GPR predictions along the line','interpreter','latex','FontSize',15)
xlabel('Location along the line, $t$','interpreter','latex','FontSize',12)
ylabel('Predicted value, $F(\beta(t))$','interpreter','latex','FontSize',12)
grid on;
xlim([linedist(1),linedist(end)])

hold on;
plot(linedist,EATTR*ones(D,1) + Egp(1),'g--','LineWidth',1.5)
hold off;
legend({"$E(F(x))$","$E(F(x))\pm1\sigma(x)$","$F(\tilde{x})+\Sigma_i attr_i(x)$"},'Location','best','Interpreter','latex','FontSize',12)

nexttile
plot(linedist,EATTR(:,i0),'b','LineWidth',1)
hold on; plot( linedist, 0*(0:N), 'k','LineWidth',1); hold off;
title('Attributions to feature 1','interpreter','latex','FontSize',15)
patch([linedist,fliplr(linedist)],[EATTR(:,i0)'+sqrt(VATTR(:,i0)'), fliplr(EATTR(:,i0)'-sqrt(VATTR(:,i0)'))],'blue','FaceAlpha',0.1,'EdgeColor','none')  
xlabel('Location along the line, $t$','interpreter','latex','FontSize',12)
ylabel('$attr_1(\beta(t)|F)$','interpreter','latex','FontSize',12)
grid on;
xlim([linedist(1),linedist(end)])

nexttile
plot( linedist,EATTR(:,j0),'r','LineWidth',1)
title('Attributions to feature 2','interpreter','latex','FontSize',15)
patch([linedist,fliplr(linedist)],[EATTR(:,j0)'+sqrt(VATTR(:,j0)'), fliplr(EATTR(:,j0)'-sqrt(VATTR(:,j0)'))],'red','FaceAlpha',0.1,'EdgeColor','none')  
xlabel('Location along the line, $t$','interpreter','latex','FontSize',12)
ylabel('$attr_2(\beta(t)|F)$','interpreter','latex','FontSize',12)
grid on;
hold on; plot(linedist, 0*(0:N), 'k','LineWidth',1); hold off;
xlim([linedist(1),linedist(end)])


nexttile()
plot(linedist,(VATTR(:,i0)'),'b',linedist,(VATTR(:,j0)'),'r','LineWidth',1)
title('Attribution variances','interpreter','latex','FontSize',15)
xlabel('Location along the line, $t$','interpreter','latex','FontSize',12)
ylabel('Variance of attribution')
grid on;
legend('Var($attr_1(\beta(t)|F)$)','Var($attr_2(\beta(t)|F)$)','Location','best','interpreter','latex','FontSize',13)
xlim([linedist(1),linedist(end)])


%% Sizing the figure for publication
% (optional)
% gcf; ans.Position = [200 200 750 462];
% saveas(gcf,'results/fig3.png')


