%% Random seed
% We fix the random seed for reproducibility.
rng(0)


%% Import wine data
wine = readmatrix('winequality-red.csv','FileType','text');
names = {'fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'};

% Randomize the data
wine = wine(randperm(size(wine,1)),:);

X = wine(:,1:11);
y = wine(:, 12 );

%% Predictive model
X = normalize(X);

% Allocate some points for the training set and testing
Xt = X(2:end,:);
yt = y(2:end,:);

idx = find(y == 8,1 );
Xp = X(idx,:);

% Make these guys column vectors
xp = Xp';
x0 = mean(X(y==5,:))';


[N,D] = size(Xt); % N samples of D-dimensional input vectors

% ARD-SE GP
gp = fitrgp(Xt,yt,'KernelFunction','ardsquaredexponential','BasisFunction','none');

%% Prediction
[yp,ys] = predict(gp,Xp);

%% Compute attributions
[Eattr,Vattr] = get_attr(gp,Xt,xp,x0);

%% Approximate Integral
% Use the right-hand rule to numerically integrate
alpha = gp.Alpha;
ell = gp.KernelInformation.KernelParameters(1:D);
sf2 = gp.KernelInformation.KernelParameters(D+1)^2;
sn2 = gp.Sigma^2;
Ain = zeros(D,N);
Bi = zeros(D,1);
mu = zeros(D,1);
Linv = diag(ell.^-1);
s = sqrt(2)/norm(Linv*(xp-x0));

Llist = round(exp(linspace(0,log(100),50)))';

EattrL = zeros(D,numel(Llist));
VattrL = zeros(D,numel(Llist));

dx = xp - x0;

for ctr =1:numel(Llist)
    L = Llist(ctr);
    %     xl = (x0 + (1:L).*dx/L)';
    dFdx = zeros(D,L);
    for l = 1:L
        xl = (x0 + l.*dx/L)';
        dkdx = (sf2)*exp(-0.5*(pdist2(xl./ell',Xt./ell')).^2).*((xl-Xt)'./(-ell.^2));
        dFdx(:,l) = dkdx*alpha;
    end

    EattrL(:,ctr) = dx.*mean(dFdx,2);


    Ktt = (sf2)*exp(-0.5*(pdist2(Xt./ell',Xt./ell')).^2);
    xl = (x0 + (1:L).*dx/L)';
    for i = 1:D
        ddKll = (sf2)*exp(-0.5*(pdist2(xl./ell',xl./ell')).^2).*( ell(i)^-2 - (ell(i)^-2*(xl(:,i)-xl(:,i)')).^2 );
        dKln = (sf2)*exp(-0.5*(pdist2(xl./ell',Xt./ell')).^2).*((xl(:,i)-Xt(:,i)')*-ell(i)^-2);
        MATRIX = ddKll - dKln*((Ktt + sn2*eye(size(Xt,1)))\dKln');
        VattrL(i,ctr) = (dx(i)/L)^2 * ones(1,L)*MATRIX*ones(L,1);
    end

end


%% Plot the attribution to Proline
i0 = 11;

figure(4)
semilogx(Llist, EattrL(i0,:),'ro-','LineWidth',1)
xx = xlim;
hold on;
plot(xx,Eattr(i0) + 0*xx,'k','LineWidth',1)
patch([xx,fliplr(xx)]', Eattr(i0) + sqrt(Vattr(i0))*[-1 -1 1 1]','black','EdgeColor','none','FaceAlpha',0.1);

% gaussbarplot(EattrL(i0,:),VattrL(i0,:),Llist)
patch([Llist;flipud(Llist)], [EattrL(i0,:)';flipud(EattrL(i0,:)')]+[sqrt(VattrL(i0,:)');-sqrt(flipud(VattrL(i0,:)'))],'red','EdgeColor','none','FaceAlpha',0.1);
hold off;

xlabel('Number of partitions, L')
ylabel('attr_i(x|F)')
title(sprintf('Attribution to %s',names{i0}),'FontSize',15);
legend('Approximate integration','Exact integration','Location','best')

grid on;
grid minor;

gcf; ans.Position = [333 922 406 275];
saveas(gcf,'results/fig4.png')


%% Supplemental Figure: Attributions to all features
% Right-hand rule for all features

figure(6)
tiledlayout(5,3,'TileSpacing','compact','Padding','tight')
for i0 = 1:11
    nexttile;

    semilogx(Llist, EattrL(i0,:),'ro-','LineWidth',1)
    xx = xlim;
    hold on;
    plot(xx,Eattr(i0) + 0*xx,'k','LineWidth',1)
    patch([xx,fliplr(xx)]', Eattr(i0) + sqrt(Vattr(i0))*[-1 -1 1 1]','black','EdgeColor','none','FaceAlpha',0.1);

    patch([Llist;flipud(Llist)], [EattrL(i0,:)';flipud(EattrL(i0,:)')]+[sqrt(VattrL(i0,:)');-sqrt(flipud(VattrL(i0,:)'))],'red','EdgeColor','none','FaceAlpha',0.1);
    hold off;

    xlabel('Number of partitions, L')
    ylabel('attr_i(x|F)')
    title(sprintf('Attr. to %s (%d)',names{i0},i0),'FontSize',15);
end

    nexttile;
shapbarplot(Eattr)
title('Overview of all attributions','FontSize',15);

    nexttile;
stem(ell.^-1,'filled')
title('Relative feature importance','FontSize',15);
legend('ARD relevance weight, $\ell^{-1}$','FontSize',12,'Interpreter','latex','location','best');

saveas(gcf,'results/fig4_suppl.png')
