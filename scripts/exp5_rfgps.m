%% Random seed
rng(0)

%% Import wine data
wine = readmatrix('winequality-red.csv','FileType','text');
featurenames = {'fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'};

% Randomize the data
wine = wine(randperm(size(wine,1)),:);

X = wine(:,1:11);
y = wine(:, 12 );

%% Predictive model
X = normalize(X);

% Allocate some points for the training set and testing
idx = find(y == 8,1 );
N = size(X,1);
Xt = X([1:idx-1,idx+1:N],:);
yt = y([1:idx-1,idx+1:N],:);
Xp = X(idx,:);


xp = Xp;
x0 = 0*mean(X(y==5,:));


[N,D] = size(Xt); % N samples of D-dimensional input vectors

% ARD-SE GP
gp = fitrgp(Xt,yt,'KernelFunction','ardsquaredexponential','BasisFunction','none');


%% Attribution with Exact GPR
[yp,ys] = predict(gp,Xp);
[Eattr,Vattr] = get_attr(gp,Xt,xp,x0);


%% Initialization for the RFGP stuff

NoIter = 500; % No. of models to test per value of M
Mlist = [10, 50, 100];


EattrM = zeros(NoIter,numel(Mlist),D);
VattrM = zeros(NoIter,numel(Mlist),D);
YpRF = zeros(NoIter,numel(Mlist));
YsRF = zeros(NoIter,numel(Mlist));

%% RFGP loop
    i0 = 11;


for ctr = 1:numel(Mlist)
    M = Mlist(ctr)
    EATTRF = zeros(NoIter,D);
    VATTRF = zeros(NoIter,D);

    fprintf('D=%d, M=%d\n',D,M)

    for iter = 1:NoIter
        ell = gp.KernelInformation.KernelParameters(1:end-1);
        sig20 = gp.KernelInformation.KernelParameters(end);
        sig2n = gp.Sigma;

        v = (ell.^-1)'.*randn(M,D);
        Phi = [sin(v*Xt');cos(v*Xt')];
        A  = Phi*Phi' + M*sig2n/sig20 * eye(2*M);


        % Predict at every test point
        phi = [sin(v*xp');cos(v*xp')];
        ypRF = phi'*(A\Phi)*yt;
        ysRF= sig2n + sig2n*phi'*(A\phi);

        ybaselineRF = [sin(v*x0');cos(v*x0')]'*(A\Phi*yt);

        %% RFGP attributions
        EattrRF = zeros(1,D);
        VarattrRF = zeros(1,D);
        for i =1:D
            zeta = [(v(:,i)./(v*xp')).*(sin(v*xp')-sin(v*x0')) ; (v(:,i)./(v*xp')).*(cos(v*xp')-cos(v*x0'))];
            EattrRF(i) = (xp(i)-x0(i)) * zeta'*(A\(Phi*yt));
            VarattrRF(i) = sig2n*(xp(i)-x0(i))^2*zeta'*(A\zeta);
        end

        YpRF(iter,ctr) = ypRF;
        YsRF(iter,ctr) = ysRF; 
        EattrM(ctr,iter,:) =  EattrRF(:);
        VattrM(ctr,iter,:) = VarattrRF(:);
    end




    %% Plotting the attributions
    figure(43)
    if ctr == 1
        tiledlayout(1,numel(Mlist),'Padding','tight','TileSpacing','compact');
    end
    
    colororder([0.83 0.14 0.14
                0.07 0.65 0.80
                0.47 0.25 0.80
                0.25 0.80 0.54])

    nexttile;
    upper = Eattr(i0) + 5*sqrt(Vattr(i0));
    lower = Eattr(i0) - 5*sqrt(Vattr(i0));
    xrange = linspace(lower,upper,1500);
    opdf = normpdf(xrange,Eattr(i0),sqrt(Vattr(i0)));
    patch([xrange,flipud(xrange)], [opdf, 0*opdf],'black','EdgeColor','none','FaceAlpha',0.15)

    names = cell(5,1);
    names{1} = 'Exact GPR';
    names{2} = 'Marginalized RFGP';
    for m = 3:5
        names{m} = 'RFGP';
    end

    hold on;

    curves = zeros(NoIter,numel(xrange));
    for iter = 1:NoIter
        curves(iter,:) = normpdf(xrange,EattrM(ctr,iter,i0),sqrt(VattrM(ctr,iter,i0)));
        curves(iter,:) = curves(iter,:);
    end
    meancurve = (1/NoIter)*ones(1,NoIter)*curves;
    patch([xrange,flipud(xrange)], [meancurve, 0*meancurve],'red','EdgeColor','none','FaceAlpha',0.15)
    plot(xrange, curves(1:2,:),'-','LineWidth',1)
    hold off;
    title(sprintf('M=%d frequencies',M))
    xlabel(['Attribution to ',featurenames{i0}]);
    grid on;


    if ctr == 1
        ylabel('Probability/confidence');
    elseif ctr == 3
        legend(names,'Location','best')
    end



end

%% Save figure
figure(43)
set(gcf,'Position',[67 796 868 186]);
saveas(gcf,'./results/fig5.png')
