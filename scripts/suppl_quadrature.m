%% Random seed
% We fix the random seed for reproducibility.
rng(0)

%% Feature of interest
i0 = 11;


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

%% Loop
for rule = 1:3
    switch rule
        case 1
            quadrule = 'Right hand';
            colorstr = 'red';
            shortstr = 'ro-';
        case 2
            quadrule = 'Trapezoid';
            colorstr = 'green';
            shortstr = 'g+-';
        case 3
            quadrule = 'Simpson';
            colorstr = 'blue';
            shortstr = 'bx-';
    end


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

    D = size(Eattr,1);
Llist = unique(round(exp(linspace(0,log(100),50)))');

    EattrL = zeros(D,numel(Llist));
    VattrL = zeros(D,numel(Llist));

    dx = xp - x0;

    for ctr =1:numel(Llist)
        L = Llist(ctr);
        %     xl = (x0 + (1:L).*dx/L)';
        dFdx = zeros(D,L);

        % Locations and weights are row vectors
        tt = linspace(0,1,L);
        w = ones(size(tt));

        % Rule 1 is just the right hand rule
        if rule == 2
            % Trapezoid rule
            w(2:end-1) = 2;
        elseif rule == 3
            % Simpson rule requires us to compute midpoints
            tmp = tt;
            L = 2*L -1;
            tt = linspace(0,1,L);
            w = ones(size(tt));
            w(2:2:end) = 4;
            w(3:2:end-1) = 2;
        end
        w = w/sum(w);

        for l = 1:L
            xl = (x0 + tt(l).*dx)';
            dkdx = (sf2)*exp(-0.5*(pdist2(xl./ell',Xt./ell')).^2).*((xl-Xt)'./(-ell.^2));
            dFdx(:,l) = dkdx*alpha;
        end

        % Mean calculation
        EattrL(:,ctr) = dx.*(dFdx*w');

        % Variance calculation
        Ktt = (sf2)*exp(-0.5*(pdist2(Xt./ell',Xt./ell')).^2);
        xl = (x0 + tt(1:L).*dx)';
        for i = 1:D
            ddKll = (sf2)*exp(-0.5*(pdist2(xl./ell',xl./ell')).^2).*( ell(i)^-2 - (ell(i)^-2*(xl(:,i)-xl(:,i)')).^2 );
            dKln = (sf2)*exp(-0.5*(pdist2(xl./ell',Xt./ell')).^2).*((xl(:,i)-Xt(:,i)')*-ell(i)^-2);
            MATRIX = ddKll - dKln*((Ktt + sn2*eye(size(Xt,1)))\dKln');
            VattrL(i,ctr) = (dx(i))^2 * w*MATRIX*w';
        end


    end


    
    %% Plot attribution vs number of partitions

    figure(8)
    if rule==1; tiledlayout(1,1,'Padding','compact'); nexttile; end
    if rule == 1
        xx = [Llist(1),Llist(end)];
        semilogx(xx,Eattr(i0) + 0*xx,'black','LineWidth',1)
        patch([xx,fliplr(xx)]', Eattr(i0) + sqrt(Vattr(i0))*[-1 -1 1 1]','black','EdgeColor','none','FaceAlpha',0.1);
    end
    hold on;
    plot(Llist, EattrL(i0,:),shortstr,'LineWidth',1)

    % gaussbarplot(EattrL(i0,:),VattrL(i0,:),Llist)
    patch([Llist;flipud(Llist)], [EattrL(i0,:)';flipud(EattrL(i0,:)')]+[sqrt(VattrL(i0,:)');-sqrt(flipud(VattrL(i0,:)'))],colorstr,'EdgeColor','none','FaceAlpha',0.1);
    hold off;

    xlabel('Number of partitions, L')
    ylabel('attr_i(x|F)')
%     sgtitle(sprintf('Attribution to %s vs L',names{i0}),'FontSize',15);
    legend('Exact integration','','Right hand rule','','Trapezoid rule','',"Simpson's rule",'')
    

    %% Plot attribution vs number of function calls
    if rule == 3
        NoEvals = Llist*2-1;
    else
        NoEvals = Llist;
    end

    figure(9)
    if rule==1; tiledlayout(1,1,'Padding','compact'); nexttile; end
    if rule == 1
        xx = [NoEvals(1),NoEvals(end)];
        semilogx(xx,Eattr(i0) + 0*xx,'black','LineWidth',1)
        patch([xx,fliplr(xx)]', Eattr(i0) + sqrt(Vattr(i0))*[-1 -1 1 1]','black','EdgeColor','none','FaceAlpha',0.1);
        xlim(xlim);
    end
    hold on;
    plot(NoEvals, EattrL(i0,:),shortstr,'LineWidth',1)

    % gaussbarplot(EattrL(i0,:),VattrL(i0,:),Llist)
    patch([NoEvals;flipud(NoEvals)], [EattrL(i0,:)';flipud(EattrL(i0,:)')]+[sqrt(VattrL(i0,:)');-sqrt(flipud(VattrL(i0,:)'))],colorstr,'EdgeColor','none','FaceAlpha',0.1);
    hold off;

    xlabel('Number of function evaluations')
    ylabel('attr_i(x|F)')
%     sgtitle(sprintf('Attribution to %s vs no. of functional evals',names{i0}),'FontSize',15);
    legend('Exact integration','','Right hand rule','','Trapezoid rule','',"Simpson's rule",'','Location','best')



    %% Plot the errors
    ErrorFunc = @(X,Y) ((X-Y).^2); % Mean square error


    figure(10)
    if rule==1; tiledlayout(1,1,'Padding','compact'); nexttile; end
    semilogy(NoEvals, ErrorFunc(EattrL(i0,:),Eattr(i0)),shortstr,'LineWidth',1)
    hold on;
    if rule == 3
        hold off;
    end

    xlabel('Number of function evaluations')
    ylabel('MSE(attr_i(x|F))')
    legend('Right hand rule','Trapezoid rule',"Simpson's rule")
    xlim([0,30])


end

%% Some reformatting to make the figures as shown in the paper
figure(8)
gcf; ans.Position = [323 818 360 294];
saveas(gcf,'results/quadrules_1.png')

figure(9)
gcf; ans.Position = [311 499 360 279];
saveas(gcf,'results/quadrules_2.png')

figure(10)
gcf; ans.Position = [315 155 360 282];
saveas(gcf,'results/quadrules_3.png')
