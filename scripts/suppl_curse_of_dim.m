%% Random seed
rng(0)

NoIter = 100; % No. of repeats for RFGPs
D = 10;
N = 500;

for dimension = 1:3
%     Xt = X(:,1:(2+4*(dimension-1)));
    D = 2+4*(dimension-1);

X = randn(N,D);
X = X*rand(D) + randn(N,D);

Xt = X;
yt = X*ones(D,1) + 0.1*randn(N,1);

    gp = fitrgp(Xt,yt,'KernelFunction','ardsquaredexponential',...
        'BasisFunction','none');

    %% Attribution with Exact GPR
    x0 = zeros(size(Xt(1,:)));
    xp = 1+0*x0;
    [yp,ys] = predict(gp,xp);
    [Eattr,Vattr] = get_attr(gp,Xt,xp,x0);

    %% Initialization for the RFGP stuff
    Mlist = [100];
    EattrM = zeros(NoIter,numel(Mlist),D);
    VattrM = zeros(NoIter,numel(Mlist),D);
    YpRF = zeros(NoIter,numel(Mlist));

    %% RFGP loop
    for ctr = 1:numel(Mlist)
        M = Mlist(ctr);
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

            %             EATTRF(iter,:) = EattrRF(:);
            %         RFmse(iter,:) = ((EattrRF(:)-Eattr).^2);
            %         VATTRF(iter,:) = VarattrRF(:);


            YpRF(iter,ctr) = ypRF;
            EattrM(ctr,iter,:) =  EattrRF(:);
            VattrM(ctr,iter,:) = VarattrRF(:);
        end


    end


    %% Plotting the attributions
    figure(43)
    if dimension == 1
        tiledlayout(1,3,'Padding','tight','TileSpacing','compact');
    end


    nexttile;
    i0 = 1;
    upper = Eattr(i0) + 10*sqrt(Vattr(i0));
    lower = Eattr(i0) - 10*sqrt(Vattr(i0));
    lower = 0.8;
    upper = 1.2;
    xrange = linspace(lower,upper,1500);
    opdf = normpdf(xrange,Eattr(i0),sqrt(Vattr(i0)));
    plot(xrange, opdf ,'k','LineWidth',2)
    names{1} = 'Exact GPR';
    hold on;
    for m = 1:numel(Mlist)
        switch m
            case 1
                cstr = 'r';
            case 2
                cstr = 'g';
            case 3
                cstr = 'b';
        end

        names{2*m} = sprintf('Marginalized (M=%d)',Mlist(m));
        names{1+2*m} = sprintf('RFGP(M=%d)',Mlist(m));
        curves = zeros(NoIter,numel(xrange));
        for iter = 1:NoIter
            curves(iter,:) = normpdf(xrange,EattrM(m,iter,i0),sqrt(VattrM(m,iter,i0)));
            curves(iter,:) = curves(iter,:);
        end
        meancurve = (1/NoIter)*ones(1,NoIter)*curves;
        plot(xrange, meancurve,cstr,'LineWidth',1)
        plot(xrange, curves(iter,:),[cstr,'-.'],'LineWidth',1)
    end
    hold off;
    title(sprintf('D=%d features',D))

    if dimension == 3
    legend(names)
    end
    grid on;
%     ylim([0,1.1*max(opdf)])
end

%% Save figure
% gcf; ans.Position = [219 401 878 229];
% saveas(gcf,'results/curse_of_dim.png')

