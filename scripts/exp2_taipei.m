%% Random seed
rng(0)

%% Taipei import script
taipei_import_script;


%% Major loop
for LoopNo = 1:2



    %% Set up
    X = Realestatevaluationdataset(:,2:end-1);
    y = Realestatevaluationdataset(:,end);
    names = ["Date of Sale", "House Age", "Distance to MRT", "Convenience stores", "Latitude", "Longitude", "House Price"];

    if LoopNo == 2
        X = normalize(X);
    end

    %% GP model
    Xt = X(1:end-50,:);
    yt = y(1:end-50,:);
    [N,D] = size(Xt);

    xtest = X(end-49:end,:);
    ytest = y(end-49:end,:);


    %% Fit the GPR model
    gp = fitrgp(Xt,yt,'KernelFunction','ardsquaredexponential',...
        'BasisFunction','none');



    %% Define points for attributions
    x0 = mean(Xt)';
    xp = xtest(1,:)';

    %% Plot the attributions
    if LoopNo == 1
        figure(1)
        tiledlayout(2,2,"TileSpacing","compact",'Padding','compact')
    end

    [Eattr,Vattr] = get_attr(gp,Xt,xp,x0);

    nexttile;
    shapbarplot(xp-x0)
    if LoopNo == 1
    title('$\mathbf{x}-\tilde{\mathbf{x}}$ before normalization','interpreter','latex','FontSize',15)
    else
    title('$\mathbf{x}-\tilde{\mathbf{x}}$ after normalization','interpreter','latex','FontSize',15)
    end
    xticks(1:D)
    xticklabels(names)
    grid on
    ylabel('Feature values')


    nexttile;
    gaussshapbarplot(Eattr,Vattr)
    xticks(1:D)
    xticklabels(names)
    ylabel('Feature attributions')
    if LoopNo == 1
    title('Feature attributions before normalization','interpreter','latex','FontSize',15)
    else
    title('Feature attributions after normalization','interpreter','latex','FontSize',15)
    end

end

%% Text output
fprintf('ARD-SE results\n')
fprintf('F(x)-F(x0):\t%0.2f\n',predict(gp,xp')-predict(gp,x0'))
fprintf('Sum(attr): \t%0.2f\n',sum(Eattr))



%% Sizing the figure for publication
% (optional)
% gcf; ans.Position = [31 653 560 420];
% saveas(gcf,'results/fig2.png')
