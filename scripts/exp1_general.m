%% Random seed
rng(0)


%% Import breast cancer data
import_breast_cancer

featurenames ={ 'ID number', ...
    'Outcome', ...
    'Recurrence time', ...
    'Radius (n1)', ...
    'Texture (n1)', ...
    'Perimeter (n1)', ...
    'Area (n1)', ...
    'Smoothness (n1)', ...
    'Compactness (n1)', ...
    'Concavity (n1)', ...
    'Concave points (n1)', ...
    'Symmetry (n1)', ...
    'Fractal dim (n1)', ...
    'Radius (n2)', ...
    'Texture (n2)', ...
    'Perimeter (n2)', ...
    'Area (n2)', ...
    'Smoothness (n2)', ...
    'Compactness (n2)', ...
    'Concavity (n2)', ...
    'Concave points (n2)', ...
    'Symmetry (n2)', ...
    'Fractal dim (n2)', ...
    'Radius (n3)', ...
    'Texture (n3)', ...
    'Perimeter (n3)', ...
    'Area (n3)', ...
    'Smoothness (n3)', ...
    'Compactness (n3)', ...
    'Concavity (n3)', ...
    'Concave points (n3)', ...
    'Symmetry (n3)', ...
    'Fractal dim (n3)'    };

%% Configure and fit the GP model 
% Let's look at only patients with recurrent tumors
outcomes = table2array(wpbc(:,2));
idcancerfree = outcomes ~= 'R'; % Patient cancer-free times

y = table2array(wpbc(idcancerfree,3));
X = table2array(wpbc(idcancerfree,4:33));
X = normalize(X);
names = featurenames(4:33);

% Leave-one-out
Xt = X(2:end,:);
yt = y(2:end,:);

[N,D] = size(Xt);

gp = fitrgp(Xt,yt,'KernelFunction','ardsquaredexponential',...
    'BasisFunction','none');

%% Do some attributional stuff
x0 = mean(Xt)';
xp = X(1,:)';

[Eattr,Varattr] = get_attr(gp,Xt,xp,x0);



%% Examine the whole poplulation
M = numel(yt);
Egp = zeros(M,1);
STDgp = zeros(M,1);
EATTR = zeros(M,D);
VATTR = zeros(M,D);
m = 0;
for n = 1:numel(yt)
    xp = Xt(n,:);
    [Eattr,Varattr] = get_attr(gp,Xt,xp,x0);
    EATTR(n,:) = Eattr(:);
    VATTR(n,:) = Varattr(:);

    [Egp(n),STDgp(n)] = predict(gp,xp);
end


%% Figure with attributions for two features, visualized across patients
i0 = 2;
j0 = 21;



figure(102)
tiledlayout(2,1,'Padding','compact','TileSpacing','compact')
nexttile
crossplot(EATTR(1:50,i0),VATTR(1:50,i0))
% hold on; plot( (0:50), 0*(0:50), 'k','LineWidth',1); hold off;
title( names(i0))
xlabel('Patient ID Number')
ylabel('Predicted attribution')
grid on;
xlim([0 51])
nexttile
crossplot(EATTR(1:50,j0),VATTR(1:50,j0))
title( names(j0))
xlabel('Patient ID Number')
ylabel('Predicted attribution')
grid on;
xlim([0 51])
% hold on; plot( (0:50), 0*(0:50), 'k','LineWidth',1); hold off;



set(gcf,'Position',[334 212 624 373]);

saveas(gcf,'./results/fig1b.png')



%% Figure with attributions for six patients
patlist = 6:-1:1;

figure(103)
tiledlayout(numel(patlist),1,'TileSpacing','tight','Padding','compact');

for patient = patlist
    nexttile
    gaussshapbarplot(EATTR(patient,:),VATTR(patient,:));
    ylabel(sprintf('Patient %d',patient))


    idx = mean(abs(EATTR)) < 0.03;
    ticks = 1:D;
    ticklabels = names;
    ticks(idx) = [];
    ticklabels(idx) = [];
    xticks(ticks)
    xticklabels({})
    xtickangle(90)
    % grid on;
    xlim([0 31])
end
xticklabels(ticklabels)
sgtitle('','FontSize',14)

%% Sizing the figure for publication
% (optional)
set(gcf,'Position',[400 712 408 536]);
saveas(gcf,'./results/fig1a.png')

