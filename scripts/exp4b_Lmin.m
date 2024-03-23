%% Random seed
% We fix the random seed for reproducibility.
rng(0)


%% Config
NoPts = 10; % No of points used to compute Lmin


%% Main loop
Ltable = zeros(3*2,NoPts);
for datasetno = 2:3
    switch datasetno
        case 1
            % ??? data
            featurelist = [1 2]; % ????, ????
        case 2
            % Wine data
            featurelist = [1 13]; % Alcohol, Proline

            wine = readmatrix('wine.data','FileType','text');
            names = {'Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'};

            wine = wine(randperm(size(wine,1)),:); % Randomize
            y = double(wine(:,1) == 1);
            X = wine(:,2:end);
        case 3
            % Taipei housing data
            featurelist = [3 4]; % Distance  to MRT, Convenience stores
            taipei_import_script;
            X = Realestatevaluationdataset(:,2:end-1);
            y = Realestatevaluationdataset(:,end);
            names = ["Date of Sale", "House Age", "Distance to MRT", "Convenience stores", "Latitude", "Longitude", "House Price"];
    end
    X = normalize(X);

    for p = 1:NoPts
        %% Train a GP
        % We used a leave-one-out strategy.
        [N,D] = size(X);
        Xt = X([1:p-1,p+1:N],:);
        yt = y([1:p-1,p+1:N],:);

        gp = fitrgp(Xt,yt,'KernelFunction','ardsquaredexponential','BasisFunction','none');

        %% Compute attributions
        xp = X(p,:)'; % these guys need to be column vectors
        x0 = 0*mean(X)';

        [Eattr,Vattr] = get_attr(gp,Xt,xp,x0);

        %% Approximate the attributions with numerical integration
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

        Llist = (1:100)';

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

            %             Ktt = (sf2)*exp(-0.5*(pdist2(Xt./ell',Xt./ell')).^2);
            %             xl = (x0 + (1:L).*dx/L)';
            %             for i = 1:D
            %                 ddKll = (sf2)*exp(-0.5*(pdist2(xl./ell',xl./ell')).^2).*( ell(i)^-2 - (ell(i)^-2*(xl(:,i)-xl(:,i)')).^2 );
            %                 dKln = (sf2)*exp(-0.5*(pdist2(xl./ell',Xt./ell')).^2).*((xl(:,i)-Xt(:,i)')*-ell(i)^-2);
            %                 MATRIX = ddKll - dKln*((Ktt + sn2*eye(size(Xt,1)))\dKln');
            %                 VattrL(i,ctr) = (dx(i)/L)^2 * ones(1,L)*MATRIX*ones(L,1);
            %             end


            % Compute Lmin
                    for featureno = 1:2
                        i0 = featurelist(featureno);
            
                        if abs(EattrL(i0,ctr) - Eattr(i0)) < 0.1*sqrt(Vattr(i0))
                        Ltable(2*(datasetno-1) + featureno, p) = Llist(ctr);
                        end
                    end
            
                    if all( Ltable(2*(datasetno-1) + (1:2), p))
                        break
                    end
        end


%         %% Compute Lmin
%         for featureno = 1:2
%             i0 = featurelist(featureno);
% 
%             indices = abs(EattrL(i0,:) - Eattr(i0)) < sqrt(Vattr(i0));
% 
%             if any(indices)
%                 Ltable(2*(datasetno-1) + featureno, p) = find(indices,1);
%             else
%                 Ltable(2*(datasetno-1) + featureno, p) = inf;
%             end
%         end
    end
end

%% Spit output to the command window
disp('Test results for Fig 4: Approximated integration')
disp('   Lmin:')
mean(Ltable,2)
