
N = 1000;
D = 200;
a = randn(D,1).*(rand(D,1)>0.6);
X = 3*randn(N,D);
y = X*a + randn(N,1);


gp = fitrgp(X,y,'KernelFunction','ardsquaredexponential',...
    'BasisFunction','none');


%% Do some attributional stuff
x0 =  zeros(1,D);
xp =  ones(1,D);

disp(' = = = ')
disp('Newest version')
tic;
[Eattr,Varattr] = get_attr(gp,X,xp,x0);
toc

disp('Improvement 2')
tic;
[Eattr,Varattr] = get_attr2(gp,X,xp,x0);
toc

disp('Improvement 1')
tic;
[Eattr,Varattr] = get_attr1(gp,X,xp,x0);
toc

%%
disp('Current Github version')
tic;
[Eattr,Varattr] = get_attr0(gp,X,xp,x0);
toc
