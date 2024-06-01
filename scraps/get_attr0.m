function [Eattr,Varattr] = get_attr0(gp,Xt,xp,x0)
% This function assumes SE or  ARD-SE kernel.

%% Reformat the input vectors
% These guys should be row vectors for the following code snippet
if size(x0,2) ~= 1
    x0 = x0';
end
if size(xp,1) ~= size(x0,1)
    xp = xp';
end

[N,D] = size(Xt);

%% Some calculations
alpha = gp.Alpha;
if strcmp(gp.KernelFunction,'SquaredExponential')
    ell = ones(1,D)* gp.KernelInformation.KernelParameters(1);
else
    ell = gp.KernelInformation.KernelParameters(1:end-1);
end
sf2 = gp.KernelInformation.KernelParameters(end)^2;
sn2 = gp.Sigma^2;

A = zeros(D,N);
B = zeros(D,1);
mu = zeros(D,1);
Linv = diag(ell.^-1);
s = sqrt(2)/norm(Linv*(xp-x0));
for i = 1:D
    if xp(i)-x0(i) ~= 0
    % Mean calculation
        a = (xp-x0)'*(Linv^2)*(xp-x0);
        for n = 1:N
            xn = Xt(n,:)';
            b = 2*(x0-xn)'*(Linv^2)*(xp-x0);
            c = (x0-xn)'*(Linv^2)*(x0-xn);

            d = -sf2*(xp(i)-x0(i))*(xp(i)-x0(i))/ell(i)^2;
            f = -sf2*(xp(i)-x0(i))*(x0(i)-xn(i))/ell(i)^2;
            A(i,n) = (exp((-a - b - c)/2)*(4*sqrt(a)*d*(-1 + exp((a + b)/2)) + exp((2*a + b)^2/(8*a))*(b*d - 2*a*f)*sqrt(2*pi)*(erf(b/(2*sqrt(2)*sqrt(a))) - erf((2*a + b)/(2*sqrt(2)*sqrt(a))))))/(4*a^(3/2));

        end
    % Variance calculation
    v = -(sf2*(xp(i)-x0(i))^2)/(ell(i)^4);
    w = sf2/(ell(i)^2);
    B(i) = (-2*(-1 + exp(a/2))*(2*v + a*w))/(exp(a/2)*a^2) + (sqrt(2*pi)*(v + a*w)*erf(sqrt(a)/sqrt(2)))/a^(3/2);
    B(i) =  B(i)*(xp(i)-x0(i))^2;
    end
end
Eattr = A*alpha;
Ktrain = sf2*exp(-1/2*pdist2(Xt*Linv,Xt*Linv).^2) + sn2*eye(N);

Varattr = zeros(size(Eattr));
Varattr(:) = B - diag(A*(Ktrain\A'));


if strcmp(gp.BasisFunction,'Linear')
    Eattr = Eattr + gp.Beta(2:end).*(xp-x0);
end
end
