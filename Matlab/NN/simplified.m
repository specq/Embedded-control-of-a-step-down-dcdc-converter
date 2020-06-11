clc
clear all
close all

% Loading sys parameters
load ohm100_10kHz.mat;

G = ohm100_10kHz;
Ts = G.Ts;
A = G.A; B = G.B; C = [0 1];
ref = 5;
val_ss = [A-eye(2) B; C 0]\[0;0;1]*ref;
xs = val_ss(1:2);
us = val_ss(3);

% MPC design
N = 10;
Q = [90,0;0,1];
R = 1;
model = LTISystem('A',A,'B',B,'Ts',Ts);
model.x.min = [-xs(1); -xs(2)];
model.x.max = [0.2-xs(1); 11.7-xs(2)];
model.u.min = -us;
model.u.max = 1-us;
model.x.penalty = QuadFunction(Q);
model.u.penalty = QuadFunction(R);
Tset = model.LQRSet;
PN = model.LQRPenalty;
model.x.with('terminalSet');
model.x.terminalSet = Tset;
model.x.with('terminalPenalty');
model.x.terminalPenalty = PN;

mpc = MPCController(model, N);
expMpc = mpc.toExplicit();

% I am using 2 as the max voltage because the feasible set reduced the
% original 6 a lot, it is just to get better scaling
xmax = [0.15; 2]; 
xmin = [-0.05; -5];

umax = 0.6621;
umin = -0.3379;

xdelta = xmax-xmin;
udelta = umax-umin;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculating the explicit solution to the first linear + pQP layers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Select the file to be loaded 
PYTHONfile = 'NNparams_var_3_run_2_epoch_109'; % GOLDEN!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Loading Neural Network parameters... \n');
filePath = ['./' PYTHONfile '.mat'];
fit = load(filePath);

% converting python params to MATLAB doubles
fit.F = double(fit.F);
fit.f = double(fit.f(:));
fit.L = double(fit.L);
fit.O = double(fit.O);
fit.o = double(fit.o(:));

% Building the pQP problem in YALMIP
nVar = size(fit.L,2);
nParam = size(fit.F,2); 
fit.Q = fit.L'*fit.L + 1e-3*eye(nVar);
z = sdpvar(nVar,1);
x = sdpvar(nParam,1); 

% pQP format: min .5*z'*Q*z + (F*x + f')'*z
%            s.t. z >= 0
% Plus, solve the parameters only in within the
% feasible range, i.e. xMin <= x <= xMax

obj = 0.5*z'*fit.Q*z + (fit.F*x+fit.f)'*z;
constr = [z >= 0; 0 <= x <= 1];
sett = sdpsettings('solver','gurobi+');

yalmipSol = solvemp(constr,obj,[],x,z);
mptSol = mpt_mpsol2pu(yalmipSol); 
mptSol.trimFunction('primal',nVar);
mptSol.toMatlab('fitMpc.m', 'primal', 'first-region');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comparing the two controllers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear x1 x2 
fprintf('Comparing the two controllers... \n')

[X1,X2] = meshgrid(xmin(1):((xmax(1)-xmin(1))/30):xmax(1),xmin(2):((xmax(2)-xmin(2))/30):xmax(2));
x1 = X1(:); x2 = X2(:);
GT = zeros(numel(x1),1); 
NN = zeros(numel(x1),1); 

for i = 1:numel(x1)
    
    % ground-truth
    gt = expMpc.feedback.feval([x1(i);x2(i)]);
    GT(i) = gt(1);
    
    % NN surrogate (what needs to be implemented)
    % scaling the input
    temp = ([x1(i);x2(i)] - xmin) ./ xdelta;
    % regular PWA function evaluation (i.e. approx eMPC)
    [nnTemp, ~] = fitMpc(temp);
    % output affine map
    nn = fit.O*nnTemp + fit.o;
    % scaling control back
    nn = (nn*udelta) + umin;
    NN(i) = max(min(nn,umax),umin);
    
end

% plotting results
figure; 
GT = reshape(GT,size(X1));
surf(X1,X2,GT); hold on; grid on; axis([-0.05 0.15 -5 2 -0.4 0.8])
xlabel('\Deltax_1'); ylabel('\Deltax_2'); zlabel('\Deltau'); 
title('Optimal Explicit MPC');
figure; 
NN = reshape(NN,size(X1));
surf(X1,X2,NN); hold on; grid on; axis([-0.05 0.15 -5 2 -0.4 0.8])
xlabel('\Deltax_1'); ylabel('\Deltax_2'); zlabel('\Deltau');
title('NN approximation n_z = 3 (6 regions)');

% EOF