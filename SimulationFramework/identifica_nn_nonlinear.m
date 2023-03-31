clc;clear all;close all
% %% parte 1: roda um monte de simulações
% g = 3;
% l = 1;
% m = 1;
% c = 0;
% k = 0;
% 
% % dot(theta) = p/(m*l^2)
% % dot(p) = -m*g*l*sin(theta) - c*dot(theta) + u
% 
% dyn = @(t,X,u)[X(2)/m/l^2;-m*g*l*sin(X(1))-k*X(1)-c*X(2)+u];
% 
% tspan = linspace(0,3,45);
% x0 = [-1.71359596; 0.12242585]
% [T X] = ode45(@(t,X)dyn(t,X,0), tspan, x0)
% 
% figure;
% plot(T,X(:,1)); hold all;
% plot(T,X(:,2)); legend('theta', 'momento');
% 
% figure;
% plot(X(:,1),X(:,2)); xlabel('theta'); ylabel('momento')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USER INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODEL DETAILS:
beam_data = 'beam_data_test.xlsx'; % Excel file containing the input properties of the dynamical structure
model = 'FOM'; % Options: 'FOM' for full-order model, 'ROM' for reduced order model by truncation method or 'BOTH' for both models analysis in a single run
DoF = {'OutBend'}; % Options: 'InBend': in-plane bending; 'OutBend': out-of-plane bending; 'Axial': axial deformation; 'Torsion': torsion
gravity = 'GravityOn'; % Options: 'GravityOn' to consider gravitational force or 'GravityOff' to disconsider it.
tspan = linspace(0,20,200);%:0.1:10; % Period of simulation [s]
Nsims = 50; % Number of data samples

% INITIAL CONDITIONS LIMITS:
p0_max = 0; % Amplitude of gen. momentum's interval (used only if IC='random')
q0_max = pi/9; % amplitude of gen. coordinate's interval (used only if IC='random')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath background
addpath background\CrossSectionData
addpath background\utils
global beam

% Builds dynamical system:
[M,I,K,C] = structure_properties(beam_data,DoF);
load beam_data.mat beam
n_DoF = length(DoF)*sum(cat(1,beam.n_element));
%K=1e-2*K;
%K(1,1) = 0;
%C(1,1) = 0;
%C=2*C;

% Creates dataset:
progressbar('creating dataset...')
input = zeros(Nsims*length(tspan),2*n_DoF); 
output = zeros(Nsims*length(tspan),2*n_DoF);
for j = 1:Nsims
    p0 = p0_max*(2*rand(n_DoF,1)-1);
    q0 = q0_max*(2*rand(n_DoF,1)-1);
    X0 = [p0;q0];

    [X,Xdot] = simulation(model,DoF,gravity,tspan,M,I,K,C,X0);

    input(j*length(tspan)-length(tspan)+1:j*length(tspan),:) = X;
    output(j*length(tspan)-length(tspan)+1:j*length(tspan),:) = Xdot;
    progressbar(j/Nsims);
end

%% parte 2: treina a rede
%net = feedforwardnet([10 10 10]);
net = feedforwardnet([256]);
%net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'radbas';
%net.layers{1}.transferFcn = 'purelin';
net.layers{1}.transferFcn = 'tansig';
net = train(net, input', output');

%X = input \ output; % treino de uma solução linear do tipo Xn+1 = A*Xn+Bun
                    % por minimos quadrados (pseudoinversa)
%% parte 3: testa a rede

x0 = randn(2,1);

dyn_nn = @(t,X) net(X);

[~,Y] = ode45(@(t,X)dyn(t,X,0), tspan, x0);
[~,Ynn] = ode45(@(t,X)dyn_nn(t,X), tspan, x0);

figure;
plot(Ynn(:,1), Ynn(:,2)); hold all;
plot(Y(:,1), Y(:,2), 'k--');
legend('NN', 'pendulum')
figure;
subplot(211);
plot(tspan, Ynn(:,1)); hold all;
plot(tspan, Y(:,1), 'k--');
subplot(212)
plot(tspan, Ynn(:,2)); hold all;
plot(tspan, Y(:,2), 'k--');
