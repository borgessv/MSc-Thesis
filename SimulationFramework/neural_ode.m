clear all; close all; clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USER INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODEL DETAILS:
beam_data = 'beam_data_test.xlsx'; % Excel file containing the input properties of the dynamical structure
model = 'FOM'; % Options: 'FOM' for full-order model, 'ROM' for reduced order model by truncation method or 'BOTH' for both models analysis in a single run
DoF = {'OutBend','Torsion'}; % Options: 'InBend': in-plane bending; 'OutBend': out-of-plane bending; 'Axial': axial deformation; 'Torsion': torsion
gravity = 'GravityOn'; % Options: 'GravityOn' to consider gravitational force or 'GravityOff' to disconsider it.
tspan = linspace(0,10,100);%:0.1:10; % Period of simulation [s]
Nsims = 50; % Number of data samples

% INITIAL CONDITIONS LIMITS:
p0_max = 2; 
q0_max = pi/6; 
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
    progressbar(j/Nsims*100);
end

%% parte 2: treina a rede
net = feedforwardnet([50 50 50]);
% net = feedforwardnet([10]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{1}.transferFcn = 'purelin';
% net.layers{1}.transferFcn = 'tansig';
net.trainParam.max_fail = 1e6;
net = train(net, input', output');

%% parte 3: testa a rede
x0 = [0;0;0;0;0;0;-pi/12;-pi/9;-pi/6;-pi/12;pi/6;pi/4];
tspan = linspace(0,30,1000);
dyn_nn = @(t,X) net(X);
[~,Xnn] = ode45(@(t,X)dyn_nn(t,X), tspan, x0);
[X,~] = simulation(model,DoF,gravity,tspan,M,I,K,C,x0);

%% Plots:
element = 2;
figure;
plot(rad2deg(Xnn(:,n_DoF+element)), Xnn(:,element),'b'); hold all;
plot(rad2deg(X(:,n_DoF+element)), X(:,element), 'k--');
xlabel('q [deg]'); ylabel('p [kg*rad/s]'); legend('NN', 'pendulum')

figure;
subplot(211);
plot(tspan, Xnn(:,n_DoF+element)); hold all;
plot(tspan, X(:,n_DoF+element), 'k--');
xlabel('t [s]'); ylabel('q [deg]'); legend('NN', 'pendulum')
subplot(212)
plot(tspan, Xnn(:,element)); hold all;
plot(tspan, X(:,element), 'k--');
xlabel('t [s]'); ylabel('p [kg*rad/s]'); legend('NN', 'pendulum')
