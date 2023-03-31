function Xdot = dynamics(t,X,M,I,K,C,DoF,gravity,model,phi_r)
%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTION OVERVIEW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   The function dynamics(t,X,M,K,C) returns the equations of motion of the
%   system using Hamiltonian mechanics.
%   INPUTS:
%       t: time;
%       X: generalized coordinates vector;
%       M: mass matrix of the system (obtained previously by the function
%       mass_matrix);
%       K: stiffness matrix of the system (obtained previously by the
%       function stiffness_matrix);
%       C: damping matrix (must be calculated previously or assumed e.g. as
%       c*K, where c is the damping coefficient.
%   OUTPUT:
%       Xdot: vector containing the time derivatives of the generalized
%       momentum and coordinates, i.e. the Hamiltonian equations of motion.
%
%   Author: Vitor Borges Santos - borgessv93@gmail.com
%   Date: 15/04/22
%
%%%%%%%%%%%%%%%%%%%%%%%% BEGINNING OF FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%
global beam
n_DoF = length(DoF)*sum(cat(1,beam.n_element));
if any(strcmp(model,'FOM'))
    %     [~,JqCM] = dlfeval(@(q) element_positionCM(q,DoF,'gradient'),dlarray(X(n_DoF+1:end)));
    %     JqCM = extractdata(JqCM);
    JqCM = complexstep(@(q) element_positionCM(q,DoF),X(n_DoF+1:end)); % Numerical calculation of the Jacobian of the generalized coordinates using the complex step approach
    [Q,~,~,~] = generalized_force(t,X(n_DoF+1:end),JqCM,DoF,gravity);

    qdot = (JqCM.'*M*JqCM + I)\X(1:n_DoF);
    pdot = -K*X(n_DoF+1:end) - C*qdot + Q;

    Xdot = [pdot;qdot];

elseif any(strcmp(model,'ROM'))
    r = size(phi_r,2);
    JqCM = complexstep(@(q) element_positionCM(q,DoF),phi_r*X(r+1:end)); % Numerical calculation of the Jacobian of the generalized coordinates using the complex step approach
    [Q,~,~,~] = generalized_force(t,phi_r*X(r+1:end),JqCM,DoF,gravity);

    etadot = (phi_r.'*JqCM.'*M*JqCM*phi_r + phi_r.'*I*phi_r)\X(1:r);
    prdot = -(phi_r.'*K*phi_r)*X(r+1:end) - (phi_r.'*C*phi_r)*etadot + phi_r.'*Q;

    Xdot = [prdot;etadot];
else
    error('The type of the model must be specified! Choose "FOM" if full-order model or "ROM" if reduced order model')
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%% END OF FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%