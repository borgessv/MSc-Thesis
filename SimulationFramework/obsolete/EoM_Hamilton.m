clear 
close all
clc

load beams_test.mat beam

g = 9.8067;
for i_beam = 1:length(beam)
    n = beam(i_beam).n_element;

    q = sym('q', [1,4*n]).';
    qdot = sym('q%ddot', [1,4*n]).';
    p = sym('p', [1,4*n]).';

    r0x(1) = sym(zeros(1,1));
    r0y(1) = sym(zeros(1,1));
    r0z(1) = sym(zeros(1,1));

    for i_element = 1:n
        r1x(i_element) = r0x(i_element) + (beam(i_beam).L_element + q(i_element))*cos(deg2rad(beam(i_beam).Gamma_deg) + sum(q(2*n+1:2*n+i_element)))*cos(deg2rad(beam(i_beam).Lambda_deg) + sum(q(3*n+1:3*n+i_element)));
        r1y(i_element) = r0y(i_element) + (beam(i_beam).L_element + q(i_element))*cos(deg2rad(beam(i_beam).Gamma_deg) + sum(q(2*n+1:2*n+i_element)))*sin(deg2rad(beam(i_beam).Lambda_deg) + sum(q(3*n+1:3*n+i_element)));
        r1z(i_element) = r0z(i_element) + (beam(i_beam).L_element + q(i_element))*sin(deg2rad(beam(i_beam).Gamma_deg) + sum(q(2*n+1:2*n+i_element)));

        rCMx(i_element) = (r0x(i_element) + r1x(i_element))/2;
        rCMy(i_element) = (r0y(i_element) + r1y(i_element))/2;
        rCMz(i_element) = (r0z(i_element) + r1z(i_element))/2;

        % Jpos = lineariza_complexstep(matlabFunction([rCMx(i_element); rCMy(i_element); rCMz(i_element)],'vars',{q}), zeros(4*n,1));

        Qg(i_element,:) = dot(repmat([0; 0; -beam(i_beam).element(i_element).m*g],1,4*n),jacobian([rCMx(i_element) rCMy(i_element) rCMz(i_element)],q));
        
        if i_element < beam(i_beam).n_element
            r0x(i_element+1) = r1x(i_element);
            r0y(i_element+1) = r1y(i_element);
            r0z(i_element+1) = r1z(i_element);
        else
        end
    end
    beam(i_beam).r0 = [r0x.' r0y.' r0z.'];
    beam(i_beam).r1 = [r1x.' r1y.' r1z.'];
    beam(i_beam).rCM = [rCMx.' rCMy.' rCMz.'];

    rCMdot = jacobian(beam(i_beam).rCM(:).',q)*qdot;
    %Jpos = lineariza_complexstep(matlabFunction(beam(i_beam).rCM,'vars',{q}), zeros(1,4*n));

    M = diag(reshape(repmat(vertcat(beam(i_beam).element.m),1,3).',[],1)); % Mass matrix
    T = 0.5*rCMdot.'*M*rCMdot; % Kinetic energy of the system

    K = stiffness_test; % Stiffness matrix 
    V = sum(0.5*K*(q.^2)); % Potential energy of the system

    L = T - V; % Lagragean of the system
    H = T + V; % Hamiltonian of the system

    P = jacobian(L,qdot);
    
    sol = solve([p(1:n);p(2*n+1:end)] - [P(1:n) P(2*n+1:end)].' == 0, [qdot(1:n);qdot(2*n+1:end)]);
    H = subs(H,[qdot(1:n);qdot(2*n+1:end)],struct2cell(sol));

    dHdq = jacobian(H,q);
    dHdp = jacobian(H,p);

    p_dot = -dHdq.' + sum(Qg,1).';
    q_dot = dHdp.';
    %eqs = [p_dot; q_dot];

    X0 = zeros(8*n,1);
    t0 = 0;
    tF = 5;
    
    %X = [p1; p2; p5; p6; p7; p8; q1; q2; q5; q6; q7; q8];
    syms t
    %dinamica = matlabFunction(eqs, 'Vars', {t, [[p(1:n);p(2*n+1:end)]; [q(1:n);q(2*n+1:end)]]});
    [T,X] = ode45(@dinamica, [t0 tF], X0);


end


function eqs = dinamica(t,p_dot,q_dot)

eqs = [p_dot; q_dot];


end