clear 
close all
clc

load beams_test.mat beam

g = 9.8067;
for i_beam = 1:length(beam)
    syms q(t) [1 4*beam(1).n_element]
    beam(i_beam).q = q(t);
    r0(:,1) = sym(zeros(3,1));
    r1 = sym(zeros(3,beam(i_beam).n_element));
    r_CM = sym(zeros(3,beam(i_beam).n_element));
    %r_beam = sym(zeros(3,2*beam(i_beam).n_element));
    for i_element = 1:beam(i_beam).n_element
        r1(:,i_element) = [
            r0(1,i_element) + (beam(i_beam).L_element + beam(i_beam).q(i_element))*cos(deg2rad(beam(i_beam).Gamma_deg) + sum(beam(i_beam).q(2*beam(i_beam).n_element+1:2*beam(i_beam).n_element+i_element)))*cos(deg2rad(beam(i_beam).Lambda_deg) + sum(beam(i_beam).q(3*beam(i_beam).n_element+1:3*beam(i_beam).n_element+i_element)))
            r0(2,i_element) + (beam(i_beam).L_element + beam(i_beam).q(i_element))*cos(deg2rad(beam(i_beam).Gamma_deg) + sum(beam(i_beam).q(2*beam(i_beam).n_element+1:2*beam(i_beam).n_element+i_element)))*sin(deg2rad(beam(i_beam).Lambda_deg) + sum(beam(i_beam).q(3*beam(i_beam).n_element+1:3*beam(i_beam).n_element+i_element)))
            r0(3,i_element) + (beam(i_beam).L_element + beam(i_beam).q(i_element))*sin(deg2rad(beam(i_beam).Gamma_deg) + sum(beam(i_beam).q(2*beam(i_beam).n_element+1:2*beam(i_beam).n_element+i_element)))
            ];
        r_CM(:,i_element) = (r0(:,i_element) + r1(:,i_element))/2;
        rdot_CM(:,i_element) = sum(jacobian(r_CM(:,i_element),t),2);
        T(i_element) = 1/2*beam(i_beam).element(i_element).m*norm(rdot_CM(:,i_element))^2;
        
        % Force of Gravity:
        F_g(i_element) = dot([0; 0; -beam(i_beam).element(i_element).m*g],r_CM(:,i_element));
        %r_beam(:,2*i_element-1) = r0(:,i_element); 
        %r_beam(:,2*i_element) = r1(:,i_element);

        
        if i_element < beam(i_beam).n_element
        r0(:,i_element+1) = r1(:,i_element);
        else
        end
    end
    beam(i_beam).r = sym(zeros(3,2*beam(i_beam).n_element));
    beam(i_beam).r(:,1:2:end-1) = r0; 
    beam(i_beam).r(:,2:2:end) = r1;

    K = stiffness_test;
    V = 0.5*K*(q.^2.');
    L = sum(T) - sum(V);
    H = sum(T) + sum(V);
    Q_e = sum(F_g);
    % Gravity:
    

    % Euler-Lagrange Equations of Motion:
    for i_q = 1:4*beam(i_beam).n_element    
        P_q(i_q) = diff(L,diff(beam(i_beam).q(i_q),t));
        EL(i_q) = diff(diff(L,diff(beam(i_beam).q(i_q),t)),t) - diff(L,beam(i_beam).q(i_q)) == diff(Q_e,beam(i_beam).q(i_q));
    end
end

r_beam = double(subs(beam(i_beam).r,beam.q,[linspace(0,0.2,beam(i_beam).n_element) repmat(linspace(pi/18,pi/18,beam(i_beam).n_element),1,2) linspace(0,pi/18,beam(i_beam).n_element)]));
figure
plot3(r_beam(1,:),r_beam(2,:),r_beam(3,:),'-ob')
axis equal
grid on