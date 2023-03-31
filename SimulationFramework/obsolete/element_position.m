function rCM = element_position(q,DoF)

global beam

for i_beam = 1:length(beam)
    n = beam(i_beam).n_element;
    beam(i_beam).r0 = zeros(n,3);
    beam(i_beam).r1 = zeros(n,3);
    beam(i_beam).rCM = zeros(n,3);
    r0_0_element = beam(i_beam).element(1).r0;
    
    for i_element = 1:n
        C_i0 = beam(i_beam).C_i0;
        C_di = eye(3);
        
        if any(strcmp(DoF,'InBend')) && any(strcmp(DoF,'OutBend'))
            C_di = DCM(3,sum(q(n+1:i_element+n)));
        elseif any(strcmp(DoF,'InBend')) && ~any(strcmp(DoF,'OutBend'))
            C_di = DCM(3,sum(q(1:i_element)));
        end
        
        if any(strcmp(DoF,'OutBend'))
            C_di = DCM(2,sum(q(1:i_element)))*C_di;
        end

        if any(strcmp(DoF,'Torsion')) && any(strcmp(DoF,'InBend')) && any(strcmp(DoF,'OutBend')) && any(strcmp(DoF,'Axial'))
            C_di = DCM(1,sum(q(3*n+1:i_element+3*n)))*C_di;
        elseif any(strcmp(DoF,'Torsion')) && any(strcmp(DoF,'InBend')) && any(strcmp(DoF,'OutBend')) && ~any(strcmp(DoF,'Axial')) || any(strcmp(DoF,'Torsion')) && any(strcmp(DoF,'InBend')) && ~any(strcmp(DoF,'OutBend')) && any(strcmp(DoF,'Axial')) || any(strcmp(DoF,'Torsion')) && ~any(strcmp(DoF,'InBend')) && any(strcmp(DoF,'OutBend')) && any(strcmp(DoF,'Axial'))
            C_di = DCM(1,sum(q(2*n+1:i_element+2*n)))*C_di;
        elseif any(strcmp(DoF,'Torsion')) && ~any(strcmp(DoF,'InBend')) && any(strcmp(DoF,'OutBend')) && ~any(strcmp(DoF,'Axial')) || any(strcmp(DoF,'Torsion')) && any(strcmp(DoF,'InBend')) && ~any(strcmp(DoF,'OutBend')) && ~any(strcmp(DoF,'Axial')) || any(strcmp(DoF,'Torsion')) && ~any(strcmp(DoF,'InBend')) && ~any(strcmp(DoF,'OutBend')) && any(strcmp(DoF,'Axial'))
            C_di = DCM(1,sum(q(n+1:i_element+n)))*C_di;
        elseif any(strcmp(DoF,'Torsion')) && ~any(strcmp(DoF,'InBend')) && ~any(strcmp(DoF,'OutBend')) && ~any(strcmp(DoF,'Axial'))
            C_di = DCM(1,sum(q(1:i_element)))*C_di;
        end

        C_d0 = C_di*C_i0; % Transformation matrix from the global axis to the deformed axis

        if any(strcmp(DoF,'Axial')) && any(strcmp(DoF,'InBend')) && any(strcmp(DoF,'OutBend'))
            r1_d_element = C_d0*r0_0_element + [beam(i_beam).L_element + q(i_element+2*n);0;0];
        elseif any(strcmp(DoF,'Axial')) && ~any(strcmp(DoF,'InBend')) && any(strcmp(DoF,'OutBend')) || any(strcmp(DoF,'Axial')) && any(strcmp(DoF,'InBend')) && ~any(strcmp(DoF,'OutBend'))
            r1_d_element = C_d0*r0_0_element + [beam(i_beam).L_element + q(i_element+n);0;0];
        elseif any(strcmp(DoF,'Axial')) && ~any(strcmp(DoF,'InBend')) && ~any(strcmp(DoF,'OutBend'))
            r1_d_element = C_d0*r0_0_element + [beam(i_beam).L_element + q(i_element);0;0];
        elseif ~any(strcmp(DoF,'Axial'))
            r1_d_element = C_d0*r0_0_element + [beam(i_beam).L_element;0;0];
        end
        
        beam(i_beam).element(i_element).C_d0 = C_d0;
        beam(i_beam).r0(i_element,:) = r0_0_element.';
        beam(i_beam).r1(i_element,:) = (C_d0.'*r1_d_element).';
        beam(i_beam).rCM(i_element,:) = (beam(i_beam).r0(i_element,:) + beam(i_beam).r1(i_element,:))/2; 
        r0_0_element = C_d0.'*r1_d_element;
    end
    rCM = reshape(beam(i_beam).rCM,[],1);
end
end