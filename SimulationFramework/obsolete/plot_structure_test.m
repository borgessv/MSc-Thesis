function plot_structure_test(Q,beam_data,varargin)
%
% plot_structure(q,Xsection,L,r0_ea_v,c,a,varargin) plots the deformed
% beam in a multibody analysis considering axial, in-plane bending,
% out-of-plane bending and torsion displacements.
%
% Inputs:
%       - q: vector of generalized coordinates in the sequence [q_axial;
%       q_torsion; q_outofplane; q_inplane] such that the size of q is
%       4*number-of-elements;
%       - Xsection: coordinate data file (.txt) of the cross-section
%       considered in the analysis. The inputs 'Rectangular' and 'Circular'
%       can be used instead, but if 'Rectangular' the tickness of the first
%       elementmust be informed in varargin{1};
%       - L: undeformed total length/span of the beam;
%       - r0_ea_v: initial/undeformed position of each element's elastic
%       axis. Must have the same size as q;
%       - c: chord/width/diameter of the beam. Must be specified for each
%       element, therefore its size must be equal the number of elements;
%       - a: elastic axis position along the chord/width of the beam. Must
%       be in the range [-1,1], where 0 is the center, i.e. c/2;
%       - varargin: optional arguments:
%           - d: tickness of the first element if 'Rectangular' is used
%           for Xsection variable.
%           Must be the first argument of varargin;
%           - 'DisplayElements': plot the contour of each element;
%           - 'DisplayUndeformed': plot the undeformed beam;
%
% Outputs:
%       - 3D plot of the deformed and undeformed (optional) beam

addpath CrossSectionData
load(beam_data,'beam')
plot_options = varargin(cellfun(@(x_plt) ischar(x_plt),varargin));

for i_beam = 1:length(beam)
    Xsection = beam(i_beam).Xsection;
    if beam(i_beam).Lambda_deg < 0 || beam(i_beam).Lambda_deg > 180
        a = -1;
    else
        a = 1;
    end
    switch isnumeric(varargin{1})
        case 0
            if endsWith(Xsection,'.txt') == 1
                Xsection_data = readtable(Xsection);
                Xsection_data = Xsection_data{:,:};
                Xsection_data(:,3) = zeros(length(Xsection_data),1);
            elseif any(strcmp(Xsection,'Circular'))
                angle_circular = 0:pi/18:2*pi;
                x_circular = cos(angle_circular)/2 + 0.5;
                y_circular = sin(angle_circular)/2 + 0.5;
                Xsection_data = [zeros(length(x_circular),1) x_circular.' y_circular.'];
            else
                error(['Cross-section .txt data file is missing. For ''Rectangular''' ...
                    ' cross-section the tickness of the first element must be specified in varargin{1}.'])
            end
        case 1
            d = varargin{1};
            if any(strcmp(Xsection,'Rectangular'))
                Xsection_data = [0 -d/2; 0 d/2; 1 d/2; 1 -d/2; 0 -d/2];
            else
                error(['varargin{1} is numeric. You must delete it in order to use ' ...
                    'a .txt data file or specify ''Rectangular'' for the cross-section input variable'])
            end
    end
    if i_beam == 1
        r0_0_def = beam(i_beam).element(1).r0;
    else
        r0_0_def = beam(beam(i_beam).connectivity).C_i0.'*beam(beam(i_beam).connectivity).element(beam(i_beam).connection_element).C_di.'*(beam(beam(i_beam).connectivity).element(beam(i_beam).connection_element).r0_d + beam(i_beam).connection_element_point*(beam(beam(i_beam).connectivity).element(beam(i_beam).connection_element).r1_d - beam(beam(i_beam).connectivity).element(beam(i_beam).connection_element).r0_d));
    end

    for i_element = 1:beam(i_beam).n_element
        r0_i_undef = beam(i_beam).C_i0*beam(i_beam).element(i_element).r0;
        r1_i_undef = beam(i_beam).C_i0*beam(i_beam).element(i_element).r1;

        if isa(Q,'double')
        else
            q = Q(i_beam).q;
            beam(i_beam).element(i_element).C_di = DCM(1,q(i_element+beam(i_beam).n_element))*DCM(2,q(i_element+2*beam(i_beam).n_element))*DCM(3,q(i_element+3*beam(i_beam).n_element));
            beam(i_beam).element(i_element).r0_d = beam(i_beam).element(i_element).C_di*beam(i_beam).C_i0*r0_0_def;
            beam(i_beam).element(i_element).r1_d = beam(i_beam).element(i_element).r0_d + [beam(i_beam).L_element + q(i_element);0;0];
            r0_0_def = beam(i_beam).C_i0.'*beam(i_beam).element(i_element).C_di.'*beam(i_beam).element(i_element).r1_d;

            if ~isnan(beam(i_beam).connectivity)
                Xsection_coord_def = (DCM(1,a*deg2rad(beam(i_beam).AoI_deg))*DCM(2,a*deg2rad(beam(i_beam).Gamma_deg))*DCM(3,pi/2)).'*(DCM(1,a*q(i_element+beam(i_beam).n_element))*DCM(2,a*q(i_element+2*beam(i_beam).n_element))*DCM(3,q(i_element+3*beam(i_beam).n_element))).'*(beam(beam(i_beam).connectivity).element(beam(i_beam).connection_element).C_di.')*[(beam(i_beam).element(i_element).c.*Xsection_data(:,3)-beam(i_beam).element(i_element).c*beam(i_beam).xCM).';(beam(i_beam).element(i_element).c.*Xsection_data(:,1)-beam(i_beam).element(i_element).c*beam(i_beam).yCM).';((beam(i_beam).element(i_element).c*Xsection_data(:,2))-beam(i_beam).element(i_element).c*beam(i_beam).zCM).'];

                r0_Xsection_0_def = beam(i_beam).C_i0.'*beam(i_beam).element(i_element).C_di.'*beam(i_beam).element(i_element).r0_d + Xsection_coord_def;
                r1_Xsection_0_def = beam(i_beam).C_i0.'*beam(i_beam).element(i_element).C_di.'*beam(i_beam).element(i_element).r1_d + Xsection_coord_def;
            else
                Xsection_coord_def = (DCM(1,a*deg2rad(beam(i_beam).AoI_deg))*DCM(2,a*deg2rad(beam(i_beam).Gamma_deg))*DCM(3,pi/2)).'*(DCM(1,a*q(i_element+beam(i_beam).n_element))*DCM(2,a*q(i_element+2*beam(i_beam).n_element))*DCM(3,q(i_element+3*beam(i_beam).n_element))).'*[(beam(i_beam).element(i_element).c.*Xsection_data(:,3)-beam(i_beam).element(i_element).c*beam(i_beam).xCM).';(beam(i_beam).element(i_element).c.*Xsection_data(:,1)-beam(i_beam).element(i_element).c*beam(i_beam).yCM).';((beam(i_beam).element(i_element).c*Xsection_data(:,2))-beam(i_beam).element(i_element).c*beam(i_beam).zCM).'];

                r0_Xsection_0_def = beam(i_beam).C_i0.'*beam(i_beam).element(i_element).C_di.'*beam(i_beam).element(i_element).r0_d + Xsection_coord_def;
                r1_Xsection_0_def = beam(i_beam).C_i0.'*beam(i_beam).element(i_element).C_di.'*beam(i_beam).element(i_element).r1_d + Xsection_coord_def;

            end
%             r0d = beam(i_beam).C_i0.'*beam(i_beam).element(i_element).C_di.'*beam(i_beam).element(i_element).r0_d;
%             plot3(r0d(1),r0d(2),r0d(3),'or')
            X3D_def = [r0_Xsection_0_def(1,:).' r1_Xsection_0_def(1,:).'];
            Y3D_def = [r0_Xsection_0_def(2,:).' r1_Xsection_0_def(2,:).'];
            Z3D_def = [r0_Xsection_0_def(3,:).' r1_Xsection_0_def(3,:).'];
            surf(X3D_def, Y3D_def, Z3D_def,'facecolor',[1 0.5 0],'linestyle','none');
            hold on
            if any(strcmp(plot_options,'DisplayElements')) || any(strcmp(plot_options,'displayelements')) || any(strcmp(plot_options,'DISPLAYELEMENTS'))
                plot3(X3D_def,Y3D_def,Z3D_def,'color',[0.5 0.5 0.5])
            else
                if i_element == 1
                    plot3(X3D_def(:,1),Y3D_def(:,1),Z3D_def(:,1),'color',[0.5 0.5 0.5])
                elseif i_element == beam(i_beam).n_element
                    plot3(X3D_def(:,2),Y3D_def(:,2),Z3D_def(:,2),'color',[0.5 0.5 0.5])
                end
            end
        end

        % Plotting undeformed shape:
        if any(strcmp(plot_options,'DisplayUndeformed')) || any(strcmp(plot_options,'displayundeformed')) || any(strcmp(plot_options,'DISPLAYUNDEFORMED'))
            Xsection_coord_undef = (DCM(1,a*deg2rad(beam(i_beam).AoI_deg))*DCM(2,a*deg2rad(beam(i_beam).Gamma_deg))*DCM(3,pi/2)).'*[(beam(i_beam).element(i_element).c.*Xsection_data(:,3)-beam(i_beam).element(i_element).c*beam(i_beam).xCM).';(beam(i_beam).element(i_element).c.*Xsection_data(:,1)-beam(i_beam).element(i_element).c*beam(i_beam).yCM).';((beam(i_beam).element(i_element).c*Xsection_data(:,2))-beam(i_beam).element(i_element).c*beam(i_beam).zCM).'];
            r0_Xsection_0 = beam(i_beam).C_i0.'*r0_i_undef + Xsection_coord_undef;
            r1_Xsection_0 = beam(i_beam).C_i0.'*r1_i_undef + Xsection_coord_undef;
            
            X3D_undef = [r0_Xsection_0(1,:).' r1_Xsection_0(1,:).'];
            Y3D_undef = [r0_Xsection_0(2,:).' r1_Xsection_0(2,:).'];
            Z3D_undef = [r0_Xsection_0(3,:).' r1_Xsection_0(3,:).'];

%             r0 = beam(i_beam).C_i0.'*r0_i_undef;
%             plot3(r0(1),r0(2),r0(3),'ob')
            surf(X3D_undef, Y3D_undef, Z3D_undef,'facecolor',[0.9 0.9 0.9],'linestyle','none');
            hold on
            
            if any(strcmp(plot_options,'DisplayElements')) || any(strcmp(plot_options,'displayelements')) || any(strcmp(plot_options,'DISPLAYELEMENTS'))
                plot3(X3D_undef,Y3D_undef,Z3D_undef,'color',[0.5 0.5 0.5])
            else
                if i_element == 1
                    plot3(X3D_undef(:,1),Y3D_undef(:,1),Z3D_undef(:,1),'color',[0.5 0.5 0.5])
                elseif i_element == beam(i_beam).n_element
                    plot3(X3D_undef(:,2),Y3D_undef(:,2),Z3D_undef(:,2),'color',[0.5 0.5 0.5])
                end
            end
        else
        end
    end
end
axis equal

xlabel('x_0 [m]')
ylabel('y_0 [m]')
zlabel('z_0 [m]')
end