function plot_structure(q,Xsection,L,r0_ea_v,c,a,varargin)
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


if a > 1 || a < -1
    error('The elastic axis position along the chord, i.e. "a", must be declared in the range [-1,1]')
else
end
plot_options = varargin(cellfun(@(x_plt) ischar(x_plt),varargin));

switch isnumeric(varargin{1})
    case 0
        if endsWith(Xsection,'.txt') == 1
            Xsection_data = readtable(Xsection);
            Xsection_data = Xsection_data{:,:};
        elseif any(strcmp(Xsection,'Circular'))
            angle_circular = 0:pi/18:2*pi;
            x_circular = cos(angle_circular)/2 + 0.5;
            y_circular = sin(angle_circular)/2;
            Xsection_data = [x_circular.' y_circular.'];
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


element = struct("ID",{},"L",{},"C_ve",{},"r_ea1_v",{},"r_ea2_v",{});

n_element = length(q)/4;
r_aft_v = r0_ea_v(:,1);
r_aft_v_undeformed = r0_ea_v(:,1);
for i_element = 1:n_element
    element(i_element).ID = i_element;
    element(i_element).L = L/n_element;
    element(i_element).c = c(i_element);
    
    % quat = quaternion([q(i_element+n_element), q(i_element+2*n_element), q(i_element+3*n_element)],'euler','ZYX','frame');
    C_ev = DCM(1,q(i_element+n_element))*DCM(2,q(i_element+2*n_element))*DCM(3,q(i_element+3*n_element));
    element(i_element).C_ve = C_ev.';

    r_ea1_e = C_ev*r_aft_v;
    r_ea2_e = r_ea1_e + C_ev*r0_ea_v(:,2*i_element) + [element(i_element).L + q(i_element); 0; 0];

    element(i_element).r_ea1_v = element(i_element).C_ve*r_ea1_e;
    element(i_element).r_ea2_v = element(i_element).C_ve*r_ea2_e;

    r_ea1_v_undeformed = r_aft_v_undeformed;
    r_ea2_v_undeformed = r_ea1_v_undeformed + r0_ea_v(:,2*i_element) + [element(i_element).L; 0; 0];
    r_aft_v_undeformed = r_ea2_v_undeformed;

    r_aft_v = element(i_element).r_ea2_v;

    % Plotting structure:
    for i = 1:length(Xsection_data(:,1))
        Xsection_coord = element(i_element).C_ve*[0;element(i_element).c*Xsection_data(i,1)-element(i_element).c/2*(1+a);element(i_element).c*Xsection_data(i,2)];
        r1 = [element(i_element).r_ea1_v(1) + Xsection_coord(1);element(i_element).r_ea1_v(2) + Xsection_coord(2);element(i_element).r_ea1_v(3) + Xsection_coord(3)];
        r2 = [element(i_element).r_ea2_v(1) + Xsection_coord(1);element(i_element).r_ea2_v(2) + Xsection_coord(2);element(i_element).r_ea2_v(3) + Xsection_coord(3)];
        
        elem_pos1(i,:) = element(i_element).C_ve*(element(i_element).C_ve.'*r1);
        elem_pos2(i,:) = element(i_element).C_ve*(element(i_element).C_ve.'*r2);
        
        % Undeformed element:
        Xsection_coord_undeformed = [0;element(i_element).c*Xsection_data(i,1)-element(i_element).c/2*(1+a);element(i_element).c*Xsection_data(i,2)];
        r1_undeformed = [r_ea1_v_undeformed(1) + Xsection_coord_undeformed(1);r_ea1_v_undeformed(2) + Xsection_coord_undeformed(2);r_ea1_v_undeformed(3) + Xsection_coord_undeformed(3)];
        r2_undeformed = [r_ea2_v_undeformed(1) + Xsection_coord_undeformed(1);r_ea2_v_undeformed(2) + Xsection_coord_undeformed(2);r_ea2_v_undeformed(3) + Xsection_coord_undeformed(3)];

        elem_pos1_undeformed(i,:) = r1_undeformed;
        elem_pos2_undeformed(i,:) = r2_undeformed;

    end
    X3D_undeformed = [elem_pos1_undeformed(:,1) elem_pos2_undeformed(:,1)];
    Y3D_undeformed = [elem_pos1_undeformed(:,2) elem_pos2_undeformed(:,2)];
    Z3D_undeformed = [elem_pos1_undeformed(:,3) elem_pos2_undeformed(:,3)];

    X3D = [elem_pos1(:,1) elem_pos2(:,1)];
    Y3D = [elem_pos1(:,2) elem_pos2(:,2)];
    Z3D = [elem_pos1(:,3) elem_pos2(:,3)];

    p(2*i_element-1) = surf(X3D, Y3D, Z3D,'facecolor',[1 0.5 0],'linestyle','none');
    hold on

    if any(strcmp(plot_options,'DisplayUndeformed')) || any(strcmp(plot_options,'displayundeformed')) || any(strcmp(plot_options,'DISPLAYUNDEFORMED'))
        p(2*i_element) = surf(X3D_undeformed, Y3D_undeformed, Z3D_undeformed,'facecolor',[200 200 200]/255,'linestyle','none');
        if any(strcmp(plot_options,'DisplayElements')) || any(strcmp(plot_options,'displayelements')) || any(strcmp(plot_options,'DISPLAYELEMENTS'))
            plot3(X3D_undeformed,Y3D_undeformed,Z3D_undeformed,'k')
        else
            if i_element == 1
                plot3(X3D_undeformed(:,1),Y3D_undeformed(:,1),Z3D_undeformed(:,1),'k')
            elseif i_element == n_element
                plot3(X3D_undeformed(:,2),Y3D_undeformed(:,2),Z3D_undeformed(:,2),'k')
            end
        end
    end

    if any(strcmp(plot_options,'DisplayElements')) || any(strcmp(plot_options,'displayelements')) || any(strcmp(plot_options,'DISPLAYELEMENTS'))
    plot3(X3D,Y3D,Z3D,'k')
    else
        if i_element == 1
            plot3(X3D(:,1),Y3D(:,1),Z3D(:,1),'k')
        elseif i_element == n_element
            plot3(X3D(:,2),Y3D(:,2),Z3D(:,2),'k')
        end
    end

%     v(:,1) = element(i_element).C_ve*(element(i_element).C_ve.'*element(i_element).r_ea1_v + [0; element(i_element).b/2; 0]);
%     v(:,2) = element(i_element).C_ve*(element(i_element).C_ve.'*element(i_element).r_ea1_v + [0; -element(i_element).b/2; 0]);
%     v(:,3) = element(i_element).C_ve*(element(i_element).C_ve.'*element(i_element).r_ea2_v + [0; -element(i_element).b/2; 0]);
%     v(:,4) = element(i_element).C_ve*(element(i_element).C_ve.'*element(i_element).r_ea2_v + [0; element(i_element).b/2; 0]);

    %vfun = matlabFunction(v,"vars",{q});
    %vfun = vfun(position_test);
%     f = [1 2 3 4];
%     patch('Faces',f,'Vertices',v','FaceColor',[225 225 225]/255,'edgecolor','k')
%     hold on

end

if any(strcmp(plot_options,'DisplayUndeformed')) || any(strcmp(plot_options,'displayundeformed')) || any(strcmp(plot_options,'DISPLAYUNDEFORMED'))
    legend([p(1) p(2)],'Deformed','Undeformed','location','best')
else
end

colormap summer
axis equal
grid on
view(45,30)
xlabel('x_0 [m]')
ylabel('y_0 [m]')
zlabel('z_0 [m]')
%set(gcf, 'Position',  [250, 50, 800, 600])
hold off
end