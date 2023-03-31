function create_beam(beam_data,n_element)

% Reading input file and creating arrays for the properties of the beams:
beam_properties = readmatrix(beam_data);
beam_properties = beam_properties(:,2:end);
ID_beam = beam_properties(1,:);
L_beam = beam_properties(2,:);
m0_beam = beam_properties(3,:);
m1_beam = beam_properties(4,:);
c0_beam = beam_properties(5,:);
c1_beam = beam_properties(6,:);
yCM_beam = beam_properties(7,:);
zCM_beam = beam_properties(8,:);
Lambda_beam_deg = beam_properties(9,:);
Gamma_beam_deg = beam_properties(10,:);
iw_beam_deg = beam_properties(11,:);
connectivity_beam = beam_properties(12,:);
connection_point_beam = beam_properties(13,:);

n_beam = length(ID_beam);
beam(1:n_beam) = struct('L',0,'cmasses',0,...
    'L',0,'EIxx',0,'EIzz',0,'EIxz',0,'GJ',0,...
    'm',0,'Iyy',0,'xC',0,'zC',0,'C_i0',eye(3));

syms x
for i_beam = 1:n_beam
    beam(i_beam).C_i0 = DCM(1,deg2rad(iw_beam_deg(i_beam)))*DCM(2,deg2rad(Gamma_beam_deg(i_beam)))*DCM(3,deg2rad(Lambda_beam_deg(i_beam)));
    beam(i_beam).L_element = L_beam(i_beam)/n_element;
    beam(i_beam).yCM = yCM_beam(i_beam);
    beam(i_beam).zCM = zCM_beam(i_beam);
    beam(i_beam).m = m0_beam(i_beam) + m1_beam(i_beam)*x;
    for i_element = 1:n_element
        if isnan(connectivity_beam(i_beam))
            if i_element == 1 && isnan(connection_point_beam(i_beam))
                beam(i_beam).element(i_element).x0 = 0;
            elseif i_element == 1 && ~isnan(connection_point_beam(i_beam))
                beam(i_beam).element(i_element).x0 = connection_point_beam(i_beam);
            else
                beam(i_beam).element(i_element).x0 = beam(i_beam).element(i_element-1).x1;
            end
        else
            beam(i_beam).element(i_element).x0 = beam(connectivity_beam(i_beam)).C_i0(:,1)*connection_point_beam(connectivity_beam(i_beam));
        end





    end
end


addpath quaternion_symbolic\
airfoil_data = readtable('NACA_2412.txt');
airfoil_data = airfoil_data{:,:};

q = sym('q',[4*n_element 1],'real');
element = struct("ID",{},"L",{},"C_ve",{},"r_ea1_v",{},"r_ea2_v",{});

r_aft_v = [0; 0; 0];

test = 0;
test = linspace(test,pi/6,n_element).';
q = [0*rand(n_element,1);2*test;test;test];
figure

for i_element = 1:n_element
    element(i_element).ID = i_element;
    element(i_element).L = L/n_element;
    element(i_element).b = b;
    
    % quat = quaternion([q(i_element+n_element), q(i_element+2*n_element), q(i_element+3*n_element)],'euler','ZYX','frame');
    C_ev = DCM(1,q(i_element+n_element))*DCM(2,q(i_element+2*n_element))*DCM(3,q(i_element+3*n_element));
    element(i_element).C_ve = C_ev.';

    r_ea1_e = C_ev*r_aft_v;
    r_ea2_e = r_ea1_e + [element(i_element).L + q(i_element); 0; 0];

    element(i_element).r_ea1_v = element(i_element).C_ve*r_ea1_e;
    element(i_element).r_ea2_v = element(i_element).C_ve*r_ea2_e;

    r_aft_v = element(i_element).r_ea2_v;

    % Plotting structure:
    for i = 1:length(airfoil_data(:,1))
        aux = element(i_element).C_ve*[0;airfoil_data(i,1)-0.25;airfoil_data(i,2)];
        r1 = [element(i_element).r_ea1_v(1) + aux(1);element(i_element).r_ea1_v(2) + aux(2);element(i_element).r_ea1_v(3) + aux(3)];
        r2 = [element(i_element).r_ea2_v(1) + aux(1);element(i_element).r_ea2_v(2) + aux(2);element(i_element).r_ea2_v(3) + aux(3)];

        pos1(i,:) = element(i_element).C_ve*(element(i_element).C_ve.'*r1);
        pos2(i,:) = element(i_element).C_ve*(element(i_element).C_ve.'*r2);

    end

    Y3D = [pos1(:,2) pos2(:,2)];
    X3D = [pos1(:,1) pos2(:,1)];
    Z3D = [pos1(:,3) pos2(:,3)];

    surf(X3D, Y3D, Z3D,'linestyle','none');
    hold on
    if i_element == 1
        plot3(X3D(:,1),Y3D(:,1),Z3D(:,1),'k')
    elseif i_element == n_element
        plot3(X3D(:,2),Y3D(:,2),Z3D(:,2),'k')
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
colormap copper
axis equal
grid on
%view(45,30)
xlabel('x_0')
ylabel('y_0')
zlabel('z_0')
end