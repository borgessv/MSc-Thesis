function plot_structure(X,DoF,type,varargin)

global beam

if ~isempty(varargin) && isnumeric(varargin{end})
    X_r = varargin{end};
end

figure
%set(gcf, 'Position',  [250, 42, 750, 645])
set(gcf,'color','w');

for i_beam = 1:length(beam)
    if any(strcmp(type,'1D'))
        element_positionCM(X,DoF);
        r = [beam(i_beam).r0(1,:);beam(i_beam).r1];
        rCM = beam(i_beam).rCM;
        r(:,2) = [rCM(1,2);rCM(:,2)];
        fig1 = plot3(r(:,1),r(:,2),r(:,3),'-|k','markersize',2,'markerfacecolor','k');
        hold on
        plot3(rCM(:,1),rCM(:,2),rCM(:,3),'sr','markersize',2,'markerfacecolor','r');
        xlabel('x [m]','interpreter','latex')
        ylabel('y [m]','interpreter','latex')
        zlabel('z [m]','interpreter','latex')
        title('Equilibrium Solution','interpreter','latex')
        grid on
        axis equal
        xlim([(min(r(:,1))-max(vertcat(beam.element.c))) (max(r(:,1))+max(vertcat(beam.element.c)))])
        ylim([(min(r(:,2))-max(vertcat(beam.element.c))) (max(r(:,2))+max(vertcat(beam.element.c)))])
        zlim([(min(r(:,3))-max(vertcat(beam.element.c))) (max(r(:,3))+max(vertcat(beam.element.c)))])

        if ~isempty(varargin) && isnumeric(varargin{end})
            element_positionCM(X_r,DoF);
            r_ROM = [beam(i_beam).r0(1,:);beam(i_beam).r1];
            rCM_ROM = beam(i_beam).rCM;
            r_ROM(:,2) = [rCM_ROM(1,2);rCM_ROM(:,2)];
            fig3 = plot3(r_ROM(:,1),r_ROM(:,2),r_ROM(:,3),'-|b','markersize',2,'markerfacecolor','b');
            plot3(rCM_ROM(:,1),rCM_ROM(:,2),rCM_ROM(:,3),'sg','markersize',2,'markerfacecolor','g');
            legend([fig1 fig3],'FOM','ROM','location','northeast')
        end

    elseif any(strcmp(type,'3D'))
        element_positionCM(X,DoF);
        r = [beam(i_beam).r0(1,:);beam(i_beam).r1];
        rCM = beam(i_beam).rCM;
        r(:,2) = [rCM(1,2);rCM(:,2)];
        Xsection = beam(i_beam).Xsection;
        if endsWith(Xsection,'.txt') == 1
            Xsection_data = readtable(Xsection);
            Xsection_data = Xsection_data{:,:};
            Xsection_data(:,3) = zeros(length(Xsection_data),1);
        elseif any(strcmp(Xsection,'Circular'))
            angle_circular = 0:pi/90:2*pi;
            x_circular = cos(angle_circular)/2 + 0.5;
            y_circular = sin(angle_circular)/2;
            Xsection_data = [x_circular.' y_circular.' zeros(length(x_circular),1)];
        elseif any(strcmp(Xsection,'Rectangular'))
            Xsection_data = [0 -beam(i_beam).t_beam/2; 0 beam(i_beam).t_beam/2; 1 beam(i_beam).t_beam/2; 1 -beam(i_beam).t_beam/2; 0 -beam(i_beam).t_beam/2];
            Xsection_data(:,3) = zeros(length(Xsection_data),1);
        end
        for i_element = 1:beam(i_beam).n_element
            Xsection_0 = beam(i_beam).element(i_element).C_d0'*[Xsection_data(:,3) beam(i_beam).element(i_element).c*((Xsection_data(:,1)-0.5)-(beam(i_beam).yCM-0.5)) beam(i_beam).element(i_element).c*Xsection_data(:,2)].';
            r0_Xsection_0_def = (r(i_element,:) + Xsection_0.').';
            r1_Xsection_0_def = (r(i_element+1,:) + Xsection_0.').';
            X3D_def(:,2*i_element-1:2*i_element) = [r0_Xsection_0_def(1,:).' r1_Xsection_0_def(1,:).'];
            Y3D_def(:,2*i_element-1:2*i_element) = [r0_Xsection_0_def(2,:).' r1_Xsection_0_def(2,:).'];
            Z3D_def(:,2*i_element-1:2*i_element) = [r0_Xsection_0_def(3,:).' r1_Xsection_0_def(3,:).'];
        end
        fig1 = surf(X3D_def,Y3D_def,Z3D_def,'facecolor',[1 0.5 0],'linestyle','none');
        hold on
        plot3(X3D_def(:),Y3D_def(:),Z3D_def(:),'color',[0.5 0.5 0.5]);
        xlabel('x [m]','interpreter','latex')
        ylabel('y [m]','interpreter','latex')
        zlabel('z [m]','interpreter','latex')
%         title('Equilibrium Solution','interpreter','latex')
        grid on
        axis equal

        if ~isempty(varargin) && isnumeric(varargin{end})
            element_positionCM(X_r,DoF);
            r_ROM = [beam(i_beam).r0(1,:);beam(i_beam).r1];
            rCM_ROM = beam(i_beam).rCM;
            r_ROM(:,2) = [rCM_ROM(1,2);rCM_ROM(:,2)];
            for i_element = 1:beam(i_beam).n_element
                Xsection_0 = beam(i_beam).element(i_element).C_d0'*[Xsection_data(:,3) beam(i_beam).element(i_element).c*((Xsection_data(:,1)-0.5)-(beam(i_beam).yCM-0.5)) beam(i_beam).element(i_element).c*Xsection_data(:,2)].';
                r0_Xsection_0_def = (r_ROM(i_element,:) + Xsection_0.').';
                r1_Xsection_0_def = (r_ROM(i_element+1,:) + Xsection_0.').';
                X3D_def(:,2*i_element-1:2*i_element) = [r0_Xsection_0_def(1,:).' r1_Xsection_0_def(1,:).'];
                Y3D_def(:,2*i_element-1:2*i_element) = [r0_Xsection_0_def(2,:).' r1_Xsection_0_def(2,:).'];
                Z3D_def(:,2*i_element-1:2*i_element) = [r0_Xsection_0_def(3,:).' r1_Xsection_0_def(3,:).'];
            end
            fig3 = surf(X3D_def,Y3D_def,Z3D_def,'facecolor',[0.3010 0.7450 0.9330],'linestyle','none');
            plot3(X3D_def(:),Y3D_def(:),Z3D_def(:),'color',[0.5 0.5 0.5]);
            legend([fig1 fig3],'FOM ($n=28$)','ROM ($r=12$)','location','northeast','FontSize',14,'Interpreter','latex')
        end
    end
end
end