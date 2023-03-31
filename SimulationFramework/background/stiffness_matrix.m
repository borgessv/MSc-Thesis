function K = stiffness_matrix(DoF,disp_progress)

if any(strcmp(disp_progress,'True'))
    progressbar('creating stiffness matrix...')
end
global beam
n_beam = length(beam);

for i_beam = 1:n_beam
    syms x
    xtest = [beam(i_beam).element(1).x0_element; vertcat(beam(i_beam).element.x1_element)];
    beam(i_beam).K = [];
    % figure
    % plot(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.EIyy),'ob')
    % hold on
    % plot(0:0.1:16,EIyy(0:0.1:16),'-r')

    % Static solution of Euler-Bernoulli's beam equation - out-of-plane bending:
    if any(strcmp(DoF,'OutBend'))
        syms z(x)
        if beam(i_beam).n_element < 2
            p_EIyy = beam(i_beam).element.EIyy;
        elseif beam(i_beam).n_element < 3
            p_EIyy = polyfit(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.EIyy),1);
        else
            p_EIyy = polyfit(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.EIyy),2);
        end
        EIyy(x) = poly2sym(p_EIyy,x);
        dz = diff(z,x,1);
        d2z = diff(z,x,2);
        d3z = diff(z,x,3);
        if i_beam == 1
            BC = [z(0)==0 dz(0)==0 d2z(beam(i_beam).L)==0 d3z(beam(i_beam).L)==-1/EIyy(beam(i_beam).L)];
        else
            BC = [z(0)==beam(i_beam-1).z(end) dz(0)==beam(i_beam-1).dz(end) d2z(beam(i_beam).L)==0 d3z(beam(i_beam).L)==-1/EIyy(beam(i_beam).L)];
        end
        z(x) = dsolve(-EIyy*diff(z,x,4) == 0,BC);
        dz(x) = diff(z,x,1);
        tauz(x) = -EIyy*diff(z,x,2);
        ztest = double(z(xtest));
        tauztest = double(tauz(xtest));
        beam(i_beam).z = ztest;
        beam(i_beam).dz = double(dz(xtest));

        thetaz = zeros(1,length(xtest)-1);
        Kz = zeros(1,length(xtest)-1);
        for i = 2:length(xtest)
            thetaz(i) = asin((ztest(i) - ztest(i-1))/beam(i_beam).L_element) ;
            Kz(i) = abs(tauztest(i-1)/(thetaz(i) - thetaz(i-1)));
        end
        Kz = diag(Kz(2:end));
        beam(i_beam).K = blkdiag(beam(i_beam).K,Kz);
    end

    % Static solution of Euler-Bernoulli's beam equation - in-plane bending:
    if any(strcmp(DoF,'InBend'))
        syms y(x)
        if beam(i_beam).n_element < 2
            p_EIzz = vertcat(beam(i_beam).element.EIzz);
        elseif beam(i_beam).n_element < 3
            p_EIzz = polyfit(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.EIzz),1);
        else
            p_EIzz = polyfit(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.EIzz),2);
        end
        EIzz(x) = poly2sym(p_EIzz,x);
        dy = diff(y,x,1);
        d2y = diff(y,x,2);
        d3y = diff(y,x,3);
        if i_beam == 1
            BC = [y(0)==0 dy(0)==0 d2y(beam(i_beam).L)==0 d3y(beam(i_beam).L)==-1/EIzz(beam(i_beam).L)];
        else
            BC = [y(0)==beam(i_beam-1).y(end) dy(0)==beam(i_beam-1).dy(end) d2y(beam(i_beam).L)==0 d3y(beam(i_beam).L)==-1/EIzz(beam(i_beam).L)];
        end
        y(x) = dsolve(-EIzz*diff(y,x,4) == 0,BC);
        dy(x) = diff(y,x,1);
        tauy(x) = -EIzz*diff(y,x,2);
        ytest = double(y(xtest));
        tauytest = double(tauy(xtest));
        beam(i_beam).y = ytest;
        beam(i_beam).dy = double(dy(xtest));

        thetay = zeros(1,length(xtest)-1);
        Ky = zeros(1,length(xtest)-1);
        for i = 2:length(xtest)
            thetay(i) = asin((ytest(i) - ytest(i-1))/beam(i_beam).L_element) ;
            Ky(i) = abs(tauytest(i-1)/(thetay(i) - thetay(i-1)));
        end
        Ky = diag(Ky(2:end));
        beam(i_beam).K = blkdiag(beam(i_beam).K,Ky);
    end

    % Static solution of beam under axial load:
    if any(strcmp(DoF,'Axial'))
        syms del(x)
        if beam(i_beam).n_element < 2
            p_EA = beam(i_beam).element.EA;
        elseif beam(i_beam).n_element < 3
            p_EA = polyfit(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.EA),1);
        else
            p_EA = polyfit(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.EA),2);
        end
        EA(x) = poly2sym(p_EA,x);
        BC = del(0) == 0;
        del(x) = dsolve(diff(del,x,1) == 1/EA,BC);
        F(x) = del*EA/x;

        deltest = double(del(xtest));
        Ftest = double(F(xtest(2:end)));
        Ka = zeros(1,length(xtest)-1);
        for i = 2:length(xtest)
            Ka(i) = Ftest(i-1)/(deltest(i)-deltest(i-1));
        end
        Ka = diag(Ka(2:end));
        beam(i_beam).K = blkdiag(beam(i_beam).K,Ka);
    end

    % Static solution of Saint Venant's beam equation - torsion:
    if any(strcmp(DoF,'Torsion'))
        syms phi(x)
        if beam(i_beam).n_element < 2
            p_GJ = beam(i_beam).element.GJ;
        elseif beam(i_beam).n_element < 3
            p_GJ = polyfit(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.GJ),1);
        else
            p_GJ = polyfit(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.GJ),2);
        end
        GJ(x) = poly2sym(p_GJ,x);
        BC = phi(0) == 0;
        phi(x) = dsolve(diff(phi,x,1) == 1/GJ,BC);
        Tphi(x) = GJ*phi/x;
        phitest = double(phi(xtest));
        Tphitest = double(Tphi(xtest(2:end)));
        beam(i_beam).phi = phitest;

        phi = zeros(1,length(xtest)-1);
        Kphi = zeros(1,length(xtest)-1);
        phi(1) = 0;
        Kphi(1) = 0;
        for i = 2:length(xtest)
            phi(i) = phitest(i) - phitest(i-1);
            Kphi(i) = abs(Tphitest(i-1)/phi(i));
        end
        Kphi = diag(Kphi(2:end));
        %Kphi(1) = 2*Kphi(1);
        beam(i_beam).K = blkdiag(beam(i_beam).K,Kphi);
    end
end
K = blkdiag(beam.K);
if any(strcmp(disp_progress,'True'))
    progressbar('done')
end
end