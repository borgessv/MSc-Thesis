function K = stiffness_matrix(DoF)

global beam
n_beam = length(beam);
for i_beam = 1:n_beam
    syms x z(x) y(x) del(x)
    p_EIyy = polyfit(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.EIyy),2);
    EIyy(x) = poly2sym(p_EIyy,x);
    p_EIzz = polyfit(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.EIzz),2);
    EIzz(x) = poly2sym(p_EIzz,x);
    p_EA = polyfit(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.EA),2);
    EA(x) = poly2sym(p_EA,x);
    % figure
    % plot(vertcat(beam(i_beam).element.x1_element),vertcat(beam(i_beam).element.EIyy),'ob')
    % hold on
    % plot(0:0.1:16,EIyy(0:0.1:16),'-r')

    % Static solution of Euler-Bernoulli's beam equation - out-of-plane bending:
    dz = diff(z,x,1);
    d2z = diff(z,x,2);
    d3z = diff(z,x,3);
    if i_beam == 1
        BC = [z(0)==0 dz(0)==0 d2z(beam(i_beam).L)==0 d3z(beam(i_beam).L)==-1/EIyy(beam(i_beam).L)];
    else
        BC = [z(0)==beam(i_beam-1).z(end) dz(0)==beam(i_beam-1).dz(end) d2z(beam(i_beam).L)==0 d3z(beam(i_beam).L)==-1/EIyy(beam(i_beam).L)];
    end
    z(x) = dsolve(-EIyy*diff(z,x,4) == 0,BC);

    % Static solution of Euler-Bernoulli's beam equation - in-plane bending:
    dy = diff(y,x,1);
    d2y = diff(y,x,2);
    d3y = diff(y,x,3);
    BC = [y(0)==0 dy(0)==0 d2y(beam(i_beam).L)==0 d3y(beam(i_beam).L)==-1/EIzz(beam(i_beam).L)];
    y(x) = dsolve(-EIzz*diff(y,x,4) == 0,BC);

    % Static solution of beam under axial load:
    BC = del(0) == 0;
    del(x) = dsolve(diff(del,x,1) == 1/EA,BC);

    xtest = [beam(i_beam).element(1).x0_element; vertcat(beam(i_beam).element.x1_element)];
    ztest = double(z(xtest));
    ytest = double(y(xtest));
    deltest = double(del(xtest));
    tauz(x) = -EIyy*diff(z,x,2);
    tauy(x) = -EIzz*diff(y,x,2);
    F(x) = del*EA/x;
    dz(x) = diff(z,x,1);
    %dztest = double(dz(xtest));
    beam(i_beam).z = ztest;
    beam(i_beam).dz = double(dz(xtest));

    tauztest = double(tauz(xtest));
    tauytest = double(tauy(xtest));
    Ftest = double(F(xtest(2:end)));

    thetaz = zeros(1,length(xtest)-1);
    thetay = zeros(1,length(xtest)-1);
    Kz = zeros(1,length(xtest)-1);
    Ky = zeros(1,length(xtest)-1);
    Ka = zeros(1,length(xtest)-1);
    thetaz(1) = 0;
    Kz(1) = 0;
    thetay(1) = 0;
    Ky(1) = 0;
    for i = 2:length(xtest)
        thetaz(i) = asin((ztest(i) - ztest(i-1))/beam(i_beam).L_element) ;
        Kz(i) = abs(tauztest(i-1)/(thetaz(i) - thetaz(i-1)));
        thetay(i) = asin((ytest(i) - ytest(i-1))/beam(i_beam).L_element) ;
        Ky(i) = abs(tauytest(i-1)/(thetay(i) - thetay(i-1)));
        Ka(i-1) = Ftest(i-1)/deltest(i);
    end

    Kz = diag(Kz(2:end));
    Ky = diag(Ky(2:end));
    Ka = diag(Ka);
    %Kt = zeros(size(Ka));
    beam(i_beam).K = blkdiag(Kz,Ky,Ka);
end
K = blkdiag(beam.K);
end