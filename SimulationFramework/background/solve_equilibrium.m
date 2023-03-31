function Xeq = solve_equilibrium(K,DoF,gravity,model,x0,disp_progress,varargin)

if length(varargin) < 3
    if any(strcmp(disp_progress,'True'))
        progressbar('solving ROM equilibrium...')
    end
    options = optimoptions('fsolve','Display','off');
    [Xeq,~,exitflag] = fsolve(@(X) equilibrium(X,K,DoF,gravity,'FOM'),x0,options);
    if any(exitflag,1:4)
        if any(strcmp(disp_progress,'True'))
            progressbar('done')
        end
    else
        error('Equilibrium solution could not be found or is not reliable!')
    end

    if any(strcmp(varargin,'PlotEq')) && any(strcmp(model,'FOM'))
        element = varargin{end};
        plot_structure(Xeq,DoF,element)
        drawnow
    end

else
    if any(strcmp(disp_progress,'True'))
        progressbar('solving ROM equilibrium...')
    end
    phi_r = varargin{1};
    Xeq_FOM = varargin{2};

    options = optimoptions('fsolve','Display','off');
    [eta_eq,~,exitflag] = fsolve(@(X) equilibrium(X,K,DoF,gravity,'ROM',phi_r),x0,options);
    if any(exitflag,1:4)
        Xeq = eta_eq;
        if any(strcmp(disp_progress,'True'))
            progressbar('done')
        end
    else
        error('Equilibrium solution could not be found or is not reliable!')
    end

    % Plotting the ROM equilibrium condition:
    if any(strcmp(varargin,'PlotEq')) && any(strcmp(model,'ROM'))
        element = varargin{end};
        Xeq_ROM = phi_r*eta_eq;
        plot_structure(Xeq_ROM,DoF,element)
        drawnow
    elseif any(strcmp(varargin,'PlotEq')) && any(strcmp(model,'BOTH'))
        element = varargin{end};
        Xeq_ROM = phi_r*eta_eq;
        plot_structure(Xeq_FOM,DoF,element,Xeq_ROM)
        drawnow
    end

end
end
