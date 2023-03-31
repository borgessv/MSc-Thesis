function status = ode_progress(t,X,flag,varargin)

persistent t_start tfinal t0_iter tinitial aux dt

if any(strcmp(varargin,'PlotProgress'))
    if any(strcmp(flag,'init'))
        figure
        status_plot = odeplot(t,X,flag);
        title('Simulation Progress','interpreter','latex')
        xlabel('t [s]','interpreter','latex')
        ylabel('State Variables','interpreter','latex')
        set(gcf,'color','w')
        grid on
    else
        status_plot = odeplot(t,X,flag);
    end

    if status_plot == 1
        status = 1;
        prompt = '\n\nThe simulation has been interrupted! Do you wish to run the animation code anyway (y/n)? ';
        answer = input(prompt,'s');
        if strcmp(answer,'Y') || strcmp(answer,'y') || strcmp(answer,'YES') || strcmp(answer,'yes')
            clear progressbar
            return
        elseif strcmp(answer,'n') || strcmp(answer,'N') || strcmp(answer,'NO') || strcmp(answer,'no')
            error('The code has been forced to stop!!')
        end
    else
    end
else
end

if nargin < 3 || isempty(flag)
    if aux == 1
        dt = t;
    end

    t_iter = toc(t0_iter);
    if t_iter > 0.2 || aux == 1
        if t(end) == tfinal
            t_end = toc(t_start);
            progressbar(t(end)/tfinal*100,t_end);
        else
            t_rem = t_iter*(length(tinitial:dt:tfinal) - find(t(end)==tinitial:dt:tfinal));
            progressbar(t(end)/tfinal*100,t_rem);
        end
        % terminate if less than one second elapsed
    else
        if t(end) == tfinal
            t_end = toc(t_start);
            progressbar(t(end)/tfinal*100,t_end);
        else
        end

    end
    status = 0;
    aux = 0;
    % initialization / end
else
    switch(flag)
        case 'init'               % odeprint(tspan,y0,'init')
            progressbar('solving ODEs...');
            tfinal = t(end);
            tinitial = t(1);
            %t_iter = tic;
            t_start = tic;
            aux = 1;

        case 'done'
            status = 0;
    end
end
t0_iter = tic;
end
