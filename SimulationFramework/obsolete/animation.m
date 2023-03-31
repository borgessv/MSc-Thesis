function animation(T,X)

n_dof = size(X,2)/2;
n_frame = length(T);
% test1 = linspace(0,pi/6,20).';
% test2 = linspace(pi/6,pi/4,10).';
% test3 = linspace(pi/4,pi/3,10).';
% test4 = linspace(0,pi/6,10).';
% t = sin(linspace(0,2*pi,n_frame));

% % Video settings:
% v = VideoWriter('video.avi');
% v.Quality = 100;
% v.FrameRate = 50;
% open(v);

% Figure settings:
f = figure('visible','off');
% f.WindowState = 'maximized';
set(gcf, 'Position',  [250, 42, 750, 645])
set(gcf,'color','w');

% Gif settings:
filename = 'simulation_test1.gif';

progressbar('loading animation...');
frame = struct('cdata',0,'colormap',0);
im = cell(n_frame,1);
for i = 1:n_frame
    t1 = tic;
%     q1 = [t(i)*0*linspace(0,0.2,10).';0*t(i)*test4;0*t(i)*test4;abs(t(i)*test4)];
%     q2 = [t(i)*0*linspace(0,0.2,20).';0*t(i)*test1;0*t(i)*test1;abs(t(i)*test1)];
%     q3 = [t(i)*0*linspace(0,0.2,10).';t(i)*0*test2;0*t(i)*test2;abs(t(i)*test2)];
%     q4 = [t(i)*0*linspace(0,0.2,10).';0*t(i)*test2;0*t(i)*test2;abs(t(i)*test4)];
%     q5 = [t(i)*linspace(0,0.3,20).';-2*t(i)*test1;t(i)*-test1;abs(t(i)*0.5*test1)];
%     q6 = [t(i)*linspace(0,0.3,10).';-2*t(i)*test4;t(i)*-test4;abs(t(i)*0.5*test4)];
%     q7 = [t(i)*linspace(0,0.3,10).';-2*t(i)*test4;t(i)*-test4;-abs(t(i)*0.5*test4)];
%     q8 = [t(i)*linspace(0,0.3,20).';2*t(i)*test1;t(i)*-test1;-abs(t(i)*0.5*test1)];
%     q9 = [t(i)*linspace(0,0.3,10).';2*t(i)*test4;t(i)*-test4;-abs(t(i)*0.5*test4)];
%     q10 = [t(i)*linspace(0,0.3,10).';2*t(i)*test4;t(i)*-test4;-abs(t(i)*0.5*test4)];
%     q11 = [t(i)*linspace(0,0.2,10).';-0*t(i)*test4;0*-t(i)*test4;abs(t(i)*0*test4)];
%     q12 = [t(i)*linspace(0,0.2,10).';0*t(i)*test4;0*-t(i)*test4;-abs(t(i)*0*test4)];
%     q13 = [t(i)*linspace(0,0.1,10).';t(i)*0*test4;abs(0*t(i)*test4);0*t(i)*test4];
%     Q(1).q = q1;
%     Q(2).q = q2;
%     Q(3).q = q3;
%     Q(4).q = q4;
%     Q(5).q = q5;
%     Q(6).q = q6;
%     Q(7).q = q7;
%     Q(8).q = q8;
%     Q(9).q = q9;
%     Q(10).q = q10;
%     Q(11).q = q11;
%     Q(12).q = q12;
%     Q(13).q = q13;
r = element_position(X(i,n_dof+1:end));
plot3(r(:,1),r(:,2),r(:,3),'-|k','markersize',2,'markerfacecolor','k')
hold on
plot3(rCM(:,1),rCM(:,2),rCM(:,3),'sr','markersize',3,'markerfacecolor','r')
xlabel('x [m]')
ylabel('y [m]')
zlabel('z [m]')
grid on
axis equal
    %plot_structure_test(Q,'beams.mat','DisplayElements','DisplayUndeformed')
    hold off
    %xlim([-25 15])
    %ylim([-35 35])
    %zlim([-25 25])
    %view(60,20)
    frame(i) = getframe(gcf);

%     % Creating .avi video:
%     writeVideo(v,frame(i));

    % Creating .gif:
    im{i} = frame2im(frame(i));
    [A,map] = rgb2ind(im{i},256);
    if i == 1
        imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',0);
    elseif any(i == 0:4:n_frame)
        imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',0);
    end

    t2 = toc(t1);
    t_rem = t2*(n_frame - i);
    progressbar(i/n_frame*100,t_rem);
    %waitbar(i/n_frame)

%     fprintf('Remaining time: %.2f',t_rem)
end
% close(v)
web('simulation.gif')
