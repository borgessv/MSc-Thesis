function X_tip = tip_position(beam_data,q,DoF)
global beam
load([beam_data(1:end-5),'.mat'], 'beam')
X_tip = zeros(size(q,1),3);
for i = 1:size(q,1)
    element_positionCM(q(i,:),DoF);
    r = [beam(1).r0(1,:);beam(1).r1];
    rCM = beam(1).rCM;
    r(:,2) = [rCM(1,2);rCM(:,2)];
    X_tip(i,:) = r(end,:);
end
end