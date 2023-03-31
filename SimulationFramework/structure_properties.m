function [M,I,K,C] = structure_properties(beam_data,DoF,disp_progress)

global beam
beam = create_beam(beam_data,disp_progress);
n_element = sum(cat(1,beam.n_element));

[M,I] = mass_matrix(DoF,disp_progress);

K = stiffness_matrix(DoF,disp_progress);

c = 0.1;
C = damping_matrix(DoF,K,c,disp_progress);
if any(strcmp(DoF,'Torsion'))
    C(end-n_element:end) = 100*C(end-n_element:end);
end

if ispc
    filename = fullfile(sprintf(['..\\SimulationFramework\\background\\',beam_data(1:end-5),'.mat']));
else
    filename = fullfile(sprintf(['../SimulationFramework/background/',beam_data(1:end-5),'.mat']));
end
save(filename,'beam','M','I','K','C')
end