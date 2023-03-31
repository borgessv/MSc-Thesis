function C = damping_matrix(DoF,K,c,disp_progress)
if any(strcmp(disp_progress,'True'))
    progressbar('creating damping matrix...')
end

global beam
C = c.*K;
if any(strcmp(DoF,'Torsion'))
    C(end-sum(beam.n_element)+1:end,end-sum(beam.n_element)+1:end) = 0.05*C(end-sum(beam.n_element)+1:end,end-sum(beam.n_element)+1:end);
end
if any(strcmp(disp_progress,'True'))
    progressbar('done')
end

end