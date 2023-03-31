function print_header(model,DoF)

global beam
if any(strcmp(model,'FOM'))
    fprintf('<strong>Model</strong>: Full-Order \n<strong>Degrees of Freedom</strong>: ')
    fprintf('%s, ',DoF{1:end-1})
    fprintf('%s\n',DoF{end})
    fprintf('<strong>Number of elements</strong>: %d\n',sum(beam.n_element))
elseif any(strcmp(model,'ROM'))
    fprintf('<strong>Model</strong>: Reduced-Order \n<strong>Degrees of Freedom</strong>: ')
    fprintf('%s, ',DoF{1:end-1})
    fprintf('%s\n',DoF{end})
    fprintf('<strong>Number of elements</strong>: %d\n',sum(beam.n_element))
elseif any(strcmp(model,'BOTH'))
    fprintf('<strong>Model</strong>: Full-Order and Reduced-Order \n<strong>Degrees of Freedom</strong>: ')
    fprintf('%s, ',DoF{1:end-1})
    fprintf('%s\n',DoF{end})
    fprintf('<strong>Number of elements</strong>: %d\n',sum(beam.n_element))
end
end