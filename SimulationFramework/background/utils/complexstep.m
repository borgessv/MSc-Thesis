function A = complexstep(fun,vec)
  Ncol = length(vec);
  Nlin = length(fun(vec));
  delta = 1e-100;
  A = zeros(Nlin, Ncol);
  for  i = 1:Ncol
      Delta = zeros(Ncol,1);
      Delta(i) = 1i*delta;
      A(:,i) = imag(fun(vec+Delta))/(delta);
  end
end