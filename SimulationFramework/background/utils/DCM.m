function C = DCM(n,mu_rad)

if length(n) > 1
    n = n/norm(n);
    C = (1 - cos(mu_rad))*(n*n.') + cos(mu_rad)*eye(3) - sin(mu_rad)*skew(n);
    
else
    switch n
        case 1
            C = [     1        0            0
                      0   cos(mu_rad)   sin(mu_rad)
                      0  -sin(mu_rad)   cos(mu_rad)];
        case 2
            C = [ cos(mu_rad)     0     -sin(mu_rad)
                      0           1          0
                  sin(mu_rad)     0      cos(mu_rad)];
        case 3
            C = [  cos(mu_rad)   sin(mu_rad)     0
                  -sin(mu_rad)   cos(mu_rad)     0
                        0             0          1];
    end
end
end


function v_tilde = skew(v)

v_tilde = [  0    -v(3)    v(2)
            v(3)    0     -v(1)
           -v(2)   v(1)     0  ];
end