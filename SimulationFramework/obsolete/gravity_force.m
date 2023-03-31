function Qe = generalized_force(X,varargin)
%POLYGEOM Geometry of a planar polygon
%
%   generalized_force(X) returns the generalized forces acting upon the 
%   system.
%
%   [ GEOM, INER, CPMO ] = POLYGEOM( X, Y ) returns
%   area, centroid, perimeter and area moments of 
%   inertia for the polygon.
%   GEOM = [ area   X_cen  Y_cen  perimeter ]
%   INER = [ Ixx    Iyy    Ixy    Iuu    Ivv    Iuv ]
%     u,v are centroidal axes parallel to x,y axes.
%   CPMO = [ I1     ang1   I2     ang2   J ]
%     I1,I2 are centroidal principal moments about axes
%         at angles ang1,ang2.
%     ang1 and ang2 are in radians.
%     J is centroidal polar moment.  J = I1 + I2 = Iuu + Ivv
 
% Author: Vitor Borges Santos - borgessv93@gmail.com
% Date: 15/04/22


global beam
g = 9.8067;
Jq = lineariza_complexstep(@(q) element_position(q),X);

if any(strcmp(varargin{:},'GravityOn')) || ~exist('varargin','var')
    for i_beam = 1:length(beam)
        n = beam(i_beam).n_element;
        for i_element = 1:n
            Fg_vec(:,i_element) = [0 0 -beam(i_beam).element(i_element).m*g];
            for i_q = 1:3*n
                Qg_element(i_element,i_q) = dot(Fg_vec(:,i_element),[Jq(i_element,i_q) Jq(n+i_element,i_q) Jq(2*n+i_element,i_q)]);
            end
        end
        Qg_beam(i_beam,:) = sum(Qg_element,1);
    end
    Qg = sum(Qg_beam,1);
elseif any(strcmp(varargin{:},'GravityOff'))
end

end