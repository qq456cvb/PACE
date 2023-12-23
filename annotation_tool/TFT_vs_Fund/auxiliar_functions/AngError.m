function [rot_err,t_err]=AngError(R_t_true,R_t_est)
%ANGERROR Computation of rotation and translation errors.
%

% Copyright (c) 2017 Laura F. Julia <laura.fernandez-julia@enpc.fr>
% All rights reserved.
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

R_true=R_t_true(:,1:3);     t_true=R_t_true(:,4);
R_est=R_t_est(:,1:3);       t_est=R_t_est(:,4);

% determine angle difference between rotations
rot_err=abs(180*acos((trace(R_true.'*R_est)-1)/2)/pi);

% determine angle difference between translations
t_err=abs(180*acos(dot( t_est/norm(t_est), t_true/norm(t_true) ))/pi);
end
