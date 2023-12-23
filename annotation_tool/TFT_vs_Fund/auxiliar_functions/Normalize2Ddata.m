function [new_points,N_matrix]=Normalize2Ddata(points)
%NORMALIZE2DDATA Isometric Normalization of 2D points
%
%  Given a set of points in R^2, outputs a normalization matrix that, applied
%  to the points (in homogeneous coordinates), transforms them into having 
%  mean (0,0) and mean distance to the center equal to sqrt(2).
%
%  Input arguments:
%  points      - 2xn-vector of n points of dimension 2
%
%  Output arguments:
%  N           - isometric normalization 3x3-matrix
%  new_points  - 3xn-vector of the n normalized points of dimension 2 in
%                homogeneous coordinates

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


n=size(points,2);
points0=mean(points,2);
norm0=mean(sqrt(sum((points-repmat(points0,1,n)).^2,1)));
N_matrix=diag([sqrt(2)/norm0;sqrt(2)/norm0;1]);
N_matrix(1:2,3)=-sqrt(2)*points0/norm0;

new_points=N_matrix(1:2,:)*[points;ones(1,n)];