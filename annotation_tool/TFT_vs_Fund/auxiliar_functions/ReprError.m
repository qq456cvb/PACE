function error=ReprError(ProjM,Corresp,Points3D)
%REPRERROE Computes the reprojection error of N points for M perspective
% cameras.
% 
%  The error is computed by projecting the space points onto the M images
%  using their projective matrices and computing the RMS of the distances
%  from the repojected point to the original image point.
%
%  Input arguments:
%  ProjM      - 1xM-cell of 3x4 projection matrices.
%  Corresp    - 2MxN matrix with the image points in each image or
%               3MxN matrix if homogeneous coordinates are used.
%  Points 3D  - 3xN matrix containing the space points if known. If the
%  (optional)   space points are not provided they will be triangulated
%               using the projective matrices.
%
%  Output arguments:
%  error      - reprojection error
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


% computing dimensions
N=size(Corresp,2);
M=size(ProjM,2);

% compute 3D triangulation if necessary
if nargin~=3
    Points3D_est=triangulation3D(ProjM,Corresp);
elseif size(Points3D,1)==3
    Points3D_est=[Points3D;ones(1,N)];
elseif size(Points3D,1)==4
    Points3D_est=Points3D;
end

% convert to affine coordinates and adapt Corresp matrix
if size(Corresp,1)==3*M
    Corresp=reshape(Corresp,3,N*M);
    Corresp=Corresp(1:2,:)./repmat(Corresp(3,:),2,1);
else
    Corresp=reshape(Corresp,2,N*M);
end

% reproject points
P=cell2mat(ProjM.');
Corresp_est=reshape(P*Points3D_est,3,M*N);
Corresp_est=Corresp_est(1:2,:)./repmat(Corresp_est(3,:),2,1);

% compute RMS of distances
error= sqrt(mean(sum((Corresp_est-Corresp).^2,1)));

end



