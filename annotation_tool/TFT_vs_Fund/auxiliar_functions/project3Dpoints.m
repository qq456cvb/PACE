function Corresp=project3Dpoints(Points3D,Pcam)
%PROJECT3DPOINTS Projects a set of N 3D points onto M images.
%
%  Input arguments:
%  Points3D   - 3xN-matrix of 3D points.
%  Pcam       - 1xM-cell of projection 3x4-matrices.
%
%  Output arguments:
%  Corresp    - 2MxN-matrix with the image points in each image.

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


M=size(Pcam,2);% number of cameras
N=size(Points3D,2); %number of points to project

Corresp=zeros(2*M,N);
for m=1:M
    x=Pcam{m}*[Points3D;ones(1,N)];
    Corresp(2*(m-1)+(1:2),:)=bsxfun(@rdivide,x(1:2,:),x(3,:));
end
end