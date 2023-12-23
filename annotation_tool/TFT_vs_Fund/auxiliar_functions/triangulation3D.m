function space_points=triangulation3D(Pcam,image_points)
%TRIANGULATION3D Triangulation of N 3d points from their image projections
% in M>1 images. The triangulation is initially computed by the DLT algorithm.
%
%  Input arguments:
%  Pcam         - M-cell of projection 3x4-matrices
%  image_points - 2MxN-matrix with the image points in each image OR
%                 3MxN-matrix with the image points in each image with
%                 homogeneous coord
%  Output arguments:
%  space_points - 4xN-array containing the 3d estimated positions (in
%                 homogeneous coordinates) of the image points.
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


M=size(Pcam,2); % number of images
if M<2
    return
end

N=size(image_points,2); % number of points
switch size(image_points,1)
    case 2*M
        % euclidean coord
    case 3*M
        % homogeneous coord
        aux=reshape(image_points,3,N*M);
        aux=aux(1:2,:)./repmat(aux(3,:),2,1);
        image_points=reshape(aux,2*M,N);
    otherwise
        return
end

space_points=zeros(4,N);
for n=1:N
    corresp_n=image_points(:,n);
    
    % DLT solution
    ls_matrix=zeros(2*M,4); %linear system matrix
    for i=1:M
        point=corresp_n(2*(i-1)+1:2*(i-1)+2,:);
        ls_matrix(2*(i-1)+1:2*(i-1)+2,:)=...
            [0 -1 point(2); 1 0 -point(1)]*Pcam{i};
    end
    [~,~,V]=svd(ls_matrix);
    PointX=V(:,4);
    space_points(:,n)=PointX;
end


end