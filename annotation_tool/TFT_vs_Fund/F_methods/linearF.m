function F=linearF(p1,p2)
%LINEARF Linear estimation of the Fundamental matrix
%
%  Computation of the fundamental matrix from corresponding points in two
%  images using linear equations given by epipolar constraints
%
%  Input arguments:
%  p1, p2  - 3xN or 2xN arrays of image points in image 1 and 2
%           respectively, in homogeneous or cartesian coordinates.
%
%  Output arguments:
%   F      - 3x3 array, the fundamental matrix
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


N=size(p1,2);

% Minimum number of correspondences is 8
if N~=size(p2,2) || N<8
    error('At least 8 correspondences are necessary to compute the fundamental matrix linearly\n');
end

if size(p1,1)==3
    p1=p1(1:2,:)./repmat(p1(3,:),2,1);
    p2=p2(1:2,:)./repmat(p2(3,:),2,1);
end

% Normalization of the data
[p1,Normal1]=Normalize2Ddata(p1(1:2,:));
[p2,Normal2]=Normalize2Ddata(p2(1:2,:));

A=zeros(N,9);
for i=1:N
    x1=p1(1:2,i); x2=p2(1:2,i);
    A(i,:)=[x1(1)*x2(1), x1(1)*x2(2), x1(1), x1(2)*x2(1),...
        x1(2)*x2(2), x1(2), x2(1), x2(2), 1];
end
[~,~,V]=svd(A);
F=reshape(V(:,size(V,2)),3,3);

% Undo normalization
F=Normal2.'*F*Normal1;

% singularity constraint
[U,D,V]=svd(F); D(3,3)=0;
F=U*D*V.';

end

