function [F,iter]=optimF(p1,p2)
%OPTIMF Optimal estimation of the Fundamental matrix.
%
%  Computation of the fundamental matrix from corresponding points in two
%  images using Gauss-Helmert optimization, initilized by the linear 
%  solution.
%
%  Input arguments:
%  p1, p2  - 3xN or 2xN arrays of image points in image 1 and 2
%           respectively, in homogeneous or cartesian coordinates.
%
%  Output arguments:
%  F       - 3x3 array, the fundamental matrix
%  iter    - number of iterations needed in GH algorithm to reach minimum
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
[x1,Normal1]=Normalize2Ddata(p1);
[x2,Normal2]=Normalize2Ddata(p2);

% initial estimate of F
F=linearF(x1,x2); F=F/sqrt(sum(F(1:9).^2));

% projection matrices from F and 3d points
[U,~,~]=svd(F); epi21=U(:,3);
P1=eye(3,4);
P2=[crossM(epi21)*F epi21];
points3D=triangulation3D({P1,P2},[x1;x2]);

% refinement using GH on F parameters
p1_est=P1*points3D; p1_est=p1_est(1:2,:)./repmat(p1_est(3,:),2,1);
p2_est=P2*points3D; p2_est=p2_est(1:2,:)./repmat(p2_est(3,:),2,1);
p=reshape(F,9,1);
x=reshape([x1(1:2,:);x2(1:2,:)],4*N,1);
x_est=reshape([p1_est;p2_est],4*N,1);
y=zeros(0,1);
P=eye(4*N);
[~,p_opt,~,iter]=Gauss_Helmert(@constraintsGH_F,x_est,p,y,x,P);

% recover parameters
F=reshape(p_opt,3,3);

% Undo normalization
F=Normal2.'*F*Normal1;

% singularity constraint
[U,D,V]=svd(F); D(3,3)=0;
F=U*D*V.';

end


%%% constraints and parameters for the optimisation of the fundamental
%%% matrix with Gauss-Helmert
function [f,g,A,B,C,D]=constraintsGH_F(x,p,~)
% Gauss-Helmert constraints for fundamental matrix optimization
N=size(x,1)/4;
x=reshape(x,4,N);

F=reshape(p,3,3);

g=[det(F); sum(F(1:9).^2)-1];

C=[ F(5)*F(9) - F(6)*F(8), F(6)*F(7) - F(4)*F(9), F(4)*F(8) - F(5)*F(7),...
    F(3)*F(8) - F(2)*F(9), F(1)*F(9) - F(3)*F(7), F(2)*F(7) - F(1)*F(8),...
    F(2)*F(6) - F(3)*F(5), F(3)*F(4) - F(1)*F(6), F(1)*F(5) - F(2)*F(4);
    2*reshape(F,9,1).'];

f=zeros(N,1);
A=zeros(N,9);
B=zeros(N,4*N);

for i=1:N
    x1=[x(1:2,i);1]; x2=[x(3:4,i);1];
    f(i,:)=x2.'*F*x1;
    A(i,:)=[x1(1)*x2(1), x1(1)*x2(2), x1(1), x1(2)*x2(1), x1(2)*x2(2), x1(2), x2(1), x2(2), 1];
    B(i,4*(i-1)+1:4*(i-1)+4)=[ F(3) + F(1)*x2(1) + F(2)*x2(2), F(6) + F(4)*x2(1) + F(5)*x2(2),...
        F(7) + F(1)*x1(1) + F(4)*x1(2), F(8) + F(2)*x1(1) + F(5)*x1(2)];
end
D=zeros(2,0);
end

